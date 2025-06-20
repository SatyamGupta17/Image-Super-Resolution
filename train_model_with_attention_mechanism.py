import os
import cv2
import numpy as np
import tensorflow as tf
from keras import Model, backend as K
from keras.layers import (Conv2D, PReLU, LeakyReLU, Dense, Input, add, 
                         Multiply, GlobalAveragePooling2D, Reshape, 
                         UpSampling2D, Flatten, Concatenate)
from keras.applications import VGG19
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==================== CRITICAL IMPROVEMENTS ====================

# 1. Gradient Penalty for WGAN-GP
def gradient_penalty(discriminator, real_images, fake_images, batch_size):
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    interpolated = real_images * alpha + fake_images * (1 - alpha)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]))
    return tf.reduce_mean((norm - 1.)**2)

# 2. Stabilized Channel Attention
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    
    shared_layer = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal')
    final_layer = Dense(channel, kernel_initializer='he_normal')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = final_layer(shared_layer(avg_pool))
    
    max_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = final_layer(shared_layer(max_pool))
    
    cbam_feature = add([avg_pool, max_pool])
    cbam_feature = PReLU(shared_axes=[1,2])(cbam_feature)
    return Multiply()([input_feature, cbam_feature])

# 3. Stabilized Residual Block
def RRCA_block(ip, filters=64, residual_scale=0.2):
    residual = Conv2D(filters, 1, padding='same')(ip) * residual_scale  # 1x1 conv to match dimensions
    
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(ip)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = channel_attention(x)
    return add([x, residual])  # Now shapes will match

def create_gen(gen_ip, num_rrca_blocks=16):
    # Initial convolution
    x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(gen_ip)
    x = PReLU(shared_axes=[1,2])(x)
    features = [x]
    
    # RRCA blocks with consistent filter size
    for _ in range(num_rrca_blocks):
        x = RRCA_block(x, filters=64)  # Fixed filter size
        features.append(x)
        if len(features) > 4:
            x = Concatenate()([x, features[-3]])
            x = Conv2D(64, 1, kernel_initializer='he_normal')(x)  # Project back to 64 channels
    
    # Feature fusion
    x = Conv2D(64, 1, kernel_initializer='he_normal')(Concatenate()(features[-3:]))
    
    # Upsampling
    x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.nn.depth_to_space(x, 2)
    
    x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.nn.depth_to_space(x, 2)
    
    return Model(gen_ip, Conv2D(3, 9, padding='same', activation='tanh')(x))

# 5. Stabilized Discriminator
def create_disc(disc_ip):
    def conv_block(x, filters, strides=1):
        x = Conv2D(filters, 3, strides=strides, padding='same', 
                  kernel_initializer='he_normal')(x)
        return LeakyReLU(0.2)(x)
    
    x = disc_ip
    x = conv_block(x, 64)  # No BN!
    x = conv_block(x, 64, 2)
    x = conv_block(x, 128)
    x = conv_block(x, 128, 2)
    x = conv_block(x, 256)
    x = conv_block(x, 256, 2)
    x = conv_block(x, 512)
    x = conv_block(x, 512, 2)
    
    x = Flatten()(x)
    x = Dense(1024, kernel_initializer='he_normal')(x)
    x = LeakyReLU(0.2)(x)
    return Model(disc_ip, Dense(1)(x))  # Linear activation for WGAN-GP

# ==================== TRAINING STABILIZATION ====================

# 1. WGAN Loss Functions
def discriminator_loss(real_output, fake_output, gp, lambda_gp=10):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + lambda_gp * gp

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

# 2. Learning Rate Schedule
def lr_schedule(epoch):
    if epoch < 10: return 1e-4
    elif epoch < 30: return 5e-5
    else: return 1e-5

# ==================== MAIN TRAINING LOOP ====================

# Load and preprocess data
lr_images = np.array([cv2.cvtColor(cv2.imread(f"LR/{img}"), cv2.COLOR_BGR2RGB)/255. 
                     for img in os.listdir("LR")])
hr_images = np.array([cv2.cvtColor(cv2.imread(f"HR/{img}"), cv2.COLOR_BGR2RGB)/255. 
                     for img in os.listdir("HR")])

lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, test_size=0.2)

# Build models
generator = create_gen(Input(lr_train.shape[1:]))
discriminator = create_disc(Input(hr_train.shape[1:]))
vgg = Model(inputs=VGG19(weights="imagenet", include_top=False, 
                        input_shape=hr_train.shape[1:]).inputs,
           outputs=VGG19().layers[10].output)
vgg.trainable = False

# Compile with WGAN-GP
optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, clipnorm=1.0)
gan_model = Model([generator.input, discriminator.input],
                 [discriminator(generator.output), vgg(generator.output)])
gan_model.compile(optimizer=optimizer,
                 loss=[generator_loss, 'mae'],
                 loss_weights=[1e-3, 1])

# Training
batch_size = 8
epochs = 50
os.makedirs('GAN2', exist_ok=True)

for epoch in range(epochs):
    K.set_value(gan_model.optimizer.lr, lr_schedule(epoch))
    
    for batch in tqdm(range(len(lr_train)//batch_size)):
        lr_batch = lr_train[batch*batch_size:(batch+1)*batch_size]
        hr_batch = hr_train[batch*batch_size:(batch+1)*batch_size]
        
        # Train Discriminator
        fake_images = generator.predict(lr_batch, verbose=0)
        with tf.GradientTape() as disc_tape:
            real_output = discriminator(hr_batch, training=True)
            fake_output = discriminator(fake_images, training=True)
            gp = gradient_penalty(discriminator, hr_batch, fake_images, batch_size)
            d_loss = discriminator_loss(real_output, fake_output, gp)
        
        # Train Generator (every 2 steps)
        if batch % 2 == 0:
            with tf.GradientTape() as gen_tape:
                fake_images = generator(lr_batch, training=True)
                vgg_features = vgg(hr_batch)
                g_loss, _, _ = gan_model([lr_batch, hr_batch], 
                                       [tf.ones((batch_size,1)), vgg_features])
        
        # Apply gradients with clipping
        disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        
        if batch % 2 == 0:
            gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
            optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    with open("training_log2.txt", "a") as f:
        f.write(f"Epoch {epoch+1}: G_Loss={g_loss:.5f}, D_Loss={d_loss:.5f}\n")
    # Save checkpoints
    if (epoch+1) % 5 == 0:
        generator.save(f"GAN2/gen_e_{epoch+1}.keras")
