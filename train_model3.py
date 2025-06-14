import os
import cv2
import numpy as np
from keras import Model
from keras.layers import (Conv2D, PReLU, LeakyReLU, Dense, Input, add,Multiply, GlobalAveragePooling2D, Reshape, UpSampling2D, Flatten)
from keras.applications import VGG19
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Channel Attention Module
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    
    # Shared layers
    shared_layer_one = Dense(channel//ratio, activation='relu')
    shared_layer_two = Dense(channel)
    
    # Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    # Global Max Pooling
    max_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    # Add and sigmoid
    cbam_feature = add([avg_pool, max_pool])
    cbam_feature = PReLU(shared_axes=[1,2])(cbam_feature)
    
    # Multiply with input feature
    return Multiply()([input_feature, cbam_feature])

# Residual in Residual Block with Channel Attention
def RRCA_block(ip, filters=64):
    residual = ip
    
    # First conv block
    model = Conv2D(filters, (3,3), padding='same')(ip)
    model = PReLU(shared_axes=[1,2])(model)
    
    # Second conv block
    model = Conv2D(filters, (3,3), padding='same')(model)
    
    # Add channel attention
    model = channel_attention(model)
    
    # Add residual
    model = add([model, residual])
    return model

def upscale_block(ip):
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D( size = 2 )(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    
    return up_model

# Generator with RRCA blocks
def create_gen(gen_ip, num_rrca_blocks=16):
    # Initial layers
    layers = Conv2D(64, (3,3), padding='same')(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)
    
    # Long skip connection
    long_skip = layers
    
    # RRCA blocks
    for _ in range(num_rrca_blocks):
        layers = RRCA_block(layers)
    
    # Post RRCA conv
    layers = Conv2D(64, (3,3), padding='same')(layers)
    layers = add([layers, long_skip])
    
    # Upsampling blocks
    layers = upscale_block(layers)
    layers = upscale_block(layers)
    
    # Output layer
    op = Conv2D(3, (9,9), padding='same', activation='tanh')(layers)
    
    return Model(inputs=gen_ip, outputs=op)

# Discriminator (similar to SRGAN but without BN)
def create_disc(disc_ip):
    df = 64
    
    d1 = Conv2D(df, (3,3), strides=1, padding='same')(disc_ip)
    d1 = LeakyReLU(alpha=0.2)(d1)
    
    d2 = Conv2D(df, (3,3), strides=2, padding='same')(d1)
    d2 = LeakyReLU(alpha=0.2)(d2)
    
    d3 = Conv2D(df*2, (3,3), strides=1, padding='same')(d2)
    d3 = LeakyReLU(alpha=0.2)(d3)
    
    d4 = Conv2D(df*2, (3,3), strides=2, padding='same')(d3)
    d4 = LeakyReLU(alpha=0.2)(d4)
    
    d5 = Conv2D(df*4, (3,3), strides=1, padding='same')(d4)
    d5 = LeakyReLU(alpha=0.2)(d5)
    
    d6 = Conv2D(df*4, (3,3), strides=2, padding='same')(d5)
    d6 = LeakyReLU(alpha=0.2)(d6)
    
    d7 = Conv2D(df*8, (3,3), strides=1, padding='same')(d6)
    d7 = LeakyReLU(alpha=0.2)(d7)
    
    d8 = Conv2D(df*8, (3,3), strides=2, padding='same')(d7)
    d8 = LeakyReLU(alpha=0.2)(d8)
    
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(disc_ip, validity)

# Build VGG (same as before)
def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

# Combined model (same as before)
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    gen_features = vgg(gen_img)
    disc_model.trainable = False
    validity = disc_model(gen_img)
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])

# Load and preprocess data (same as before)
lr_list = os.listdir("LR")
lr_images = [cv2.cvtColor(cv2.imread("LR/"+img), cv2.COLOR_BGR2RGB) for img in lr_list]
hr_list = os.listdir("HR")
hr_images = [cv2.cvtColor(cv2.imread("HR/"+img), cv2.COLOR_BGR2RGB) for img in hr_list]

lr_images = np.array(lr_images) / 255.
hr_images = np.array(hr_images) / 255.

lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, test_size=0.33, random_state=42)

# Build models
hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])

lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

generator = create_gen(lr_ip, num_rrca_blocks=16)  # Try 5 or 10 blocks
generator.summary()

discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
discriminator.summary()

vgg = build_vgg((128,128,3))
vgg.trainable = False

gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[5e-3, 1], optimizer="adam")

# Training loop (same as before but with adjusted parameters)
batch_size = 8 
train_lr_batches = []
train_hr_batches = []
for it in range(int(hr_train.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])
    
    
epochs = 50
#Enumerate training over epochs
for e in range(epochs):
    
    fake_label = np.zeros((batch_size, 1)) # Assign a label of 0 to all fake (generated images)
    real_label = np.ones((batch_size, 1)) # Assign a label of 1 to all real images.
    
    #Create empty lists to populate gen and disc losses. 
    g_losses = []
    d_losses = []
    g_loss_log = []
    d_loss_log = [] 

    
    #Enumerate training over batches. 
    for b in tqdm(range(len(train_hr_batches))):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]
        
        fake_imgs = generator.predict_on_batch(lr_imgs)  
        discriminator.trainable = True
        d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_label) 
        discriminator.trainable = False
         
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real) 
        image_features = vgg.predict(hr_imgs)
     
        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
        
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)
    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)
    g_loss_epoch = np.mean(g_losses)
    d_loss_epoch = np.mean(d_losses)
    g_loss_log.append(g_loss_epoch)
    d_loss_log.append(d_loss_epoch) 
    
    print("Epoch : ", e+1, " Completed.." )
    if (e + 1) % 5 == 0:
        print("epoch:", e+1 ,"g_loss:", g_loss, "d_loss:", d_loss)
        generator.save("GAN2/gen_e_"+ str(e+1) +".keras")
    
    with open("training_log2.txt", "a") as log_file:
        log_file.write(f"Epoch {e+1}: G_Loss={g_loss_epoch:.5f}, D_Loss={d_loss_epoch:.5f} -> g_loss = {g_loss}, d_loss = {d_loss}\n")
