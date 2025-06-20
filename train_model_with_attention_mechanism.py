import os
import cv2
import numpy as np
from keras import Model, backend as K
from keras.layers import (Conv2D, PReLU, LeakyReLU, Dense, Input, add, Multiply,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape,
                          UpSampling2D, Flatten, Lambda)
from keras.applications import VGG19
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.optimizers import Adam

# Channel Attention Module (Fixed)
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal')
    shared_dense_two = Dense(channel, kernel_initializer='he_normal')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    cbam_feature = add([avg_pool, max_pool])
    cbam_feature = PReLU(shared_axes=[1, 2])(cbam_feature)

    return Multiply()([input_feature, cbam_feature])

# Residual in Residual Channel Attention Block (Fixed)
def RRCA_block(ip, filters=64):
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(ip)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = channel_attention(x)
    x = Lambda(lambda t: t * 0.2)(x)
    return add([x, ip])

def upscale_block(ip):
    x = Conv2D(256, (3, 3), padding="same")(ip)
    x = UpSampling2D(size=2)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    return x

def create_gen(gen_ip, num_rrca_blocks=16):
    x = Conv2D(64, (3, 3), padding='same')(gen_ip)
    x = PReLU(shared_axes=[1, 2])(x)
    long_skip = x

    for _ in range(num_rrca_blocks):
        x = RRCA_block(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = add([x, long_skip])

    x = upscale_block(x)
    x = upscale_block(x)

    out = Conv2D(3, (9, 9), padding='same', activation='tanh')(x)
    return Model(inputs=gen_ip, outputs=out)

def create_disc(disc_ip):
    df = 64
    x = Conv2D(df, (3, 3), strides=1, padding='same')(disc_ip)
    x = LeakyReLU(alpha=0.2)(x)

    for i in range(1, 4):
        x = Conv2D(df * (2 ** i), (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(df * (2 ** i), (3, 3), strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(disc_ip, out)

def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    gen_features = vgg(gen_img)
    disc_model.trainable = False
    validity = disc_model(gen_img)
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])

# --- Data loading and preprocessing ---
lr_list = os.listdir("LR")
hr_list = os.listdir("HR")

lr_images = [cv2.cvtColor(cv2.imread("LR/" + img), cv2.COLOR_BGR2RGB) for img in lr_list]
hr_images = [cv2.cvtColor(cv2.imread("HR/" + img), cv2.COLOR_BGR2RGB) for img in hr_list]

os.makedirs('GAN2', exist_ok=True)

# Rescale to [-1, 1] for tanh compatibility
lr_images = (np.array(lr_images) / 127.5) - 1
hr_images = (np.array(hr_images) / 127.5) - 1

lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, test_size=0.33, random_state=42)

hr_shape = hr_train.shape[1:]
lr_shape = lr_train.shape[1:]

lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

generator = create_gen(lr_ip, num_rrca_blocks=16)
generator.summary()

discriminator = create_disc(hr_ip)
opt = Adam(learning_rate=1e-4, beta_1=0.9, clipnorm=1.0)
discriminator.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
discriminator.summary()

vgg = build_vgg(hr_shape)
vgg.trainable = False

gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer=opt)

# --- Training ---
batch_size = 8
train_lr_batches = [lr_train[i:i + batch_size] for i in range(0, len(lr_train), batch_size)]
train_hr_batches = [hr_train[i:i + batch_size] for i in range(0, len(hr_train), batch_size)]

epochs = 50
for e in range(epochs):
    fake_label = np.zeros((batch_size, 1))
    real_label = np.ones((batch_size, 1))
    g_losses = []
    d_losses = []

    for b in tqdm(range(min(len(train_hr_batches), len(train_lr_batches)))):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]

        fake_imgs = generator.predict_on_batch(lr_imgs)
        discriminator.trainable = True
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_label)
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
        discriminator.trainable = False

        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        image_features = vgg.predict(hr_imgs)

        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])

        g_losses.append(g_loss)
        d_losses.append(d_loss[0])  # d_loss is (loss, acc)

    g_loss_epoch = np.mean(g_losses)
    d_loss_epoch = np.mean(d_losses)

    print(f"Epoch {e + 1}: G_Loss={g_loss_epoch:.5f}, D_Loss={d_loss_epoch:.5f}")
    with open("training_log2.txt", "a") as f:
        f.write(f"Epoch {e + 1}: G_Loss={g_loss_epoch:.5f}, D_Loss={d_loss_epoch:.5f}\n")

    if (e + 1) % 5 == 0:
        generator.save(f"GAN2/gen_e_{e + 1}.keras")
