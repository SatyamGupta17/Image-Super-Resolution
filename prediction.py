import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, PReLU, BatchNormalization, UpSampling2D, Input, add

# Define Residual Block
def res_block(ip):
    res_model = Conv2D(64, (3,3), padding="same")(ip)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1,2])(res_model)
    res_model = Conv2D(64, (3,3), padding="same")(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    return add([ip, res_model])

# Define Upscale Block
def upscale_block(ip):
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D(size=2)(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    return up_model

# Create Generator Model with dynamic input size
def create_gen(gen_ip, num_res_block=16):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)

    temp = layers
    for _ in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers, temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9,9), padding="same")(layers)
    return Model(inputs=gen_ip, outputs=op)

# Predict on original size LR image and save results
def predict_original_size(generator, lr_img, save_dir="result", index=10):
    os.makedirs(save_dir, exist_ok=True)

    lr_input = np.expand_dims(lr_img, axis=0)
    sr = generator.predict(lr_input)[0]
    sr = np.clip(sr, 0, 1)

    # Save LR and SR images
    plt.imsave(os.path.join(save_dir, f"lr_image_{index}.png"), lr_img)
    plt.imsave(os.path.join(save_dir, f"sr_image_{index}.png"), sr)

    # Plot and save side by side comparison
    plt.figure(figsize=(15, 5))
    titles = ['Low Resolution', 'Super Resolved']
    imgs = [lr_img, sr]

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(imgs[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"comparison_{index}.png"))
    plt.close()

if __name__ == "__main__":
    # Path to your LR image
    lr_path = "HR/im1.jpg"
    # Path to your trained generator weights
    MODEL_PATH = "new_GAN/gen_e_20.h5"

    # Load LR image without resizing, convert BGR to RGB and normalize
    lr_img = cv2.imread(lr_path)
    if lr_img is None:
        raise FileNotFoundError(f"LR image not found: {lr_path}")

    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB) / 255.0
    print(f"Original LR image shape: {lr_img.shape}")

    # Create generator model with input shape matching LR image
    lr_ip = Input(shape=(lr_img.shape[0], lr_img.shape[1], 3))
    generator = create_gen(lr_ip, num_res_block=16)

    # Load pretrained weights
    generator.load_weights(MODEL_PATH)

    # Predict and save results
    predict_original_size(generator, lr_img, index=1)

    print("Prediction done and images saved in 'result/' folder.")
