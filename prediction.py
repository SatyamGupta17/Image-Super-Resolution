import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, PReLU, BatchNormalization, UpSampling2D, Input, add
import cloudinary
import cloudinary.uploader

# Configuration       
cloudinary.config( 
    cloud_name = "djtudleky", 
    api_key = "343741845783828", 
    api_secret = "Fsj4QJmCF2Wh1z3pwPaqPlRNdNY", # Click 'View API Keys' above to copy your API secret
    secure=True
)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


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
def predict_original_size(generator, lr_img, image_path, save_dir="result"):
    os.makedirs(save_dir, exist_ok=True)

    # Get image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    lr_input = np.expand_dims(lr_img, axis=0)
    sr = generator.predict(lr_input)[0]
    sr = np.clip(sr, 0, 1)

    # Define image paths
    lr_path = os.path.join(save_dir, f"{image_name}.png")
    sr_path = os.path.join(save_dir, f"{image_name}_enhanced.png")
    comp_path = os.path.join(save_dir, f"{image_name}_comparison.png")

    # Save images
    plt.imsave(lr_path, lr_img)
    plt.imsave(sr_path, sr)

    # Save side-by-side comparison
    plt.figure(figsize=(15, 5))
    titles = ['Low Resolution', 'Super Resolved']
    imgs = [lr_img, sr]

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(imgs[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(comp_path)
    plt.close()

    # Upload to Cloudinary
    lr_url = cloudinary.uploader.upload(lr_path)["secure_url"]
    sr_url = cloudinary.uploader.upload(sr_path)["secure_url"]
    comp_url = cloudinary.uploader.upload(comp_path)["secure_url"]
    print("Image uploaded to Cloudinary:")
    print(f"Low Resolution Image URL: {lr_url}")
    print(f"Super Resolved Image URL: {sr_url}")
    print(f"Comparison Image URL: {comp_url}")
    return lr_url, sr_url, comp_url

if __name__ == "__main__":
    lr_path = "HR/im1.jpg"
    MODEL_PATH = "GAN/gen_e_20.h5"

    # Load image
    lr_img = cv2.imread(lr_path)
    if lr_img is None:
        raise FileNotFoundError(f"LR image not found: {lr_path}")
    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB) / 255.0
    print(f"Original LR image shape: {lr_img.shape}")

    # Create generator model
    lr_ip = Input(shape=(lr_img.shape[0], lr_img.shape[1], 3))
    generator = create_gen(lr_ip, num_res_block=16)
    generator.load_weights(MODEL_PATH)

    # Predict and upload
    lr_url, sr_url, comp_url = predict_original_size(generator, lr_img, image_path=lr_path)

    print("Prediction done. Image URLs:")
    print(f"Low-Res URL: {lr_url}")
    print(f"Super-Res URL: {sr_url}")
    print(f"Comparison URL: {comp_url}")

