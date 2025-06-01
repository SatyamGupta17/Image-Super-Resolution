import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from keras.models import load_model 
from keras import Model
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add

def load_images(lr_dir, hr_dir, max_images = 20):
    lr_images = []
    hr_images = []
    lr_list = sorted(os.listdir(lr_dir))[:max_images]
    hr_list = sorted(os.listdir(hr_dir))[:max_images]
    for lr_name, hr_name in zip(lr_list, hr_list):
        lr = cv2.imread(os.path.join(lr_dir, lr_name))
        hr = cv2.imread(os.path.join(hr_dir, hr_name))

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB) / 255.0
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB) / 255.0

        lr_images.append(lr)
        hr_images.append(hr)

    return np.array(lr_images), np.array(hr_images)

def load_images_new(lr_dir, hr_dir, max_images=20, lr_size=(128, 128), hr_size=(512, 512)):
    lr_images = []
    hr_images = []
    lr_list = sorted(os.listdir(lr_dir))[:max_images]
    hr_list = sorted(os.listdir(hr_dir))[:max_images]

    for lr_name, hr_name in zip(lr_list, hr_list):
        lr_path = os.path.join(lr_dir, lr_name)
        hr_path = os.path.join(hr_dir, hr_name)

        lr = cv2.imread(lr_path)
        hr = cv2.imread(hr_path)

        if lr is None or hr is None:
            print(f"Warning: Unable to load {lr_name} or {hr_name}. Skipping.")
            continue

        # Resize images
        lr = cv2.resize(lr, lr_size, interpolation=cv2.INTER_CUBIC)
        hr = cv2.resize(hr, hr_size, interpolation=cv2.INTER_CUBIC)

        # Convert to RGB and normalize
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB) / 255.0
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB) / 255.0

        lr_images.append(lr)
        hr_images.append(hr)

    return np.array(lr_images), np.array(hr_images)

def res_block(ip):
    
    res_model = Conv2D(64, (3,3), padding = "same")(ip)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)
    
    res_model = Conv2D(64, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    
    return add([ip,res_model])


def upscale_block(ip):
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D( size = 2 )(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    
    return up_model

def create_gen(gen_ip, num_res_block):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)

    temp = layers

    for i in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers,temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9,9), padding="same")(layers)

    return Model(inputs=gen_ip, outputs=op)

def evaluate_model(generator, lr_images, hr_images):
    psnr_vals = []
    ssim_vals = []

    for i in range(len(lr_images)):
        lr = np.expand_dims(lr_images[i], axis=0)
        sr = generator.predict(lr)[0]
        hr = hr_images[i]

        sr = np.clip(sr, 0, 1)

        psnr_val = psnr(hr, sr, data_range=1.0)
        ssim_val = ssim(hr, sr, channel_axis=2, data_range=1.0)

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val) 
    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)

    print(f"\n Evaluation Results:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    return psnr_vals, ssim_vals

def plot_sample(lr_img, sr_img, hr_img, index): 
    # Resize LR to match HR for fair comparison
    lr_resized = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Compute PSNR and SSIM
    psnr_lr = psnr(hr_img, lr_resized, data_range=1.0)
    ssim_lr = ssim(hr_img, lr_resized, channel_axis=2, data_range=1.0)
    psnr_sr = psnr(hr_img, sr_img, data_range=1.0)
    ssim_sr = ssim(hr_img, sr_img, channel_axis=2, data_range=1.0)
    # Plot side by side with metrics
    plt.figure(figsize=(18, 8))
    imgs = [lr_img, sr_img, hr_img]
    titles = [
        f'Low Resolution\nPSNR: {psnr_lr:.2f} dB\nSSIM: {ssim_lr:.4f}',
        f'Super Resolved\nPSNR: {psnr_sr:.2f} dB\nSSIM: {ssim_sr:.4f}',
        'High Resolution (Ground Truth)'
    ]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(np.clip(imgs[i], 0, 1))
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"results/comparison_{index}.png")
    plt.close()
    
LR_DIR = "LR1"    
HR_DIR = "HR1"   
MODEL_PATH = "models/3051crop_weight_200.h5"   
# SHOW_EXAMPLES = 3     


lr_ip = Input(shape=(128, 128, 3)) 
generator = create_gen(lr_ip, num_res_block=16)
generator.load_weights(MODEL_PATH)
lr_images, hr_images = load_images_new(LR_DIR, HR_DIR, 100) 
psnr_scores, ssim_scores = evaluate_model(generator, lr_images, hr_images) 

scores = [(i, psnr_scores[i], ssim_scores[i], (psnr_scores[i])) for i in range(len(psnr_scores))]
top_10 = sorted(scores, key=lambda x: x[3], reverse=True)[:10]
print("\nTop 10 images based on PSNR + SSIM:")
for idx, ps, ss, _ in top_10:
    print(f"Index: {idx}, PSNR: {ps:.2f}, SSIM: {ss:.4f}")
    lr = lr_images[idx]
    hr = hr_images[idx]
    sr = generator.predict(np.expand_dims(lr, axis=0))[0]
    plot_sample(lr, sr, hr, f"top_{idx}")
