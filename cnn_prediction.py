import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, InputLayer
from tensorflow.keras.optimizers import Adam
import cloudinary
import cloudinary.uploader
# define a function for peak signal to noise ration (PSNR)
cloudinary.config( 
    cloud_name = "djtudleky", 
    api_key = "343741845783828", 
    api_secret = "Fsj4QJmCF2Wh1z3pwPaqPlRNdNY", # Click 'View API Keys' above to copy your API secret
    secure=True
)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def psnr(target, ref):
    target_data = target.astype(float)
    ref_data = ref.astype(float)
    
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff **2.))
    return 20 * math.log10(255./rmse)

def mse(target, ref):
    err = np.sum((target.astype('float')- ref.astype('float'))**2)
    err /= float(target.shape[0]*target.shape[1])
    return err

def compare_image(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, win_size=3, channel_axis=-1)) 
    return scores

def load_images(lr_dir, hr_dir, max_images=20):
    lr_images, hr_images = [], []
    lr_list = sorted(os.listdir(lr_dir))[:max_images]
    hr_list = sorted(os.listdir(hr_dir))[:max_images]
    for lr_name, hr_name in zip(lr_list, hr_list):
        lr = cv2.imread(os.path.join(lr_dir, lr_name))
        hr = cv2.imread(os.path.join(hr_dir, hr_name))

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB) / 255.0
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB) / 255.0

        # Upscale LR to HR size (bicubic) before feeding SRCNN
        lr_upscaled = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

        lr_images.append(lr_upscaled)
        hr_images.append(hr)
    return np.array(lr_images), np.array(hr_images)

def evaluate_and_save(model, lr_images, hr_images, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    psnr_vals, ssim_vals = [], []
    for i in range(len(lr_images)):
        lr = np.expand_dims(lr_images[i], axis=0)  # Add batch dimension
        sr = model.predict(lr)[0]
        sr = np.clip(sr, 0, 1)
        hr = hr_images[i]

        # Compute metrics
        psnr_val = psnr(hr, sr, data_range=1.0)
        ssim_val = ssim(hr, sr, channel_axis=2, data_range=1.0)
        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)

        # Plot and save comparison image
        plot_comparison(lr_images[i], sr, hr, i, save_dir)

        print(f"Image {i}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")

    print(f"\nAverage PSNR: {np.mean(psnr_vals):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_vals):.4f}")

    return psnr_vals, ssim_vals

# define the SRCNN model
def model():
    # define model type
    SRCNN = Sequential()
    # add model layers
    SRCNN.add(Conv2D(filters = 128, kernel_size=  (9,9), kernel_initializer= 'glorot_uniform', activation = 'relu', padding = 'valid', use_bias = True, input_shape = (None, None, 1)))   
    SRCNN.add(Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer='glorot_uniform', activation = 'relu', padding = 'same', use_bias = True))
    SRCNN.add(Conv2D(filters = 1, kernel_size = (5, 5), kernel_initializer='glorot_uniform', activation = 'linear', padding = 'valid', use_bias = True))
    
    # define optimizer
    adam = Adam(learning_rate=0.0003)
    
    SRCNN.compile(optimizer = adam, loss= 'mean_squared_error', metrics = ['mean_squared_error'])
    return SRCNN

# define necessary image processing functions
def modcrop(img, scale):
    h, w = img.shape[:2]
    h = h - (h % scale)  # Ensure divisibility
    w = w - (w % scale)
    return img[:h, :w]


def shave(image, border):
    img = image[border : -border, border : -border]
    return img
    
# define main predication function
# def predict_original_size(image_path):
#     srcnn = model()
#     srcnn.load_weights('models/gen_e_20.h5')
    
#     # load the degraded and refernce images
#     path, file = os.path.split(image_path)
#     degraded = cv2.imread(f'LR2/{file}')
#     degraded_dummy = cv2.imread(image_path)
#     ref = cv2.imread(f'HR1/{file}')
    
#     # preprocess the image with modcrop
#     ref = modcrop(ref, 3)
#     degraded = modcrop(degraded, 3)
#     degraded_dummy = modcrop(degraded_dummy, 3)
    
#     #convert the image to YCrCb (red diff, blue diff) - (srcnn trained on Y channel)
#     temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
    
#     # create image slice and normalize
#     Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype = float)
#     Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255 
#     # perform super-resolution with srcnn 
#     pre = srcnn.predict(Y) 
#     # post-process output
#     pre *= 255
#     pre[pre[:] > 255] = 255
#     pre[pre[:] < 0] = 0
#     pre = pre.astype(np.uint8)
    
#     #copy Y channel back tom image and convert to BGR
#     temp = shave(temp, 6)
#     temp[:, :, 0] = pre[0, :, :, 0]
#     output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
#     # remove border from reference and degraded image
#     ref = shave(ref.astype(np.uint8), 6)
#     degraded = shave(degraded.astype(np.uint8), 6)
#     degraded_dummy = shave(degraded_dummy.astype(np.uint8), 6)
    
#     # image quality calculations
#     scores = []
#     scores.append(compare_image(degraded_dummy, ref))
#     scores.append(compare_image(output, ref))
#     lr_url = cloudinary.uploader.upload(degraded)["secure_url"]
#     sr_url = cloudinary.uploader.upload(output)["secure_url"]
#     comp_url = cloudinary.uploader.upload(ref)["secure_url"]
#     print("Image uploaded to Cloudinary:")
#     print(f"Low Resolution Image URL: {lr_url}")
#     print(f"Super Resolved Image URL: {sr_url}")
#     print(f"Comparison Image URL: {comp_url}")
#     # return lr_url, sr_url, comp_url
#     return ref, degraded, output, scores
def predict_original_size(image_path, save_dir="results"):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    srcnn = model()
    srcnn.load_weights('models/gen_e_20.h5')
    
    # Extract image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load images
    degraded = cv2.imread(f'LR2/{os.path.basename(image_path)}')
    degraded_dummy = cv2.imread(image_path)
    ref = cv2.imread(f'HR1/{os.path.basename(image_path)}')
    
    # Check if images loaded properly
    if degraded is None or degraded_dummy is None or ref is None:
        raise ValueError("One or more images failed to load")
    
    # Preprocess images
    ref = modcrop(ref, 3)
    degraded = modcrop(degraded, 3)
    degraded_dummy = modcrop(degraded_dummy, 3)
    
    # Convert to RGB for matplotlib
    degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
    degraded_dummy_rgb = cv2.cvtColor(degraded_dummy, cv2.COLOR_BGR2RGB)
    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    
    # Super-resolution processing
    temp = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
    pre = srcnn.predict(Y)
    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    
    # Create output image
    temp = shave(temp, 6)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    # Shave borders
    ref_shaved = shave(ref_rgb.astype(np.uint8), 6)
    degraded_shaved = shave(degraded_rgb.astype(np.uint8), 6)
    degraded_dummy_shaved = shave(degraded_dummy_rgb.astype(np.uint8), 6)
    output_shaved = shave(output_rgb.astype(np.uint8), 6)
    
    # Define paths for saving
    lr_path = os.path.join(save_dir, f"{image_name}.png")
    sr_path = os.path.join(save_dir, f"{image_name}_enhanced.png")
    hr_path = os.path.join(save_dir, f"{image_name}_hr.png")
    comp_path = os.path.join(save_dir, f"{image_name}_comparison.png")
    
    # Save images using plt.imsave
    plt.imsave(lr_path, degraded_dummy_shaved)
    plt.imsave(sr_path, output_shaved)
    plt.imsave(hr_path, ref_shaved)
    
    # Create and save comparison plot
    # plot_comparison(degraded_dummy_shaved, output_shaved, ref_shaved, image_name, save_dir)
    
    # Calculate scores
    # Upload to Cloudinary
    def upload_to_cloudinary(file_path, prefix):
        try:
            result = cloudinary.uploader.upload(
                file_path,
                folder="super_resolution",
                public_id=f"{prefix}_{image_name}",
                overwrite=True
            )
            return result['secure_url']
        except Exception as e:
            print(f"Error uploading {file_path}: {str(e)}")
            return None
    
    lr_url = upload_to_cloudinary(lr_path, "lr")
    sr_url = upload_to_cloudinary(sr_path, "sr")
    hr_url = upload_to_cloudinary(hr_path, "hr")
    comp_url = upload_to_cloudinary(comp_path, "comp")
    
    print("Image URLs:")
    print(f"Low-res: {lr_url}")
    print(f"Super-res: {sr_url}")
    print(f"High-res: {hr_url}")
    print(f"Comparison: {comp_url}")
    
    return { 
        'urls': {
            'lr': lr_url,
            'sr': sr_url,
            'hr': hr_url,
            'comp': comp_url
        },
        # 'scores': scores
    }
def plot_comparison(lr_img, sr_img, hr_img, idx, save_dir):
    psnr_lr = psnr(hr_img, lr_img, data_range=1.0)
    ssim_lr = ssim(hr_img, lr_img, channel_axis=2, data_range=1.0)
    psnr_sr = psnr(hr_img, sr_img, data_range=1.0)
    ssim_sr = ssim(hr_img, sr_img, channel_axis=2, data_range=1.0)

    plt.figure(figsize=(18, 8))
    imgs = [lr_img, sr_img, hr_img]
    titles = [
        f'Low Resolution (Upscaled)\nPSNR: {psnr_lr:.2f} dB\nSSIM: {ssim_lr:.4f}',
        f'Super Resolved (SRCNN)\nPSNR: {psnr_sr:.2f} dB\nSSIM: {ssim_sr:.4f}',
        'High Resolution (Ground Truth)'
    ]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(np.clip(imgs[i], 0, 1))
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"comparison_{idx}.png"))
    plt.close()

def main():
    # LR_DIR = "LR2"         # Folder with LR images
    # HR_DIR = "HR1"         # Folder with HR ground truth images
    MODEL_PATH = "models\gen_e_20.h5"  # Path to your trained SRCNN model file
    # MAX_IMAGES = 20        # Number of test images to evaluate
    # ref, degraded, output = predict_original_size('LR2/im1000.jpg')
    # print("Loading images...")
    # lr_images, hr_images = load_images(LR_DIR, HR_DIR, max_images=MAX_IMAGES)
    # print(f"Loaded {len(lr_images)} image pairs.")
    # # model = build_srcnn()
    # print("Loading trained SRCNN model...")
    # model.load_weights(MODEL_PATH)
    print(f"Degraded Image : \nPSNR: {scores[0][0]}\n MSE : {scores[0][1]}\nSSIM : {scores[0][2]}")
    print(f"Reconstructed Image : \nPSNR: {scores[1][0]}\n MSE : {scores[1][1]}\nSSIM : {scores[1][2]}")

    # print("Evaluating model and saving results...")
    # evaluate_and_save(model, lr_images, hr_images, save_dir="results")

if __name__ == "__main__":
    main()
