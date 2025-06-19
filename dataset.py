import os
import cv2

train_dir = "mirflickr"
hr_dir = "HR"
lr_dir = "LR"
count = 0
os.makedirs(hr_dir, exist_ok=True)
os.makedirs(lr_dir, exist_ok=True)
 
for img in os.listdir(train_dir): 
    img_path = os.path.join(train_dir, img)
    img_arr = cv2.imread(img_path)
    if count > 15000:
        break
    count+=1
    # Skip if the image can't be read
    if img_arr is None:
        continue

    hr_img = cv2.resize(img_arr, (128, 128))
    lr_img = cv2.resize(img_arr, (32, 32))

    cv2.imwrite(os.path.join(lr_dir, img), lr_img)
    cv2.imwrite(os.path.join(hr_dir, img), hr_img)