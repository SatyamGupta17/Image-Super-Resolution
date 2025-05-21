import os
import cv2
train_dir = "mirflickr"
count = 0
for img in os.listdir(train_dir):
    if count > 5000:
        break
    count+=1
    img_arr = cv2.imread(train_dir + "/" +img)
    hr_img = cv2.resize(img_arr, (128, 128))
    lr_img = cv2.resize(img_arr, (32, 32))
    cv2.imwrite('LR' + "/" + img, lr_img)
    cv2.imwrite('HR' + '/' + img, hr_img)