import cv2
import numpy as np
import os

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
#3,7,15,31,51,71,101
kernel_size = 201
data_in_path = 'data/tt/1'
data_out_path = 'data/tt/{}/images_1/'.format(kernel_size)
mkdir(data_out_path)

if __name__ == "__main__":
    imgs_T = os.listdir(data_in_path)
    for image_name in imgs_T:
        if image_name == '.ipynb_checkpoints':
            continue
        img_T = cv2.imread(os.path.join(data_in_path , image_name))
        print(data_in_path + image_name)
        dst = cv2.GaussianBlur(img_T, (kernel_size, kernel_size), 0) 
        cv2.imwrite(os.path.join(data_out_path , image_name), dst)
        # cv2.waitKey()
