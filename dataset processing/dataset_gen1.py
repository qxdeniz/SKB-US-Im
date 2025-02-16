import cv2
import numpy as np
import os


def adjust_brightness(image, brightness=0):

    return cv2.convertScaleAbs(image, alpha=1, beta=brightness)

input_folder = 'quality_classifier/images'

output_folder_dark = 'dataset - brightness/dark'
output_folder_bright = 'dataset - brightness/bright'
output_folder_normal = 'dataset - brightness/normal'

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    dark_img = adjust_brightness(img, brightness=-100)
    cv2.imwrite(os.path.join(output_folder_dark, img_name), dark_img)

    bright_img = adjust_brightness(img, brightness=100)
    cv2.imwrite(os.path.join(output_folder_bright, img_name), bright_img)

    normal_img = img  
    cv2.imwrite(os.path.join(output_folder_normal, img_name), normal_img)
