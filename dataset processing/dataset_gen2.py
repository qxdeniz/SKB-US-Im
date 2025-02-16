import cv2
import numpy as np
import os


def adjust_alpha(image, alpha):

    return cv2.convertScaleAbs(image, alpha=alpha)

input_folder = 'quality_classifier/images'
output_folder_dim = 'dataset - contrast/dim'
output_folder_m_contrast = 'dataset - contrast/much contrasting'


for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)
    print(img_path)

    dim_img = adjust_alpha(img, -5)
    cv2.imwrite(os.path.join(output_folder_dim, img_name), dim_img)

    contrast_img = adjust_alpha(img, +5)
    cv2.imwrite(os.path.join(output_folder_m_contrast, img_name), contrast_img)
