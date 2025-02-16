import cv2
import numpy as np

class Corrector:
    def __init__(self, image_path):
        self.image_path = image_path

    def calculate_brightness(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        return brightness

    def more_brightness(self):
        image = cv2.imread(self.image_path)
        current_brightness = self.calculate_brightness(image)
        beta = int((255 - current_brightness) / 2)
        brightened_image = cv2.convertScaleAbs(image, alpha=1, beta=beta)
        cv2.imwrite(self.image_path, brightened_image)

    def reduce_brightness(self):
        image = cv2.imread(self.image_path)
        current_brightness = self.calculate_brightness(image)
        beta = int(-current_brightness / 2)
        brightened_image = cv2.convertScaleAbs(image, alpha=1, beta=beta)
        cv2.imwrite(self.image_path, brightened_image)

    def calculate_contrast(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        return contrast

    def more_contrast(self):
        image = cv2.imread(self.image_path)
        current_contrast = self.calculate_contrast(image)
        target_contrast = 100 
        alpha = target_contrast / current_contrast if current_contrast != 0 else 1.5
        contrasted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        cv2.imwrite(self.image_path, contrasted_image)

    def reduce_contrast(self):
        image = cv2.imread(self.image_path)
        current_contrast = self.calculate_contrast(image)
        target_contrast = 50 
        alpha = target_contrast / current_contrast if current_contrast != 0 else 0.5
        contrasted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        cv2.imwrite(self.image_path, contrasted_image)