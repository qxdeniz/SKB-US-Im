import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model



class ResNetModel:
    def __init__(self, img_path):
        self.img_path = img_path

    def use_brightness_model(self):
        model = load_model('model_classifier.h5')
        model.summary()
        img_path = self.img_path
        img = image.load_img(img_path, target_size=(224, 224))


        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  

        img_array /= 255.0

        prediction = model.predict(img_array)

        classes = ['bright', 'dark', 'normal']
        predicted_class = classes[np.argmax(prediction)]
        print(f'Результат: {predicted_class}')
        return predicted_class


    def use_contrast_model(self):
        model = load_model('model_contrast_classifier.h5')
        model.summary()
        img_path = self.img_path
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  

        img_array /= 255.0
        prediction = model.predict(img_array)
        classes = ['dim', 'contrast', 'normal']
        predicted_class = classes[np.argmax(prediction)]
        print(f'Результат: {predicted_class}')
        return predicted_class
        






