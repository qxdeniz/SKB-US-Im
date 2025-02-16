from model_use import ResNetModel
from brightness_corrector import Corrector
import cv2

MAX_DEPTH = 10

def correct_brightness(img_path, depth=0):
    if depth > MAX_DEPTH:
        return
    model = ResNetModel(img_path=img_path)
    prediction = model.use_brightness_model()
    print(prediction)
    corrector = Corrector(image_path=img_path)
    if prediction == 'bright':
        corrector.reduce_brightness()
        correct_brightness(img_path=img_path, depth=depth+1)
    elif prediction == 'dark':
        corrector.more_brightness()
        correct_brightness(img_path=img_path, depth=depth+1)
    elif prediction == 'normal':
         return True
    

def correct_contrast(img_path, depth=0):
    if depth > MAX_DEPTH:
        return
    model = ResNetModel(img_path=img_path)
    prediction = model.use_contrast_model()
    print(prediction)
    corrector = Corrector(image_path=img_path)
    if prediction == 'dim':
        corrector.more_contrast()
        correct_contrast(img_path=img_path, depth=depth+1)
    elif prediction == 'contrast':
        corrector.reduce_contrast()
        correct_contrast(img_path=img_path, depth=depth+1)
    elif prediction == 'normal':
         return True


def run_pipe(img_path):
    correct_brightness(img_path, 0)
    correct_brightness(img_path, 0)
    cv2.imshow('Новый снимок', cv2.imread(img_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



run_pipe('test2.jpg')