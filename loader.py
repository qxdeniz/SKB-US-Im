from tensorflow.keras.preprocessing.image import ImageDataGenerator

directory = '/Users/deniz_mlg/Desktop/Универ/СКБ/ML УЗИ Снимки/quality_classifier/dataset - contrast'


train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  

)

train_generator = train_datagen.flow_from_directory(
    directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  
)


validation_generator = train_datagen.flow_from_directory(
    directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  
)
