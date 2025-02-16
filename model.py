import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from loader import train_generator, validation_generator
from tensorflow.keras.callbacks import EarlyStopping


base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)  
x = Dense(1024, activation='relu')(x)  
predictions = Dense(3, activation='softmax')(x)  


model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
early_stopping = EarlyStopping(monitor='loss', patience=3)


history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,  
    callbacks=[early_stopping]
)

loss, accuracy = model.evaluate(validation_generator, steps=50)
print(f'Validation Loss: {loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')

model.save('model_classifier.h5')