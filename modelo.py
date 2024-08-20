import tensorflow as tf
import numpy as np
from keras.src.models import sequential as sq
from keras.src.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras_preprocessing.image import ImageDataGenerator as imdg

datagen = imdg(rescale = 0.1, validation_split = 0.2)

train_generator = datagen.flow_from_directory(
    'path_to_dataset', 
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'path_to_dataset', 
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

model = sq([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Guardar el modelo
model.save('pattern_model.h5')