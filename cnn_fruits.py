#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint


# Load training images (as filenames for now) and labels,
# same for validation/test images
train_dir = 'dataset/Training'
test_dir = 'dataset/Test'


def load_dataset(directory):
    categories = os.listdir(directory)
    x_data = []
    y_data = []

    for category in categories:
        path = os.path.join(directory, category)
        category_num = categories.index(category)
        for img in os.listdir(path):
            x_data.append(os.path.join(path, img))
            y_data.append(category_num)
    return (np.array(x_data), np.array(y_data), np.array(categories))


x_train, y_train, y_labels = load_dataset(train_dir)
x_test, y_test, _ = load_dataset(test_dir)

print('Training dataset size:', len(x_train))
print('Testing dataset size:', len(x_test))

num_of_classes = len(np.unique(y_train))
print('number of classes', num_of_classes)

# Encode every label as one-hot (only element corresponding to
# it's label has value 1 and others are 0)
y_train = tf.keras.utils.to_categorical(y_train, num_of_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_of_classes)
print(y_train[0])

# Divide testing dataset into test and validation dataset
x_test, x_valid = x_test[12000:], x_test[:12000]
y_test, y_valid = y_test[12000:], y_test[:12000]


# Load images ussing filenames and convert to numpy array
def convert_image_to_array(files):
    images_as_array = []
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array


x_train = np.array(convert_image_to_array(x_train))
print('Training set shape:', x_train.shape)

x_valid = np.array(convert_image_to_array(x_valid))
print('Validation set shape:', x_valid.shape)

x_test = np.array(convert_image_to_array(x_test))
print('Test set shape:', x_test.shape)

# scale images between 0 and 1
x_train = x_train.astype('float32') / 255
x_valid = x_valid.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Creating Keras model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=5, input_shape=(100, 100, 3),
                 padding='same', use_bias=False))
model.add(BatchNormalization(center=True, scale=False))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=5, use_bias=False, padding='same'))
model.add(BatchNormalization(center=True, scale=False))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=5, use_bias=False, padding='same'))
model.add(BatchNormalization(center=True, scale=False))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=5, use_bias=False, padding='same'))
model.add(BatchNormalization(center=True, scale=False))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(131, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Model is compiled!')

# Save best model (with minimum validation loss)
checkpointer = ModelCheckpoint(
    filepath='cnn_model.hdf5', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=30,
                    validation_data=(x_valid, y_valid),
                    callbacks=[checkpointer],
                    verbose=1, shuffle=True)

# Load best model
model.load_weights('cnn_model.hdf5')

# Evaluate on the testing dataset
score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest accuracy:', score[1])

# Predict classes of the testing dataset
y_pred = model.predict(x_test)

# print predicted class and actual class of the image
for idx in range(len(y_pred)):
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    print(y_labels[pred_idx], y_labels[true_idx])
