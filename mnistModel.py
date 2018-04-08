#!/usr/bin/env python3

import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Convolution2D
from keras.utils import to_categorical
from keras.models import Sequential
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

# to ensure results are reproducible
seed = 2
np.random.seed(seed)

#number of pixels 28 * 28
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#doing to_categorical because the data is being categorized into different numbers
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

#https://www.datacamp.com/courses/deep-learning-in-python
#https://www.tensorflow.org/versions/master/tutorials/layers
model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.33))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

EarlyStopping_monitor = EarlyStopping(patience=2)
model.fit(X_train, y_train, validation_data=(X_test, y_test), validation_split=0.3, epochs=30, callbacks=[EarlyStopping_monitor], batch_size=200) #batch 128

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Baseline Error: %.2f%%" % (100-score[1]*100))

model.save('mnist_model.h5')
