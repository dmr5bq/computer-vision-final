'''Trains a simple convnet on the sample video feed data
'''

from __future__ import print_function
import keras
import time
import numpy as np
import skimage.io
import skimage.color
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

__author__ = "Divya Bhaskara"


# Training parameters
#batch_size = 128
epochs = 12
num_classes = 3
classes = ['falling', 'sitting', 'standing']

# Input image dimensions
img_rows, img_cols = 388, 284

# Extract the data
x_train_path = "../data/augment-subtract/"
x_train = []
y_train = []

train_data_dir = '../data/augment-subtract'      # Path to training images
validation_data_dir = 'cats_and_dogs_medium/test'  # Validation and test set are the same here

for i in range(len(classes)):
	mypath = x_train_path + classes[i]
	for f in listdir(mypath):
		filepath = join(mypath, f)
		if isfile(filepath) and (filepath[-3:] == 'png' or filepath[-3:] == 'jpg'):
			img = skimage.io.imread(filepath)
			img = skimage.img_as_float(img)
			img = skimage.color.rgb2gray(img)

			x_train.append(img)
			y_train.append(i)

x_train = np.stack(x_train, axis=0)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Plotting images: sanity check for reshaping
# for m in range(x_train.shape[0]):
# 	img = x_train[m, :, :, 0]
# 	print(img.shape)
# 	plt.imshow(img)
# 	plt.show()

x_train = x_train.astype('float32')
x_train /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
y_train = keras.utils.to_categorical(y_train, num_classes)


# Building the neural network 
model = Sequential()

# ----- Based off of convnet architecture that worked for MNIST handwritten samples -----
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, kernel_size=(3, 3), activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, kernel_size=(3, 3), activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_classes, activation='sigmoid'))


# ----- Based off of simple deep net for CIFAR small images dataset ------
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Train the weights of the network
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])


start = time.time()
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples)

# this just outputs the training accuracy for now
end = time.time()
print('Training time:', end-start)


model.save('model')

