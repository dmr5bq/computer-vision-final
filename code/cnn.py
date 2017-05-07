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
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import optimizers
import h5py

__author__ = "Divya Bhaskara"

# Training parameters
#batch_size = 128
epochs = 10
num_classes = 3
classes = ['falling', 'sitting', 'standing']

# Input image dimensions
img_rows, img_cols = 266, 400

# Extract the data
train_path = "Training/"
validation_path = "Validation/"
validation_samples = 107
batch_size = 16
total_samples = 528

def grayscale(img):

	rows = img.shape[0]
	cols = img.shape[1]

	print(img.shape)
	averaged = np.zeros((rows, cols))
	for r in range(rows):
		for c in range(cols):
			luminance = .21 * img[r][c][0] + .72 * img[r][c][1] + .07 * img[r][c][2]
			averaged[r][c] = luminance

	return averaged

if __name__ == "__main__":

	model = Sequential()

	if K.image_data_format() == 'channels_first':
		input_shape = (3, img_rows, img_cols)
	else:
		input_shape = (img_rows, img_cols, 3)

	# ----- Based off of simple deep net for CIFAR small images dataset ------
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss='sparse_categorical_crossentropy',
		optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
		metrics=['accuracy'])

	# Set up data 
	train_datagen = ImageDataGenerator(rescale=1. / 255)
	validation_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
		train_path,
		target_size=(img_rows, img_cols),
		batch_size=batch_size,
		class_mode='binary')

	validation_generator = validation_datagen.flow_from_directory(
		validation_path,
		target_size=(img_rows, img_cols),
		batch_size=batch_size,
		class_mode='binary')

	start = time.time()

	model.fit_generator(
		train_generator,
		epochs=epochs,
		steps_per_epoch = total_samples//batch_size,
		validation_data=validation_generator,
		validation_steps=validation_samples)

	# Find training time
	end = time.time()
	print('Training time:', end-start)
	start = time.time()

	# Save the model
	model.save('GlennaFalling.hd5')
	end = time.time()
	print('Saving time: ', end-start)
