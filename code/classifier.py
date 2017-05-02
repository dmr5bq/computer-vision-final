'''Use model to test sample image and return classification and accuracy
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
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from subtraction import grayscale
import skimage, sys, cv2
from skimage import filters

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import pprint


__author__ = 'Amir Gurung'

def find_confusion_matrix(y_test, prediction):
	test_i = []
	train_i = []

	for elem in y_test:
		test_i.append(elem.argmax())


	for elem in prediction:
		train_i.append(elem.argmax())

	test = np.asarray(test_i)
	train = np.asarray(train_i)

	result = confusion_matrix(test, train)

	# print(result)
	return result, normalize(result)


plt.rcParams['image.cmap'] = 'gray'

epochs = 1
num_classes = 3
classes = ['falling', 'sitting', 'standing']
model = load_model('amir-model') #model-best


bg_path = '../data/person1/seq1/000053.jpg' #sys.argv[1] #this is the background image with no human
input_directory = '../data/person1/seq1/' #sys.argv[2]
output_directory = '../data/test/standing2/' #sys.argv[3] Not really necessary

background = grayscale(bg_path)
b_sobel = filters.sobel(background)


start_int = 122
num_samples = 5
for i in range(num_samples):
	fallen = grayscale(input_directory + '000' + str(start_int+i) + '.jpg')
	f_sobel = filters.sobel(fallen)

	sub = cv2.subtract(b_sobel, f_sobel)

	thresh = filters.threshold_otsu(sub) * 1.2
	sub = sub > thresh

	#TODO: output the image without white margins!!
	plt.imshow(sub)
	plt.axis('off')
	plt.savefig(output_directory + '000' + str(start_int+i) +'.png', bbox_inches='tight', pad_inches=0)


# Input image dimensions
# img_rows, img_cols = 485, 657
img_rows, img_cols = 391, 535


# Extract the data
x_test_path = "../data/test/"
x_test = []
y_test = []

# Append the testing images x_test and y_test
for i in range(len(classes)):
	mypath = x_test_path + classes[i] + '2'
	for f in listdir(mypath):
		filepath = join(mypath, f)
		# print(filepath)
		if isfile(filepath) and filepath[-3:] == 'png':

			img = skimage.io.imread(filepath)
			img = skimage.img_as_float(img)
			img = skimage.color.rgb2gray(img)

			# print(img.shape)

			x_test.append(img)
			y_test.append(i)


#CNN takes an np_array of images into convolution filter
x_test = np.stack(x_test, axis=0)
# x_test = np.asarray(x_test)
# print(x_test.shape)

if K.image_data_format() == 'channels_first':
    print('helloo')
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    print('goodbye')
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

y_test = keras.utils.to_categorical(y_test, num_classes)
# print(y_test.shape)

score = model.evaluate(x_test, y_test, verbose=0)

prediction = model.predict(x_test)

result, n_result = find_confusion_matrix(y_test, prediction)

pprint.pprint(result)
print('---')
pprint.pprint(n_result)
