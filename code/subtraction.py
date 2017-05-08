from __future__ import print_function
import argparse, imutils, skimage, numpy, sys, cv2
import matplotlib.pyplot as plt
import os.path
from image_processor import edge_detector
from imutils.object_detection import non_max_suppression
from imutils import paths
from numpy import sqrt, abs, mean, newaxis
from PIL import Image
from skimage import io, filters
from sklearn.metrics import confusion_matrix

KERNEL_SHAPE = 5
LUMINANCE_RED = .21
LUMINANCE_GREEN = .72
LUMINANCE_YELLOW = 0.07

def grayscale(imgpath):
	"""Given an image path, return the grayscaled version of that image"""
	img = skimage.img_as_float(io.imread(imgpath))

	rows = img.shape[0]
	cols = img.shape[1]

	averaged = numpy.zeros((rows, cols))
	for r in range(rows):
		for c in range(cols):
			luminance = LUMINANCE_RED * img[r][c][0] + LUMINANCE_GREEN * img[r][c][1] + LUMINANCE_YELLOW * img[r][c][2]
			averaged[r][c] = luminance

	return averaged

def find_background_image(class_name, path):
	"""Given a directory and a class name, search for a viable background image"""
	# This method based on: http://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
	 
	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# loop over the image paths for each class
	class_path = os.path.join(path, class_name) 
	for imagePath in paths.list_images(class_path):
		# load the image and resize it to [faster, more accurate]
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=min(400, image.shape[1]))
		orig = image.copy() 

		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
			padding=(8, 8), scale=1.05)

		# draw bounding boxes
		for (x, y, w, h) in rects:
			cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# If there are no people (no boxes), return the path of this image
		if len(rects) == 0:
			return imagePath

def convolved(averaged_img):
	kernel = numpy.ones((KERNEL_SHAPE, KERNEL_SHAPE))
	convolved = ndimage.filters.convolve(averaged_img, kernel)
	return convolved

def process(background, image_path):
	"""Grayscale, Resize, Sobel, Subtract, Otsu"""

	# Grayscale Image
	image = grayscale(image_path)

	# Resize image to maximum of 400 pixels
	image = imutils.resize(image, width=min(400, len(image[0])))

	# Apply sobel filter to test image
	image = filters.sobel(image)

	# Subtract background from test image
	result = cv2.subtract(background, image)

	# Find otsu threshold		
	thresh = filters.threshold_otsu(result) * 1.2

	# Apply threshold
	result = result > thresh

	# Rescale to uint8
	rescaled = (((result - result.min()) / (result.max() - result.min())) * 255.9).astype(numpy.uint8)

	return rescaled

if __name__ == "__main__": 
	base_path = "../data/Training"
	classes = ['falling', 'sitting', 'standing']

	for class_name in classes:
		# Assuming each class has a different background image
		background_path = find_background_image(class_name, base_path)

		# Resize, grayscale, and sobel filter background
		background = skimage.img_as_float(background_path)
		background = imutils.resize(background, width=min(400, background.shape[1]))
		background = filters.sobel(grayscale_image(background))

		class_path = os.path.join(base_path, class_name)
		for name in os.listdir(class_path):
			# Find png or jpg images
			if name[-3:].lower() == 'jpg' or name[-3:].lower() == 'png':
				result = process(background, os.path.join(class_path), name)
				save_path = os.path.join(class_path, "processed", name)

				im = Image.fromarray(result)
				io.imsave(save_path, im)

				print(name)

		



	