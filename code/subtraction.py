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

KERNEL_SHAPE = 5

def grayscale(imgpath):
	img = skimage.img_as_float(io.imread(imgpath))

	rows = img.shape[0]
	cols = img.shape[1]

	averaged = numpy.zeros((rows, cols))
	for r in range(rows):
		for c in range(cols):
			luminance = .21 * img[r][c][0] + .72 * img[r][c][1] + .07 * img[r][c][2]
			averaged[r][c] = luminance

	return averaged

def find_background_image():
	# construct the argument parse and parse the arguments from command line
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--images", required=True, help="path to images directory")
	args = vars(ap.parse_args())
	 
	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# loop over the image paths
	for imagePath in paths.list_images(args["images"]):
		# load the image and resize it to (1) reduce detection time
		# and (2) improve detection accuracy
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=min(400, image.shape[1]))
		orig = image.copy() 

		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
			padding=(8, 8), scale=1.05)

		# If there are no people (no boxes), return the path of this image
		number = len(rects)
		if number == 0:
			return imagePath
 
def find_person(myimg):
	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	image = imutils.resize(myimg, width=min(400, myimg.shape[1]))
	orig = image.copy() 

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = numpy.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
	# show some information on the number of bounding boxes
	print("[INFO] {}: {} original boxes, {} after suppression".format(
		'subtracted', len(rects), len(pick)))
 
	# show the output images
	cv2.imshow("Before NMS", orig)
	cv2.imshow("After NMS", image)
	cv2.waitKey(0)

def subtraction(background, fallen):
	if background.shape == fallen.shape:
		rows = background.shape[0]
		cols = background.shape[1]
	else:
		raise ValueError("Background and Fallen image shapes must match")

	result = numpy.zeros((rows, cols))

	for r in range(rows):
		for c in range(cols):
			diff = abs(fallen[r][c] - background[r][c])
			if diff > .3:
				result[r][c] = diff

	return result

def convolved(averaged_img):
	kernel = numpy.ones((KERNEL_SHAPE, KERNEL_SHAPE))
	convolved = ndimage.filters.convolve(averaged_img, kernel)
	return convolved

if __name__ == "__main__": 
	fallen_directory = '../data/person4/seq3'

	background_path = '../data/background.jpg'
	fallen_path = '../data/fallen.jpg'

	plt.rcParams['image.cmap'] = 'gray'


	background_path = find_background_image()
	print(background_path)
	background = grayscale(background_path)
	other = '../data/pedestrian/000223.jpg'
	fallen = grayscale(other)
	sub = cv2.subtract(fallen, background)

	plt.imshow(sub)
	plt.title("subtracted")
	plt.show()

	print("Fallen, Background, Sub")

	rescaled = (((sub - sub.min()) / (sub.max() - sub.min())) * 255.9).astype(numpy.uint8)

	rescaled_fallen = (((fallen - fallen.min()) / (fallen.max() - fallen.min())) * 255.9).astype(numpy.uint8)

	rotated = numpy.rot90(rescaled, 1)

	plt.imshow(rotated)
	plt.title("rotate")
	plt.show()

	# io.imsave('subtracted.jpg', rescaled)

	find_person(rescaled_fallen)


	# background_path = '../data/kar-bgd.png'


	# -- Good: Grayscale, then subtract -- #
	# averaged_bgd = grayscale(background_path)
	# averaged_f = grayscale(fallen_path)
	# sub = cv2.subtract(averaged_f, averaged_bgd)

	# -- Good: Sub, then canny edge detector -- #
	# averaged_bgd = grayscale(background_path)
	# averaged_f = grayscale(fallen_path)
	# sub = cv2.subtract(averaged_f, averaged_bgd)
	# sub_edges = edge_detector(sub)

	# -- canny edge detector then subtract -- #
	# -- Not really usable IMO but who knows what neural networks learn best -- #
	# background_edges = edge_detector(background_path)
	# fallen_edges = edge_detector(fallen_path)
	# sub = cv2.subtract(background_edges, fallen_edges)



	# -- Good, but different: Grayscale, sobel, subtract -- #
	# -- Basically leaves a gray background with a black outline of the person -- #
	# background = grayscale(background_path) 
	# b_sobel = filters.sobel(background)

	# fallen = grayscale(fallen_path) 
	# f_sobel = filters.sobel(fallen)

	# sub = cv2.subtract(b_sobel, f_sobel)

	# plt.imshow(sub)
	# plt.title("Grayscale, sobel, subtract")
	# plt.show()

	