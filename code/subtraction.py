import skimage, numpy, sys, cv2
from skimage import io, filters
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import sqrt, abs, mean, newaxis
import os.path
from image_processor import edge_detector

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
	# plt.imshow(sub)
	# plt.title("edge detector")
	# plt.show()


	# -- Good, but different: Grayscale, sobel, subtract -- #
	# -- Basically leaves a gray background with a black outline of the person -- #
	background = grayscale(background_path) 
	b_sobel = filters.sobel(background)

	fallen = grayscale(fallen_path) 
	f_sobel = filters.sobel(fallen)

	sub = cv2.subtract(b_sobel, f_sobel)

	plt.imshow(sub)
	plt.title("Grayscale, sobel, subtract")
	plt.show()

	