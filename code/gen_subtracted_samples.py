'''
Quick script to generate sample BG - subtracted images
Creates a folder called \subtracted that holds the training data after background subtraction

Change input_directory and output_directory to configure filepaths
Change start_int and num_samples to configure how many samples and which samples to process

'''

import matplotlib.pyplot as plt
from subtraction import grayscale
import skimage, numpy, sys, cv2
from skimage import io, filters


plt.rcParams['image.cmap'] = 'gray'

bg_path = '../data/person1/seq1/000053.jpg' #sys.argv[1] #this is the background image with no human
input_directory = '../data/person1/seq3/' #sys.argv[2]
output_directory = '../data/subtracted/falling/' #sys.argv[3]

# Process BG image
background = grayscale(bg_path) 
b_sobel = filters.sobel(background)


start_int = 196
num_samples = 10
for i in range(num_samples):
	fallen = grayscale(input_directory + '000' + str(start_int+i) + '.jpg') 
	f_sobel = filters.sobel(fallen)

	sub = cv2.subtract(b_sobel, f_sobel)

	#TODO: output the image without white margins!!
	plt.imshow(sub)
	plt.axis('off')
	plt.savefig(output_directory + '000' + str(start_int+i) +'.png', bbox_inches='tight', pad_inches=0)