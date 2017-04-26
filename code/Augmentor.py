
import os
import numpy as np
from skimage.io import imread, imsave
import skimage
from random import randint, random
from time import time
from gc import *

__author__ = "Dominic Ritchey"

class Augmentor:

    NO_LOAD_EXC = RuntimeError("This object did not load any data-- please execute 'load()' or 'load_from()'"
                               " before attempting to use 'run()' to load proper image data.")

    def __init__(self, datapath='', mode='default'):
        if mode == 'abs':
            self.datapath = datapath

        elif mode == 'rel':
            self.datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), datapath))

        elif mode == 'default':

            self.datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data'))

            print("You are now pointed at the project's data root. If this isn't what you intended, use "
                  "\"rel\" mode or \"abs\" mode")

        self.input_data = None
        self.output_data = []

    def load(self):
        src_dir = self.datapath

        self.input_data = []

        list_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

        for f_name in list_files:
            img = imread(os.path.join(src_dir, f_name))
            self.input_data.append(img)

    def load_from(self, path):
        src_dir = os.path.join(self.datapath, path)

        self.datapath = src_dir

        self.input_data = []

        list_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

        #for f_name in list_files:
        for i in range(10):
            f_name = list_files[i]
            img = imread(os.path.join(src_dir, f_name))
            self.input_data.append(img)

    def run(self):
        start_t = time()

        self._process()
        print ("{} elapsed -- pass 1 completed".format(time() - start_t))

        self.input_data = self.output_data[:]
        self._process()
        print ("{} elapsed -- pass 2 completed".format(time() - start_t))

        self._write_data()
        print ("{} elapsed -- write completed".format(time() - start_t))

        self._clean()

    def _process(self):
        collect()
        thres = 0.5

        if self.input_data is not None:
            for i in range(len(self.input_data)):
                if i % 50 == 0:
                    print("{} / {} processed".format(i, len(self.input_data)))

                image = self.input_data[i]
                self.output_data.append(image)


                self._add_flips(image)

                for x in range(10):
                    if random() < thres:
                        self._add_crop_random(image)
                    if random() < thres:
                        self._add_color_random(image)
        else:
            raise self.NO_LOAD_EXC

    def _add_flips(self, image):
        self.output_data.append(np.fliplr(image))
        self.output_data.append(np.flipud(image))

    def _add_crop_random(self, image):

        w = image.shape[0]
        h = image.shape[1]

        area = w * h

        area_frac = 0.8  # this seems to be a good threshold for now

        x_rg = w / 4, 3 * w / 4
        y_rg = h / 4, 3 * h / 4

        x, y = randint(x_rg[0], x_rg[1]), randint(y_rg[0], y_rg[1])

        w_x = randint(w/5, w/4 - 1)
        w_y = randint(h/5, h/4 - 1)

        img_crop_1 = image[0: x + w_x, 0: y + w_y]
        img_crop_2 = image[x - w_x: w, y - w_y: h]
        img_crop_3 = image[0: x + w_x, y - w_y: h]
        img_crop_4 = image[x - w_x: w, 0: y + w_y]

        area_1 = img_crop_1.shape[0] * img_crop_1.shape[1]
        area_2 = img_crop_2.shape[0] * img_crop_2.shape[1]
        area_3 = img_crop_1.shape[0] * img_crop_1.shape[1]
        area_4 = img_crop_2.shape[0] * img_crop_2.shape[1]

        if area * area_frac < area_1:
            self.output_data.append(img_crop_1)

        if area * area_frac < area_2:
            self.output_data.append(img_crop_2)

        if area * area_frac < area_3:
            self.output_data.append(img_crop_3)

        if area * area_frac < area_4:
            self.output_data.append(img_crop_4)

    def _add_color_random(self, image):

        mult = 0.3
        off_r = random()
        off_g = random()
        off_b = random()

        i_copy = skimage.img_as_float(image)

        i_copy[:, :, 0] = np.add(i_copy[:, :, 0], (off_r * mult) - mult/2)
        i_copy[:, :, 1] = np.add(i_copy[:, :, 1], (off_g * mult) - mult/2)
        i_copy[:, :, 2] = np.add(i_copy[:, :, 2], (off_b * mult) - mult/2)

        for x in range(i_copy.shape[0]):
            for y in range(i_copy.shape[1]):
                for i in range(3):
                    i_copy[x, y, i] = min(i_copy[x, y, i], 1.)
                    i_copy[x, y, i] = max(0, i_copy[x, y, i])

        self.output_data.append(skimage.img_as_ubyte(i_copy))

    def _clean(self):
        self.input_data = []
        self.output_data = []
        self.datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data'))

    def _write_data(self):
        n = len(self.output_data)
        if not os.path.exists(self.datapath + '/output'):
            os.makedirs(self.datapath + '/output')
        for i in range(n):
            if i % 50 == 0:
                print("{} / {} saved".format(i, n))
            imsave(os.path.join(self.datapath, 'output/{}.png'.format(i)), self.output_data[i])


if __name__ == '__main__':
    a = Augmentor()
    a.load_from('person4/seq3')
    a.run()
