
import os
import numpy as np
from skimage.io import imread, imsave
import skimage
from random import randint, random
from time import time
from gc import *
from math import ceil

__author__ = "Dominic Ritchey"

class AugmentorOptimized:

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

        self.accumulator = []
        self.input_data = None
        self.start_time = 0

    def load(self):
        src_dir = self.datapath

        self.input_data = []

        list_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

        self.input_data = list_files[:]

    def load_from(self, path):
        src_dir = os.path.join(self.datapath, path)

        self.datapath = src_dir

        list_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

        self.input_data = list_files[:]

    def run(self):
        self.start_time = time()

        self._process()
        print ("{} elapsed -- pass 1 completed".format(time() - self.start_time))

        self._clean()

    def _process(self):
        thres = 0.25
        batch_size = 10

        if self.input_data is not None:
            for batch in range(int(ceil(float(len(self.input_data)) / float(batch_size)))):
                print("\n---\nBatch no: {} Pos: {} / {}"
                      "\n\nTime elapsed: {} s".format(batch, batch_size*batch, len(self.input_data), int(time() - self.start_time)))
                self._load_batch(batch * batch_size, batch_size)

                original_imgs = self.accumulator[:]
                print("\tProcessing...")
                for image in original_imgs:


                    self._add_flips(image)

                    for x in range(10):
                        if random() < thres:
                            self._add_crop_random(image)
                        if random() < thres:
                            self._add_color_random(image)

                self._write_data()
        else:
            raise self.NO_LOAD_EXC

    def _add_flips(self, image):
        self.accumulator.append(np.fliplr(image))
        self.accumulator.append(np.flipud(image))

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
            self.accumulator.append(img_crop_1)

        if area * area_frac < area_2:
            self.accumulator.append(img_crop_2)

        if area * area_frac < area_3:
            self.accumulator.append(img_crop_3)

        if area * area_frac < area_4:
            self.accumulator.append(img_crop_4)

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

        self.accumulator.append(skimage.img_as_ubyte(i_copy))

    def _clean(self):
        self.input_data = []
        self.datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data'))

    def _write_data(self):
        n = len(self.accumulator)
        print("\tWriting...\n\t\tTotal images: {}".format(n))
        if not os.path.exists(self.datapath + '/output'):
            os.makedirs(self.datapath + '/output')
        for i in range(n):
            imsave(os.path.join(self.datapath, 'output/{}.png'.format(time())), self.accumulator[i])

    def _load_batch(self, start_position, batch_size):
        print("\tLoading...")
        del self.accumulator[:]
        collect()
        for i in range(batch_size):
            if start_position + i < len(self.input_data):
                fname = self.input_data[start_position + i]
                self.accumulator.append(imread(os.path.join(self.datapath, fname)))

if __name__ == '__main__':
    a = AugmentorOptimized()
    a.load_from('test')
    a.run()

