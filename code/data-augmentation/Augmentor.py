
import os
import numpy as np
from skimage.io import imread, imsave
import skimage
from random import randint, random
from time import time

class Augmentor:

    NO_LOAD_EXC = Exception("This object did not load any data-- please execute 'load()' or 'load_from()'"
                            " before attempting to use 'run()' to load proper image data.")

    def __init__(self, datapath='', mode='default'):

        if mode == 'abs':
            self.datapath = datapath
        elif mode == 'rel':
            self.datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), datapath))
        elif mode == 'default':
            one_up = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

            self.datapath = os.path.abspath(os.path.join(one_up, os.pardir, 'data'))

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

        for f_name in list_files:
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


    def _process(self):
        if self.input_data is not None:
            for i in range(len(self.input_data)):
                if i % 50 == 0:
                    print("{} / {} processed".format(i, len(self.input_data)))

                image = self.input_data[i]
                self.output_data.append(image)

                decision = random()
                thres = 0.3

                if decision <= thres:

                    self._add_flips(image)

                    for x in range(10):
                        self._add_crop_random(image)
                        self._add_color_random(image)
        else:
            raise self.NO_LOAD_EXC

    def _add_flips(self, image):
        self.output_data.append(np.fliplr(image))
        self.output_data.append(np.flipud(image))

    def _add_translate_random(self, image):
        low = -image.shape[0]/5
        high = image.shape[0]/5
        d = randint(low, high)

    def _add_crop_random(self, image):

        w = image.shape[0]
        h = image.shape[1]

        x, y = randint(w / 4, 3 * w / 4), randint(h / 4, 3 * h / 4)

        w_x = randint(w/5, w/4 - 1)
        w_y = randint(h/5, h/4 - 1)

        img_crop_l = image[0: x + w_x, y + w_y]
        img_crop_r = image[x - w_x: w, y - w_y: h]

        area = w * h

        area_frac = 0.55 # this seems to be a good threshold for now

        area_l = img_crop_l.shape[0] * img_crop_l.shape[1]
        area_r = img_crop_r.shape[0] * img_crop_r.shape[1]

        if area * area_frac < area_l:
            self.output_data.append(img_crop_l)

        if area * area_frac < area_r:
            self.output_data.append(img_crop_r)

    def _add_color_random(self, image):

        mult = 0.25
        off_r = random()
        off_g = random()
        off_b = random()

        i_copy = skimage.img_as_float(image)

        i_copy[:, :, 0] = np.add(i_copy[:, :, 0], off_r * mult)
        i_copy[:, :, 1] = np.add(i_copy[:, :, 1], off_g * mult)
        i_copy[:, :, 2] = np.add(i_copy[:, :, 2], off_b * mult)

        for x in range(i_copy.shape[0]):
            for y in range(i_copy.shape[1]):
                for i in range(3):
                    i_copy[x, y, i] = min(i_copy[x, y, i], 1.)
                    i_copy[x, y, i] = max(0, i_copy[x, y, i])

        self.output_data.append(skimage.img_as_ubyte(i_copy))

    def _write_data(self):
        n = len(self.output_data)
        for i in range(n):
            if i % 50 == 0:
                print("{} / {} saved".format(i, n))
            imsave(os.path.join(self.datapath, 'output/{}.png'.format(i)), self.output_data[i])


if __name__ == '__main__':
    a = Augmentor()
    a.load_from('person4/seq3')
    a.run()
