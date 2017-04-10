
import os
import numpy as np


class Augmentor:

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
        self.output_data = None
        self.complete = False

    def load(self, path):
        pass

    def run(self):

        if self.input_data is not None:
            while not self.complete:
                for image in self.input_data:
                    pass

        else:
            raise Exception("This object did not load any data-- please execute 'load()'"
                            " before attempting to use 'run()' to load proper image data.")

    def _flip_random(self, image):
        pass

    def _translate_random(self, image):
        pass

    def _crop_random(self, image):
        pass

    def _color_random(self, image):
        pass

    def _write_data(self):
        pass



