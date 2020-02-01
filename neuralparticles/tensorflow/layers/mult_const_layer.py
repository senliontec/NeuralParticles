import keras
from keras.layers.core import Layer
import numpy as np
from keras import backend as K

class MultConst(Layer):
    def __init__(self, c, **kwargs): 
        super(MultConst, self).__init__(**kwargs)
        self.c = c

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, X, mask=None):
        return X * self.c

    def get_config(self):
        config = super(MultConst, self).get_config()
        try:
            config['c'] = list(self.c)
        except TypeError:
            config['c'] = self.c
        
        return config

