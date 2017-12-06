import keras
from keras.layers.core import Layer

class SplitLayer(Layer):
    def __init__(self, layer, **kwargs):
        self.layer = layer
        super(SplitLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.layer.build(input_shape)
        self.trainable_weights = self.layer.trainable_weights

    def compute_output_shape(self, input_shape):
        return [self.layer.output_shape] * len(input_shape)

    def call(self, X, mask=None):
        Y = []
        for x in X:
            Y.append(self.layer(x))
        return Y