import keras
from keras.layers.core import Layer

class SpatialTransformer(Layer):
    def __init__(self,
                 localization_net,
                 **kwargs):
        self.locnet = localization_net        
        super(SpatialTransformer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, X, mask=None):
        transform = self.locnet.call(X)
        output = keras.backend.batch_dot(X, transform)
        return output