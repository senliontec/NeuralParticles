import keras
from keras.layers.core import Layer
from keras.models import Sequential
from keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

def locnet(cnt,features,kernel):
    m = Sequential()
    m.add(Reshape((cnt,features,1), input_shape=(cnt,features)))
    m.add(Conv2D(64, kernel))
    m.add(Conv2D(128, 1))
    m.add(Conv2D(1024, 1))
    m.add(MaxPooling2D((cnt,1)))
    m.add(Flatten())
    m.add(Dense(512))
    m.add(Dense(256))

    b = np.eye(features, dtype='float32').flatten()
    W = np.zeros((256, features*features), dtype='float32')
    m.add(Dense(features*features, weights=[W,b]))
    m.add(Reshape((features,features)))

    return m

class SpatialTransformer(Layer):
    def __init__(self,
                 cnt,
                 features=3,
                 kernel=(1,3),
                 **kwargs):
        self.cnt = cnt
        self.features=features
        self.kernel = kernel
        self.locnet = locnet(cnt,features,kernel)        
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

    def get_config(self):
        config = {'cnt':self.cnt,
                  'features':self.features,
                  'kernel':self.kernel }        
        return config