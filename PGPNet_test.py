import sys
sys.path.append("manta/scenes/tools/")

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Conv1D, Conv2DTranspose, BatchNormalization, Input, ZeroPadding2D, Dense, MaxPooling2D, Lambda
from keras.layers import Reshape, RepeatVector, Permute, concatenate, add, Activation, Flatten
from keras import regularizers
import numpy as np
#from spatial_transformer import SpatialTransformer 

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

def test_locnet(cnt,b=None):
    m = Sequential()
    m.add(Flatten(input_shape=(cnt,3)))
    if b is None:
        b = np.eye(3, dtype='float32').flatten()
    W = np.zeros((cnt*3, 9), dtype='float32')
    m.add(Dense(9, weights=[W,b]))
    m.add(Reshape((3,3)))
    return m

def input_locnet(cnt):
    m = Sequential()
    m.add(Reshape((cnt,3,1), input_shape=(cnt,3)))
    m.add(Conv2D(64, (1,3)))
    m.add(Conv2D(128, 1))
    m.add(Conv2D(1024, 1))
    m.add(MaxPooling2D((cnt,1)))
    m.add(Flatten())
    m.add(Dense(512))
    m.add(Dense(256))

    b = np.eye(3, dtype='float32').flatten()
    W = np.zeros((256, 9), dtype='float32')
    m.add(Dense(9, weights=[W,b]))
    m.add(Reshape((3,3)))

    return m

def feature_locnet(cnt, K=64):
    m = Sequential()
    m.add(Reshape((cnt,K,1), input_shape=(cnt,K)))
    m.add(Conv2D(64, 1))
    m.add(Conv2D(128, 1))
    m.add(Conv2D(1024, 1))
    m.add(MaxPooling2D((cnt,1)))
    m.add(Flatten())
    m.add(Dense(512))
    m.add(Dense(256))

    b = np.eye(K, dtype='float32').flatten()
    W = np.zeros((256, K*K), dtype='float32')
    m.add(Dense(K*K, weights=[W,b]))
    m.add(Reshape((K,K)))

    return m

k = 128
in_cnt = 2
out_cnt = 2

inputs = [Input((1,3)) for i in range(in_cnt)]

branch = Sequential()
branch.add(SpatialTransformer(test_locnet(1), input_shape=(1,3)))
branch.add(Dense(64, activation='tanh'))
branch.add(Dense(64, activation='tanh'))
branch.add(SpatialTransformer(feature_locnet(1)))
branch.add(Dense(64, activation='tanh'))
branch.add(Dense(128, activation='tanh'))
branch.add(Dense(k, activation='tanh'))

x = SplitLayer(branch)(inputs)

x = add(x)

x = Dense(512, activation='tanh')(x)
x = Dense(256, activation='tanh')(x)
x = Dense(3*out_cnt, activation='tanh')(x)

x = Reshape((out_cnt,3))(x)

model = Model(inputs=inputs, outputs=x)
model.compile(loss='mse', optimizer=keras.optimizers.adam(lr=0.001))

model.summary()

particles = [np.array([[[3,4,6]]]),np.array([[[1,2,3]]])]

print(model.predict(particles, batch_size=1))