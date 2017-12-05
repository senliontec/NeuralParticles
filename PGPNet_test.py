import sys
sys.path.append("manta/scenes/tools/")

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Conv1D, Conv2DTranspose, BatchNormalization, Input, ZeroPadding2D, Dense, MaxPooling2D, Lambda
from keras.layers import Reshape, RepeatVector, Permute, concatenate, add, Activation, Flatten
from keras import regularizers
import numpy as np
import tensorflow as tf
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
        print(transform.get_shape())
        output = keras.backend.batch_dot(X, transform)
        return output

def test_locnet(cnt,b=None):
    m = Sequential()
    if b is None:
        b = np.eye(3, dtype='float32').flatten()
    W = np.zeros((3, 9), dtype='float32')
    m.add(Dense(9, weights=[W,b], input_shape=(cnt,3)))
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
in_cnt = 1
out_cnt = 1

inputs = Input((in_cnt,3))

x = SpatialTransformer(test_locnet(in_cnt, np.array([0,0,1,0,1,0,1,0,0])))(inputs)

'''g = []
for x in inputs:
    #x = tf.matmul(x, transform)
    
    x = Dense(64, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    g.append(x)

transform = concatenate(g)
transform = Reshape((in_cnt,64))(transform)
transform = feature_transform_net(transform)

h = []
for x in g:
    x = Dense(64, activation='tanh')(x)
    x = Dense(128, activation='tanh')(x)
    h.append(Dense(k, activation='tanh')(x))

# sum 
x = add(h)

x = Dense(512, activation='tanh')(x)
x = Dense(256, activation='tanh')(x)
x = Dense(3*out_cnt, activation='tanh')(x)'''

#x = Reshape((out_cnt,3))(x)

model = Model(inputs=inputs, outputs=x)
model.compile(loss='mse', optimizer=keras.optimizers.adam(lr=0.001))

model.summary()

particles = np.array([[[1,2,3]]])
print(inputs.shape)
print(particles.shape)
print(model.predict(particles, batch_size=1))