import keras
from keras.layers.core import Layer
from keras.models import Sequential, Model
from keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Dense, Input
import numpy as np

import tensorflow as tf

import sys
sys.path.append("manta/scenes/tools/")
from quaternion_mul import quaternion_rot, quaternion_conj

def locnet(cnt,features,kernel,quat=False):
    m = Sequential()
    m.add(Reshape((cnt,features,1), input_shape=(cnt,features)))
    m.add(Conv2D(64, kernel))
    m.add(Conv2D(128, 1))
    m.add(Conv2D(1024, 1))
    m.add(MaxPooling2D((cnt,1)))
    m.add(Flatten())
    m.add(Dense(512, activation='tanh'))
    m.add(Dense(256, activation='tanh'))

    if quat:
        b = np.array([1,0,0,0], dtype='float32')
        W = np.zeros((256, 4), dtype='float32')
        m.add(Dense(4,weights=[W,b]))
    else:
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
                 quat=False,
                 **kwargs):
        self.cnt = cnt
        self.features=features
        self.kernel = kernel
        self.locnet = locnet(cnt,features,kernel,quat)      
        self.quat = quat  
        super(SpatialTransformer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, X, mask=None):
        if type(X) is list:
            x, y = X
        else:
            x, y = X, X
        
        self.transform = self.locnet.call(x)
        if self.quat:
            return quaternion_rot(y,self.transform)
        else:
            return keras.backend.batch_dot(y, self.transform)

    def get_config(self):
        config = {'cnt':self.cnt,
                  'features':self.features,
                  'kernel':self.kernel,
                  'quat':self.quat }        
        return config


if __name__ == "__main__":
    cnt = 100
    par = 10
    pos = np.random.rand(cnt,par,3)

    inputs = Input((par,3))
    x = SpatialTransformer(par,quat=True)(inputs)
    m = Model(inputs=inputs, outputs=x)
    m.summary()
    m.compile( loss='mse', optimizer=keras.optimizers.adam(lr=0.001))

    m.fit(x=pos,y=pos,epochs=20,batch_size=32)
    print(pos[0:1])
    print(m.predict(x=pos[0:1]))

class InverseTransform(Layer):
    def __init__(self,
                 stn,
                 **kwargs):
        self.stn = stn
        super(InverseTransform, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, X, mask=None):
        if self.stn.quat:
            return quaternion_rot(X,quaternion_conj(self.stn.transform))
        else:
            return keras.backend.batch_dot(X, tf.matrix_inverse(self.stn.transform))

    def get_config(self):
        config = {'stn':self.stn.get_config() }        
        return config