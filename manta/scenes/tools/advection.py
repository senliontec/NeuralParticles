import keras
from keras.layers.core import Layer
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Reshape, Flatten
import numpy as np
from keras import backend as K

import tensorflow as tf
from quaternion_mul import quaternion_rot, quaternion_conj, quaternion_norm
from helpers import *

def interpol(sdf, pos, batch_size, bnd):
    pos_wx = int(sdf.get_shape()[1])
    pos_wy = int(sdf.get_shape()[2])
    pos = K.reshape(pos, (batch_size, pos_wx*pos_wy, 2))+bnd-0.5
    sdf = tf.pad(sdf, tf.constant([[0,0],[bnd,bnd],[bnd,bnd]]), 'constant', constant_values=1)

    w = int(sdf.get_shape()[1])

    x = K.cast(pos[:,:,0], 'int32')
    y = K.cast(pos[:,:,1], 'int32')
    idx = tf.unstack(x + y * w)

    facX = pos[:,:,0]-K.cast(x, 'float32')
    facY = pos[:,:,1]-K.cast(y, 'float32')

    sdf = tf.unstack(K.reshape(sdf, (batch_size, w*w)))

    v  = tf.stack([K.gather(sdf[i], idx[i]) for i in range(batch_size)]) * (1-facX) * (1-facY)
    v += tf.stack([K.gather(sdf[i], idx[i]+1) for i in range(batch_size)]) * facX * (1-facY)
    v += tf.stack([K.gather(sdf[i], idx[i]+w) for i in range(batch_size)]) * (1-facX) * facY
    v += tf.stack([K.gather(sdf[i], idx[i]+w+1) for i in range(batch_size)]) * facX * facY

    return K.reshape(v, (batch_size,pos_wx, pos_wy))

def advection(grid, vec, batch_size, bnd):
    idx = K.constant(np.array([[[[x,y] for x in range(grid.get_shape()[1])] for y in range(grid.get_shape()[2])]], dtype='float32') + 0.5) - vec
    return interpol(grid, idx, batch_size, bnd)

def rotation_vec(quat, size):
    bs = int(quat.get_shape()[0])
    res = np.empty((bs,size,size,3))
    for y in range(size):
        for x in range(size):
            v = np.array([[x-size/2+0.5,y-size/2+0.5,0]]*bs)
            res[:,y,x] = v
    res = K.reshape(K.constant(res), (bs, size*size, 3))
    print(K.eval(res))
    print(K.eval(quaternion_rot(res, quat)))
    return K.reshape(res - quaternion_rot(res, quat), (bs, size, size, 3))[:,:,:,:2]

class Advection(Layer):
    def __init__(self, batch_size, bnd, **kwargs): 
        self.batch_size = batch_size
        self.bnd = bnd
        super(Advection, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, X, mask=None):
        return advection(X[0],X[1], self.batch_size, self.bnd)

    def get_config(self):
        config = {'batch_size':self.batch_size,
                  'bnd':self.bnd }        
        return config


if __name__ == "__main__":
    batch_size = 32
    bnd = 5
    def insert_square(src,x,y):
        x = int(x*7)
        y = int(y*7)
        src[x,y] = -1
        src[x,y+1] = -1
        src[x+1,y] = -1
        src[x+1,y+1] = -1

    src = np.ones((batch_size*10,10,10))
    dst = np.ones((batch_size*10,10,10))
    for i in range(batch_size*10):
        x = np.random.rand()
        y = np.random.rand()
        insert_square(src[i],x,y)
        insert_square(dst[i],x+(2/7),y+(2/7))

    vec = rotation_vec(K.constant(np.array([[-0.7071068,0,0,0.7071068]]*batch_size), dtype="float32"), 10)# K.reshape(np.array([2,0]*100*batch_size, dtype='float32'), (batch_size,10,10,2))

    #plot_vec(K.eval(vec[0]*0.1), [0,10], [0,10])
    print(K.eval(K.constant(src[0:1])))
    print(K.eval(advection(K.constant(src[:batch_size]),vec, batch_size, bnd))[0:1])

    #print(K.eval(rotation_vec(K.constant(np.array([[0.7071068,0,0,0.7071068]]*2), dtype="float32"), 5)))

    inputs = Input((10,10))
    x = Flatten()(inputs)
    x = Dense(100, activation='tanh')(x)
    b = np.zeros((200,), dtype='float32')
    W = np.zeros((100, 200), dtype='float32')
    x = Dense(200, activation='tanh', weights=[W,b])(x)

    vec = Reshape((10,10,2))(x)
    x = Advection(batch_size, bnd)([inputs,vec])

    m = Model(inputs=inputs, outputs=x)
    
    m.compile( loss='mse', optimizer=keras.optimizers.adam(lr=0.001))

    m.summary()

    m.fit(x=src,y=dst,epochs=20,batch_size=batch_size)

    print(np.floor(m.predict(x=src,batch_size=batch_size)[0]*10))
    #print(Model(inputs=inputs, output=vec).predict(x=src,batch_size=batch_size)[0])
    
