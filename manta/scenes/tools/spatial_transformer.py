import keras
import keras.backend as K
from keras.layers.core import Layer
from keras.models import Sequential, Model
from keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, Lambda
import numpy as np
import math

import tensorflow as tf

import sys
sys.path.append("manta/scenes/tools/")
from quaternion_mul import quaternion_rot, quaternion_conj, quaternion_norm

from advection import advection, rotate_grid, transform_grid

from helpers import *

def SpatialTransformer(inputs, cnt,features=3,kernel=(1,3),dropout=0.2,fac=64,quat=False,norm=False):
    x = Reshape((cnt,features,1))(inputs)
    x = Conv2D(64, kernel)(x)
    x = Conv2D(128, 1)(x)
    x = Conv2D(fac*4, 1)(x)
    x = MaxPooling2D((cnt,1))(x)
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*2, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(fac, activation='tanh')(x)
    x = Dropout(dropout)(x)

    if quat:
        b = np.array([1,0,0,0], dtype='float32')
        W = np.zeros((fac, 4), dtype='float32')
        x = Dense(4,weights=[W,b])(x)
        if norm:
            x = Lambda(quaternion_norm)(x)
    else:
        b = np.eye(features, dtype='float32').flatten()
        W = np.zeros((fac, features*features), dtype='float32')
        x = Dense(features*features, weights=[W,b])(x)
        x = Reshape((features,features))(x)
    return x

def stn_transform(transform, x, quat=False):
    def tmp(v):
        return quaternion_rot(v[0],v[1]) if quat else K.batch_dot(v[0],v[1])
    return Lambda(tmp)([x,transform])

def stn_transform_inv(transform, x, quat=False):
    def tmp(v):
        return quaternion_rot(v[0],quaternion_conj(v[1])) if quat else K.batch_dot(v[0],tf.matrix_inverse(v[1]))
    return Lambda(tmp)([x,transform])

def stn_grid_transform(transform, x, quat=False):
    def tmp(v):
        return rotate_grid(v[0],v[1]) if quat else transform_grid(v[0],v[1])
    return Lambda(tmp)([x,transform])

def stn_grid_transform_inv(transform, x, quat=False):
    def tmp(v):
        return rotate_grid(v[0],quaternion_conj(v[1])) if quat else transform_grid(v[0],tf.matrix_inverse(v[1]))
    return Lambda(tmp)([x,transform])

if __name__ == "__main__":
    cnt = 1000
    par = 100
    grid_size = 15
    pos = np.random.rand(cnt,par,3) * np.array([[[2,1,1]]]) - np.array([[[1,0,0]]])

    src = np.empty((cnt,par,3))
    grid_src = np.ones((cnt,grid_size,grid_size))
    for i in range(cnt):
        theta = math.pi * np.random.rand()
        c, s = np.cos(theta), np.sin(theta)
        mat = np.matrix([[c,-s,0],[s,c,0],[0,0,1]])
        src[i] = pos[i] * mat
        for p in src[i]:
            x = max(min(int((p[0]+1)*grid_size/2),grid_size-1),0)
            y = max(min(int((p[1]+1)*grid_size/2),grid_size-1),0)
            grid_src[i,y,x] = -1

    #plot_sdf(grid_src[0], [0,grid_size],[0,grid_size])
    #print(grid_src[0])

    inputs = Input((par,3))
    grid_inputs = Input((grid_size,grid_size))

    stn = SpatialTransformer(inputs,par,quat=True,norm=False)
    x = stn_transform(stn, inputs, quat=True)

    m = Model(inputs=[inputs,grid_inputs], outputs=x)

    #x = Flatten()(x)
    #x = Dense(grid_size*grid_size, activation='tanh')(x)
    #x = Reshape((grid_size,grid_size))(x)

    x = stn_grid_transform(stn, grid_inputs, quat=True)

    m2 = Model(inputs=[inputs,grid_inputs], outputs=x)#Model(inputs=inputs, outputs=InverseTransform(stn)(x))

    m.summary()
    m.compile( loss='mse', optimizer=keras.optimizers.adam(lr=0.001))
    m.fit(x=[src,grid_src],y=pos,epochs=2,batch_size=32)
    #print(pos[0:1])
    #print(m.predict(x=src[0:1]))
    #print(Model(inputs=inputs, outputs=stn).predict(x=src[0:1]))
    #plot_particles(pos[0], [-1,1], [-1,1], 1, ref=m.predict(x=src[0:1])[0])
    #plot_particles(pos[0], [-1,1], [-1,1], 1, ref=m2.predict(x=src[0:1])[0])

    plot_particles(src[0], [-1,1], [-1,1], 1, ref=m.predict(x=[src[:32],grid_src[:32]])[0])
    #plot_particles(src[0], [-1,1], [-1,1], 1, ref=m2.predict(x=src[0:1])[0])
    plot_sdf(grid_src[0], [0,grid_size], [0,grid_size], ref=m2.predict(x=[src[:32],grid_src[:32]], batch_size=32)[0])
