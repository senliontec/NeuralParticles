import keras
from keras.layers.core import Layer
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Reshape, Flatten
import numpy as np
from keras import backend as K

import tensorflow as tf
from quaternion_mul import quaternion_rot, quaternion_conj, quaternion_norm
from helpers import *

def get_cell(grid, x, y):
    sh = tf.shape(x)
    bs = sh[0]
    h = sh[1]
    w = sh[2]

    batch_idx = tf.range(0, bs)
    batch_idx = tf.reshape(batch_idx, (bs, 1, 1))
    b = tf.tile(batch_idx, (1, h, w))

    idx = tf.stack([b, y, x], 3)

    return tf.gather_nd(grid, idx)

def interpol(sdf, pos):
    pos -= 0.5
    max_x = tf.cast(tf.shape(pos)[1] - 1, 'int32')
    max_y = tf.cast(tf.shape(pos)[2] - 1, 'int32')
    zero = tf.zeros([], dtype='int32')
    
    x0 = K.cast(pos[:,:,:,0], 'int32')
    y0 = K.cast(pos[:,:,:,1], 'int32')
    x1 = x0+1
    y1 = y0+1

    x0 = tf.clip_by_value(x0, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y1 = tf.clip_by_value(y1, zero, max_y)

    facX = pos[:,:,:,0]-tf.floor(pos[:,:,:,0])
    facY = pos[:,:,:,1]-tf.floor(pos[:,:,:,1])

    v  = get_cell(sdf, x0, y0) * (1-facX) * (1-facY)
    v += get_cell(sdf, x1, y0) * facX * (1-facY)
    v += get_cell(sdf, x0, y1) * (1-facX) * facY
    v += get_cell(sdf, x1, y1) * facX * facY

    return v

def advection(grid, vec):
    idx = K.constant(np.array([[[[x,y] for x in range(grid.get_shape()[1])] for y in range(grid.get_shape()[2])]], dtype='float32') + 0.5) - vec
    return interpol(grid, idx)

def linear_grid_like(bs, size):
    x = tf.linspace(-size/2+0.5,size/2-0.5,size)
    y = tf.linspace(-size/2+0.5,size/2-0.5,size)
    x, y = tf.meshgrid(x, y)
    z = tf.zeros_like(x)

    sg = tf.stack([K.flatten(x),K.flatten(y),K.flatten(z)],axis=-1)

    sg = tf.expand_dims(sg, axis=0)
    sg = tf.tile(sg, tf.stack([bs, 1, 1]))

    return sg

def rotate_grid(grid, quat):
    bs = tf.shape(grid)[0]
    size = int(grid.get_shape()[1])

    sg = linear_grid_like(bs,size)
    sg = quaternion_rot(sg,quaternion_conj(quat)) + size/2

    return interpol(grid, tf.reshape(sg,(-1,size,size,3)))

def transform_grid(grid, mt):
    bs = tf.shape(grid)[0]
    size = int(grid.get_shape()[1])

    sg = linear_grid_like(bs,size)
    sg = K.batch_dot(sg, tf.matrix_inverse(mt)) + size/2

    return interpol(grid, tf.reshape(sg,(-1,size,size,3)))

def rotation_grid(quat, size):
    bs = tf.shape(quat)[0]

    sg = linear_grid_like(bs, size)
    trans_sg = quaternion_rot(sg, quaternion_conj(quat))

    return K.reshape(sg-trans_sg, (-1,size,size,3))[:,:,:,:2]

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

    vec = rotation_grid(K.constant(np.array([[0.7,0,0,0.7]]*2), dtype="float32"), 10)# K.reshape(np.array([2,0]*100*batch_size, dtype='float32'), (batch_size,10,10,2))

    #plot_vec(K.eval(vec[0]*0.1), [0,10], [0,10])
    print(K.eval(K.constant(src[0:1])))
    print((K.eval(advection(K.constant(src[:2]),vec[:2]))[0]*10).astype('int32'))

    print((K.eval(rotate_grid(K.constant(src[:1]),K.constant(np.array([[0.7,0,0,0.7]]))))*10).astype('int32'))

    #print(K.eval(rotation_grid(K.constant(np.array([[0.7071068,0,0,0.7071068]]*2), dtype="float32"), 5, batch_size)))

    inputs = Input((10,10))
    x = Flatten()(inputs)
    x = Dense(100, activation='tanh')(x)
    b = np.zeros((200,), dtype='float32')
    W = np.zeros((100, 200), dtype='float32')
    x = Dense(200, activation='tanh', weights=[W,b])(x)

    vec = Reshape((10,10,2))(x)
    x = Lambda(advection)([inputs,vec])

    m = Model(inputs=inputs, outputs=x)
    
    m.compile( loss='mse', optimizer=keras.optimizers.adam(lr=0.001))

    m.summary()

    m.fit(x=src,y=dst,epochs=20,batch_size=batch_size)

    print(np.floor(m.predict(x=src,batch_size=batch_size)[0]*10))
    #print(Model(inputs=inputs, output=vec).predict(x=src,batch_size=batch_size)[0])
    
