import keras
from keras.layers.core import Layer
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dense, Reshape, Flatten
import numpy as np
from keras import backend as K

import tensorflow as tf
from .quaternion import quaternion_rot, quaternion_conj

def get_cell(grid, x, y):
    sh = tf.shape(x)
    bs = sh[0]
    s = sh[1]
    #w = sh[2]
    b = tf.range(0, bs)
    b = tf.reshape(b, (bs, 1))
    b = tf.tile(b, (1, s))
    idx = tf.stack([b, y, x], 2)

    if len(grid.get_shape()) > 3:
        vs = int(grid.get_shape()[3])

        v = tf.range(0, vs)
        v = tf.reshape(v, (1, 1, vs, 1))
        v = tf.tile(v, (bs, s, 1, 1))

        idx = K.expand_dims(idx,axis=2)
        idx = tf.tile(idx, (1, 1, vs, 1))

        idx = tf.concat([idx,v],axis=-1)

    return tf.gather_nd(grid, idx)

def interpol(grid, pos):
    max_x = tf.cast(tf.shape(grid)[1] - 1, 'int32')
    max_y = tf.cast(tf.shape(grid)[2] - 1, 'int32')
    zero = tf.zeros([], dtype='float32')

    pos = tf.maximum(zero,pos-0.5)
    
    x0 = tf.minimum(K.cast(pos[:,:,0], 'int32'), max_x)
    y0 = tf.minimum(K.cast(pos[:,:,1], 'int32'), max_y)
    x1 = tf.minimum(x0+1, max_x)
    y1 = tf.minimum(y0+1, max_y)

    facX = pos[:,:,0]-tf.floor(pos[:,:,0])
    facY = pos[:,:,1]-tf.floor(pos[:,:,1])

    while len(facX.get_shape()) < len(grid.get_shape())-1:
        facX = K.expand_dims(facX)
        facY = K.expand_dims(facY)

    v  = get_cell(grid, x0, y0) * (1-facX) * (1-facY)
    v += get_cell(grid, x1, y0) * facX * (1-facY)
    v += get_cell(grid, x0, y1) * (1-facX) * facY
    v += get_cell(grid, x1, y1) * facX * facY
    return v

def grid_centers(bs, size):
    num = tf.cast(size, "int32")
    size = tf.cast(size,"float32")
    x = tf.linspace(0.5,size-0.5,num)
    y = tf.linspace(0.5,size-0.5,num)
    x, y = tf.meshgrid(x, y)
    z = tf.zeros_like(x)

    sg = tf.stack([K.flatten(x),K.flatten(y),K.flatten(z)],axis=-1)

    sg = tf.expand_dims(sg, axis=0)
    sg = tf.tile(sg, tf.stack([bs, 1, 1]))

    return sg

def advection(grid, vec):
    sh = tf.shape(grid)
    idx = grid_centers(sh[0], sh[1])[:,:,:2] - K.reshape(vec, (-1,sh[1]*sh[2],2))
    return K.reshape(interpol(grid, idx), (-1,sh[1],sh[2]))

def rotate_grid(grid, quat):
    sh = tf.shape(grid)
    bs = sh[0]
    si = sh[1]
    size = tf.cast(si, 'float32')

    sg = grid_centers(bs,si) - size/2
    sg = quaternion_rot(sg,quaternion_conj(quat)) + size/2

    return K.reshape(interpol(grid, sg),(-1,si,si))

def rotate_vec_grid(grid, quat):
    sh = tf.shape(grid)
    bs = sh[0]
    si = sh[1]
    size = tf.cast(si, 'float32')

    grid = K.reshape(quaternion_rot(K.reshape(grid,(-1,si*si,3)), quat), (-1,si,si,3))

    sg = grid_centers(bs,si) - size/2
    sg = quaternion_rot(sg,quaternion_conj(quat)) + size/2

    return K.reshape(interpol(grid, sg),(-1,si,si,3))

def transform_grid(grid, mt):
    sh = tf.shape(grid)
    bs = sh[0]
    si = sh[1]
    size = tf.cast(si, 'float32')

    sg = grid_centers(bs,si) - size/2
    sg = K.batch_dot(sg, tf.matrix_inverse(mt)) + size/2

    return K.reshape(interpol(grid, sg),(-1,si,si))

def transform_vec_grid(grid, mt):
    sh = tf.shape(grid)
    bs = sh[0]
    si = sh[1]
    size = tf.cast(si, 'float32')

    grid = K.reshape(K.batch_dot(K.reshape(grid,(-1,si*si,3)), mt), (-1,si,si,3))

    sg = grid_centers(bs,si) - size/2
    sg = K.batch_dot(sg, tf.matrix_inverse(mt)) + size/2

    return K.reshape(interpol(grid, sg),(-1,si,si, 3))

def rotation_grid(quat, size):
    bs = tf.shape(quat)[0]

    sg = grid_centers(bs, size) - size/2
    trans_sg = quaternion_rot(sg, quaternion_conj(quat))

    return K.reshape(sg-trans_sg, (-1,size,size,3))[:,:,:,:2]

if __name__ == "__main__":
    a = K.constant(np.array([[[[0,1],[2,3]],[[4,5],[6,7]]]]))
    print(a.get_shape())
    idx = K.constant(np.array([[[0,0],[2,2]]]))
    print(idx.get_shape())
    print(K.eval(interpol(a,idx)))
    exit()
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
    def tmp(v):
        return advection(v[0],v[1])
    x = Lambda(tmp)([inputs,vec])

    m = Model(inputs=inputs, outputs=x)
    
    m.compile( loss='mse', optimizer=keras.optimizers.adam(lr=0.001))

    m.summary()

    m.fit(x=src,y=dst,epochs=20,batch_size=batch_size)

    print(np.floor(m.predict(x=src,batch_size=batch_size)[0]*10))
    #print(Model(inputs=inputs, output=vec).predict(x=src,batch_size=batch_size)[0])
    
