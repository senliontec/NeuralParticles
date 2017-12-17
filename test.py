import numpy as np
import sys
sys.path.append("manta/scenes/tools/")
from helpers import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, Dense
from keras.layers import Reshape, concatenate, add, Flatten, Lambda
from spatial_transformer import *
from split_dense import *

import math

np.random.seed(3)

cnt = 500
test_cnt = 10
epochs = 5
par_cnt = 50

use_quat=False

def rotate(pos, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    pos = np.matmul(pos,R)
    return pos

def gen_patch(N,theta=0):
    rot = np.random.rand(N)*math.pi
    rad = np.random.rand(N)

    x = np.cos(rot)*rad
    y = -np.sin(rot)*rad
    z = np.zeros((N,1))

    x = np.reshape(x,(N,1))
    y = np.reshape(y,(N,1))

    pos = np.concatenate([x,y,z],axis=1)

    if theta > 0:
        pos = rotate(pos, theta)
    return pos


#src = np.array([gen_patch(par_cnt for i in range(cnt)])
dst = np.array([gen_patch(par_cnt) for i in range(cnt)])
src = dst#np.array([rotate(d, np.random.rand()*2.*math.pi) for d in dst])
test = np.array([gen_patch(par_cnt) for i in range(test_cnt)])

for i in range(5):
    plt.scatter(src[i,:,0],src[i,:,1])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.savefig("test_out/src_%03d.png"%i)
    plt.clf()

for i in range(5):
    plt.scatter(dst[i,:,0],dst[i,:,1])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.savefig("test_out/dst_%03d.png"%i)
    plt.clf()

k = 128

inputs = Input((par_cnt,3), name="main")

x = SpatialTransformer(par_cnt,quat=use_quat)(inputs)

intermediate = x

x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(par_cnt)]

x = SplitDense(64, activation='tanh')(x)
x = SplitDense(64, activation='tanh')(x)

x = concatenate(x, axis=1)
x = SpatialTransformer(par_cnt,64,1)(x)

x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(par_cnt)]

x = SplitDense(64, activation='tanh')(x)
x = SplitDense(128, activation='tanh')(x)
x = SplitDense(k, activation='tanh')(x)

x = add(x)

x = Dense(512, activation='tanh')(x)
x = Dense(256, activation='tanh')(x)
x = Dense(3*par_cnt, activation='tanh')(x)

out = Reshape((par_cnt,3))(x)

model = Model(inputs=inputs, outputs=out)
model.compile( loss='mse', optimizer=keras.optimizers.adam(lr=0.001))
        
model.summary()

history=model.fit(x=src,y=dst,validation_split=0.2,epochs=epochs,batch_size=32)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.savefig("test_out/loss.png")
plt.clf()

interm = Model(inputs=inputs, outputs=intermediate)

result = model.predict(x=test,batch_size=32)
inter_result = interm.predict(x=test,batch_size=32)

for i in range(test_cnt):
    plt.scatter(test[i,:,0],test[i,:,1])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.savefig("test_out/test_%03d.png"%i)
    plt.clf()

    plt.scatter(result[i,:,0],result[i,:,1])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.savefig("test_out/res_%03d.png"%i)
    plt.clf()
    
    plt.scatter(inter_result[i,:,0],inter_result[i,:,1])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.savefig("test_out/inter_res_%03d.png"%i)
    plt.clf()
