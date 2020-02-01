import numpy as np
import keras
from neuralparticles.tools.plot_helpers import plot_particles
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

data_cnt = 10000
train_cnt = int(data_cnt * 0.9)
np.random.seed(10)

data = np.random.random((data_cnt,1,4))-0.5
data[:,0,:2] *= 0
data = np.repeat(data, 3, axis=1)

for t in range(1,3):
    data[:,t,:2] = data[:,t-1,:2] + data[:,t-1,2:] + np.random.normal(scale=0.01,size=(data_cnt,2))
    data[:,t,2:] = data[:,t-1,2:] + np.random.normal(scale=0.01,size=(data_cnt,2))

src = data[:,0]
src[...,:2] += (np.random.random((data_cnt,2)) - 0.5) * 0.1#normal(scale=0.01,size=(data_cnt,10,2))

inputs = keras.layers.Input((4,))
x = keras.layers.Dense((100))(inputs) 
x = keras.layers.Dense((100))(x)
x = keras.layers.Dense((6))(x)
x = keras.layers.Reshape((3,2))(x)
m = keras.models.Model(input=inputs, output=x)

m.compile(optimizer=keras.optimizers.adam(lr=0.001), loss=keras.losses.mse)

m.fit(src[:train_cnt,:4], data[:train_cnt,:,:2], epochs=3)

for t in range(train_cnt, train_cnt+10):
    s = src[t:t+1,:4]
    res = m.predict(s)
    plot_particles(res, xlim=[-1,1], ylim=[-1,1], s=5, ref=data[t], src=src[t:t+1])
    s[:,:2] += np.random.random((1,2)) * 0.1
    res = m.predict(s)
    plot_particles(res, xlim=[-1,1], ylim=[-1,1], s=5, ref=data[t], src=s)
