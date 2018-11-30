import numpy as np
import keras
from neuralparticles.tools.plot_helpers import plot_particles

data_cnt = 10000
train_cnt = int(data_cnt * 0.9)
np.random.seed(10)

data = np.random.random((data_cnt,1,4))-0.5
data = np.repeat(data, 10, axis=1)

for t in range(1,10):
    data[:,t,:2] = data[:,t-1,:2] + data[:,t-1,2:]
    data[:,t,2:] = data[:,t-1,2:] + np.random.normal(scale=0.1,size=(data_cnt,2))

src = data.copy()
src[...,:2] += (np.random.random((data_cnt,10,2)) - 0.5) * 0.1#normal(scale=0.01,size=(data_cnt,10,2))


inputs = keras.layers.Input((4,))
x = keras.layers.Dense((100))(inputs) 
x = keras.layers.Dense((100))(x)
x = keras.layers.Dense((2))(x)
m = keras.models.Model(input=inputs, output=x)

m.compile(optimizer=keras.optimizers.adam(lr=0.001), loss=keras.losses.mse)

train_x = np.concatenate(src[train_cnt:],axis=0)
train_y = np.concatenate(data[train_cnt:],axis=0)
print(train_x.shape)
print(train_y.shape)
m.fit(train_x, train_y[...,:2], epochs=2)


res = m.predict(src[train_cnt])

plot_particles(res, s=5, ref=data[train_cnt], src=src[train_cnt])