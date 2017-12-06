import sys
sys.path.append("manta/scenes/tools/")

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Conv1D, Conv2DTranspose, BatchNormalization, Input, ZeroPadding2D, Dense, MaxPooling2D, Lambda
from keras.layers import Reshape, RepeatVector, Permute, concatenate, add, Activation, Flatten
from keras import regularizers
import numpy as np
from spatial_transformer import SpatialTransformer 
from split_dense import SplitDense

from keras.layers.core import Layer

def test_locnet(cnt,b=None):
    m = Sequential()
    m.add(Flatten(input_shape=(cnt,3)))
    if b is None:
        b = np.eye(3, dtype='float32').flatten()
    W = np.zeros((cnt*3, 9), dtype='float32')
    m.add(Dense(9, weights=[W,b]))
    m.add(Reshape((3,3)))
    return m

k = 128
par_cnt = 2
out_cnt = 2

inputs = Input((par_cnt,3), name="main")

print(inputs.get_shape())

x = SpatialTransformer(par_cnt)(inputs)
print(x.get_shape())
x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(par_cnt)]
print(x[0].get_shape())

x = SplitDense(64)(x)

x = add(x)

'''
main_input = Input((in_cnt,3))
inputs = [(Lambda(lambda x: x[:,i:i+1,:])(main_input)) for i in range(in_cnt)]

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

x = Reshape((out_cnt,3))(x)'''

model = Model(inputs=inputs, outputs=x)
model.compile(loss='mse', optimizer=keras.optimizers.adam(lr=0.001))

model.summary()

model.save("test.h5")
model = load_model("test.h5", custom_objects={"SpatialTransformer":SpatialTransformer, "SplitDense":SplitDense})

particles = np.array([[[3,4,6],[1,2,3]]])

print(particles)
print(model.predict(particles, batch_size=1))