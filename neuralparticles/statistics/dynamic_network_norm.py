import numpy as np 
import os

import keras
import keras.backend as K
from keras.layers import Lambda, Add, Average, Input, Masking, Reshape, Dense, Flatten, Multiply
from keras.models import Model

import tensorflow as tf

np.random.seed(23423)

data_size = 100
data_cnt = 4096

mask_val = -1.0

def gen_data(cnt):
    i_cnt = int(cnt*data_size)
    return np.concatenate((np.ones((i_cnt,3))*1.0, mask_val * np.ones((data_size-i_cnt,3))))

def gen_mask(cnt):
    r = K.arange(data_size, dtype=K.floatx())+0.5
    return K.sigmoid((cnt*data_size-r)*10)

test = K.constant([4.5/data_size])
func = gen_mask(test)
print(K.eval(tf.gradients(func, test)[0]))

def zero_mask(inputs):
    mask = K.expand_dims(K.any(K.equal(inputs, mask_val),axis=-1), axis=-1)
    return 1 - K.cast(mask, K.floatx())

def mask_loss(y_true, y_pred):
    return keras.losses.mse(y_true * zero_mask(y_true), y_pred)

inputs = Input((data_size,3))

l = [(Lambda(lambda v: v[:,i,:])(inputs)) for i in range(data_size)]

l = list(map(Dense(10, activation='tanh'),l))

mask = Lambda(zero_mask)(inputs)
l = [Lambda(lambda v: v * mask[:,i])(l[i]) for i in range(data_size)]

l = Average()(l)

l = Dense(10)(l)
trunc = Dense(1)(l)

l = Lambda(gen_mask)(trunc)

l = Multiply()([inputs, Reshape((data_size,1))(l)])

m = Model(inputs=inputs, outputs=[l,trunc])

m.compile(loss=[mask_loss,'mse'],optimizer=keras.optimizers.adam(lr=0.001), loss_weights=[1.,0.])

val_x = np.array([gen_data(0.2)])
val_y = [val_x, np.array([[0.2]])]

print(m.predict(val_x)[1]*data_size)
print("Without masking:", np.mean(K.eval(keras.losses.mse(val_x,np.zeros(val_x.shape)))))
print("With masking:", m.evaluate(val_x,val_y,verbose=0))

print("training")

rand_cnts = np.random.rand(data_cnt)
x = np.array(list(map(gen_data,rand_cnts)))
y = [x, rand_cnts]

m.fit(x,y,batch_size=32,epochs=15)

print(m.predict(val_x)[1]*data_size)
print("Without masking:", np.mean(K.eval(keras.losses.mse(val_x,np.zeros(val_x.shape)))))
print("With masking:", m.evaluate(val_x,val_y,verbose=0))



