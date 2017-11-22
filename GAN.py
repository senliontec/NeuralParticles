import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Input, Flatten, Dense
from keras.layers import Reshape, RepeatVector, Permute, concatenate, add, Activation, Dropout, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras_utils.subpixel import *
import numpy as np

inputs = Input((5, 5, 1), name="main_input")
aux_input = Input((5,5,3),name="aux_input")

base = Reshape((5*5,), name="reshape_flat")(inputs)
base = RepeatVector(9, name="repeate")(base)
base = Permute((2, 1), name="permute")(base)
base = Reshape((5,5,9), name="reshape_back")(base)

x = concatenate([inputs, aux_input], name="concatenate")
x = Reshape((5*5*4,), name="reshape_flat_res")(x)
x = RepeatVector(9, name="repeate_res")(x)
x = Permute((2, 1), name="permute_res")(x)
x = Reshape((5, 5,9*4), name="reshape_back_res")(x)

x = Conv2D(filters=16*4, kernel_size=3, 
           strides=1, activation='tanh', padding='same', name="conv2D_0")(base)
x = BatchNormalization(name="normalize_0")(x)
x = Conv2D(filters=32*4, kernel_size=3,
           strides=1, activation='tanh', padding='same', name="conv2D_1")(x)    
x = BatchNormalization(name="normalize_1")(x)
x = Conv2DTranspose(filters=16*4, kernel_size=3, 
                    strides=1, activation='tanh', padding='same', name="deconv2D_0")(x)
x = BatchNormalization(name="normalize_2")(x)
x = Conv2DTranspose(filters=9, kernel_size=3, 
                    strides=1, activation='tanh', padding='same', name="deconv2D_1")(x)
x = BatchNormalization(name="normalize_3")(x)

x = add([base,x], name="add")
x = Activation('tanh', name="activation")(x)
predictions = Subpixel(filters=1, kernel_size=3, r=3,activation='tanh', padding='same', name="subpixel_conv")(x)

generator = Model(inputs=[inputs,aux_input], outputs=predictions)
generator.summary()
generator.compile( loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=0.001))

model = Sequential()

img_shape = (15, 15, 1)

model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
model.add(ZeroPadding2D(padding=((0,1),(0,1))))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization(momentum=0.8))
model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization(momentum=0.8))
model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
model.add(LeakyReLU(alpha=0.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

img = Input(shape=img_shape)
validity = model(img)

model.summary()
discriminator = Model(img, validity)

discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=0.001), metrics=['accuracy'])

z = Input(shape=(5,5,1))
z_aux = Input(shape=(5,5,3))
img = generator([z,z_aux])

# For the combined model we will only train the generator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator)
combined = Model([z,z_aux], valid)
combined.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=0.001), metrics=['accuracy'])

import sys
sys.path.append("2D_SPH/scenes/tools")

from dataset import Dataset

src_patches_path = "2D_data/patches/lowres/sph_2D_v02-01_d%03d_var%02d_%03d"
ref_patches_path = "2D_data/patches/highres/ref_sph_2D_v02-01_d%03d_var%02d_%03d"

train_data = Dataset(src_patches_path, 0, 18, 5, 15, ['sdf','vel'], 1, ref_patches_path, ['sdf'])
test_data = Dataset(src_patches_path, 18, 20, 5, 15, ['sdf','vel'], 1, ref_patches_path, ['sdf'])

print(train_data.data.shape)
print(train_data.ref_data.shape)

batch_size = 32
epochs = 1#train_data.data.shape[0]//batch_size*250
half_batch = batch_size//2

for ep in range(epochs):
    x = train_data.get_batch(half_batch)[1]
    y = np.split(train_data.get_batch(half_batch)[0],[1],axis=3)
    y = generator.predict(y)
    
    d_loss_real = discriminator.train_on_batch(x, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(y, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    x = np.split(train_data.get_batch(batch_size)[0],[1],axis=3)
    g_loss = combined.train_on_batch(x, np.ones((batch_size, 1)))
    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (ep, d_loss[0], 100*d_loss[1], g_loss[0], 100*g_loss[1]))
    
combined.save('2D_data/model/gan_v3.h5')