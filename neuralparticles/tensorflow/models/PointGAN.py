from .architecture import Network

import numpy as np
import math

import keras
from keras.layers import Input, multiply, concatenate, Conv1D, Lambda, add, Dropout, Dense, Reshape, RepeatVector, Flatten, Permute
from keras.models import Model, load_model

from neuralparticles.tensorflow.tools.pointnet_util import pointnet_sa_module, pointnet_fp_module, Interpolate, SampleAndGroup
from neuralparticles.tensorflow.tools.zero_mask import zero_mask, soft_trunc_mask
from neuralparticles.tools.data_augmentation import *

from neuralparticles.tools.data_helpers import get_data

from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss, approx_match, match_cost
#from neuralparticles.tensorflow.losses.tf_auctionmatch import emd_loss as emd_loss
from neuralparticles.tensorflow.layers.mult_const_layer import MultConst
from neuralparticles.tensorflow.layers.add_const_layer import AddConst
from neuralparticles.tensorflow.losses.repulsion_loss import repulsion_loss, get_repulsion_loss4


def stack(X, axis, **kwargs):
    def tmp(X):
        import tensorflow as tf
        return tf.stack(X,axis=axis)
    return Lambda(tmp, **kwargs)(X)

def unstack(X, axis, **kwargs):
    def tmp(X):
        import tensorflow as tf
        return tf.unstack(X,axis=axis)
    return Lambda(tmp, **kwargs)(X)

def extract_xyz(X, **kwargs):
    return Lambda(lambda x: x[...,:3], **kwargs)(X)

def extract_aux(X, **kwargs):
    return Lambda(lambda x: x[...,3:], **kwargs)(X)

class PUNet(Network):
    def _init_vars(self, **kwargs):
        self.model = None

        self.decay = kwargs.get("decay", 0.0)
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.fac = kwargs.get("fac", 32)
        self.dropout = kwargs.get("dropout", 0.2)
        self.l2_reg = kwargs.get("l2_reg", 0.0)

        self.pad_val = kwargs.get("pad_val", 0.0)
        self.mask = kwargs.get("mask", False)
        
        self.particle_cnt = kwargs.get("par_cnt_ref")


    def _init_optimizer(self, epochs=1):
        self.optimizer = keras.optimizers.adam(lr=self.learning_rate, decay=self.decay)

    def _build_model(self):            
        activation = keras.activations.tanh#lambda x: keras.activations.relu(x, alpha=0.1)
        inputs = Input((self.particle_cnt, 3), name="main_input")
        input_xyz = extract_xyz(inputs, name="extract_pos")
        input_points = input_xyz
        
        if not self.mask:
            mask = zero_mask(input_xyz, self.pad_val, name="mask_1")
            input_points = multiply([input_points, mask])
            input_xyz = multiply([input_xyz, mask])
            input_xyz_m = input_xyz
            input_points_m = input_points


        l1_xyz, l1_points = pointnet_sa_module(input_xyz, input_points, self.particle_cnt, 0.25, self.fac*4, 
                                               [self.fac*4,
                                                self.fac*4,
                                                self.fac*8], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        l2_xyz, l2_points = pointnet_sa_module(l1_xyz, l1_points, self.particle_cnt//2, 0.5, self.fac*4, 
                                               [self.fac*8,
                                                self.fac*8,
                                                self.fac*16], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        l3_xyz, l3_points = pointnet_sa_module(l2_xyz, l2_points, self.particle_cnt//4, 0.6, self.fac*4, 
                                               [self.fac*16,
                                                self.fac*16,
                                                self.fac*32], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        l4_xyz, l4_points = pointnet_sa_module(l3_xyz, l3_points, self.particle_cnt//8, 0.7, self.fac*4, 
                                               [self.fac*32,
                                                self.fac*32,
                                                self.fac*64], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))


        if self.mask:
            mask = zero_mask(input_xyz, self.pad_val, name="mask_2")
            input_xyz_m = multiply([input_xyz, mask])
            input_points_m = multiply([input_points, mask])

        # interpoliere die features in l2_points auf die Punkte in x
        up_l2_points = pointnet_fp_module(input_xyz_m, l2_xyz, None, l2_points, [self.fac*8], kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)
        up_l3_points = pointnet_fp_module(input_xyz_m, l3_xyz, None, l3_points, [self.fac*8], kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)
        up_l4_points = pointnet_fp_module(input_xyz_m, l4_xyz, None, l4_points, [self.fac*8], kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)

        x = concatenate([up_l4_points, up_l3_points, up_l2_points, l1_points, input_points_m], axis=-1)

        x = add(x)
        x = Dense(self.fac*32, activation=activation)(x)
        x = Dense(self.fac*16, activation=activation)(x)
        x = Dense(1, activation=keras.activations.sigmoid)(x)
        out = x
        self.model = Model(inputs=inputs, outputs=out)
        
    def compile_model(self):
        self.model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay))

    def _train(self, epochs, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        if "generator" in kwargs:
            return self.model.fit_generator(generator=kwargs['generator'], validation_data=kwargs.get('val_generator'), use_multiprocessing=False, workers=1, verbose=1, callbacks=callbacks, epochs=epochs, shuffle=False)
        else:
            src_data = kwargs.get("src")
            ref_data = [kwargs.get("ref")]
            
            val_split = kwargs.get("val_split", 0.1)
            batch_size = kwargs.get("batch_size", 32)

            return self.model.fit(x=src_data,y=ref_data, validation_split=val_split, 
                                epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)

    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        if self.model:
            self.model.load_weights(path)
        else:
            self.model = load_model(path, custom_objects={'Interpolate': Interpolate, 'SampleAndGroup': SampleAndGroup, 'MultConst': MultConst, 'AddConst': AddConst})