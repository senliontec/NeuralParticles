from .architecture import Network

import numpy as np
import math

import keras
from keras.layers import Input, multiply, concatenate, Conv1D, Lambda, add, Dropout, Dense, Reshape
from keras.models import Model, load_model

from neuralparticles.tensorflow.tools.pointnet_util import pointnet_sa_module, pointnet_fp_module, Interpolate, SampleAndGroup
from neuralparticles.tensorflow.tools.zero_mask import zero_mask, trunc_mask

from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss

class PUNet(Network):
    def _init_vars(self, **kwargs):
        self.model = None

        self.decay = kwargs.get("decay", 0.0)
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.fac = kwargs.get("fac", 32)
        self.dropout = kwargs.get("dropout", 0.2)
        self.l2_reg = kwargs.get("l2_reg", 0.0)

        self.truncate = kwargs.get("truncate", False)
        self.pad_val = kwargs.get("pad_val", 0.0)
        self.mask = kwargs.get("mask", False)
        
        self.particle_cnt_src = kwargs.get("par_cnt")
        self.particle_cnt_dst = kwargs.get("par_cnt_ref")

        self.features = kwargs.get("features")
        self.dim = kwargs.get("dim", 2)
        self.factor = kwargs.get("factor")
        self.factor_d = math.pow(self.factor, 1/self.dim)

        self.res = kwargs.get("res")
        self.lres = int(self.res/self.factor_d)

    def _init_optimizer(self, epochs=1):
        self.optimizer = keras.optimizers.adam(lr=self.learning_rate, decay=self.decay)

    def _build_model(self):

        def stack(X):
            import tensorflow as tf
            return tf.stack(X,axis=1)

        def unstack(X):
            import tensorflow as tf
            return tf.unstack(X,axis=1)
            
        inputs = [Input((self.particle_cnt_src, 3), name="main_input")]

        x = inputs[0]
        l1_xyz, l1_points = pointnet_sa_module(x, None, self.particle_cnt_src, 0.25, self.fac*4, 
                                               [self.fac*4,
                                                self.fac*4,
                                                self.fac*8], mask_val=self.pad_val if self.mask else None)
        l2_xyz, l2_points = pointnet_sa_module(l1_xyz, l1_points, self.particle_cnt_src//2, 0.5, self.fac*4, 
                                               [self.fac*8,
                                                self.fac*8,
                                                self.fac*16])
        l3_xyz, l3_points = pointnet_sa_module(l2_xyz, l2_points, self.particle_cnt_src//4, 0.6, self.fac*4, 
                                               [self.fac*16,
                                                self.fac*16,
                                                self.fac*32])
        l4_xyz, l4_points = pointnet_sa_module(l3_xyz, l3_points, self.particle_cnt_src//8, 0.7, self.fac*4, 
                                               [self.fac*32,
                                                self.fac*32,
                                                self.fac*64])

        if self.mask:
            mask = zero_mask(x, self.pad_val, name="mask_2")
            x = multiply([x, mask])

        # interpoliere die features in l2_points auf die Punkte in x
        up_l2_points = pointnet_fp_module(x, l2_xyz, None, l2_points, [self.fac*8])
        up_l3_points = pointnet_fp_module(x, l3_xyz, None, l3_points, [self.fac*8])
        up_l4_points = pointnet_fp_module(x, l4_xyz, None, l4_points, [self.fac*8])

        x = concatenate([up_l4_points, up_l3_points, up_l2_points, l1_points, x], axis=-1)
        l = []
        for i in range(self.particle_cnt_dst//self.particle_cnt_src):
            tmp = Conv1D(self.fac*32, 1, name="expansion_1_"+str(i+1))(x)
            tmp = Conv1D(self.fac*16, 1, name="expansion_2_"+str(i+1))(tmp)
            l.append(tmp)
        x = concatenate(l, axis=1, name="pixel_conv") if self.particle_cnt_dst//self.particle_cnt_src > 1 else l[0]

        if self.truncate:
            x_t = Lambda(unstack, name='unstack')(x)
            x_t = add(x_t, name='merge_features')
            x_t = Dropout(self.dropout)(x_t)
            x_t = Dense(self.fac, activation='elu', kernel_regularizer=keras.regularizers.l2(self.l2_reg), name="truncation_1")(x_t)
            x_t = Dropout(self.dropout)(x_t)
            b = np.ones(1, dtype='float32')
            W = np.zeros((self.fac, 1), dtype='float32')
            trunc = Dense(1, activation='elu', kernel_regularizer=keras.regularizers.l2(self.l2_reg), weights=[W,b], name="truncation_2")(x_t)
            out_mask = trunc_mask(trunc, self.particle_cnt_dst, name="truncation_mask")

        x = Conv1D(self.fac*8, 1, name="coord_reconstruction_1")(x)
        x = Conv1D(3, 1, name="coord_reconstruction_2")(x)
        
        out = x

        if self.truncate:
            out = multiply([out, Reshape((self.particle_cnt_dst,1))(out_mask)], name="masked_coords")
            self.model = Model(inputs=inputs, outputs=[out,trunc])
        else:
            self.model = Model(inputs=inputs, outputs=out)

    def _compile_model(self):
        def mask_loss(y_true, y_pred):
            return emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred) if self.mask else emd_loss(y_true, y_pred)
        if self.truncate:
            self.model.compile(loss=[mask_loss, 'mse'], optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay), loss_weights=[1.0,1.0])
        else:
            self.model.compile(loss=mask_loss, optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay))

    def _train(self, epochs, **kwargs):
        src_data = kwargs.get("src")
        ref_data = kwargs.get("ref")

        val_split = kwargs.get("val_split", 0.1)
        batch_size = kwargs.get("batch_size", 32)

        callbacks = kwargs.get("callbacks", [])

        trunc_ref = np.count_nonzero(ref_data[:,:,:1] != self.pad_val, axis=1)/self.particle_cnt_dst
        
        return self.model.fit(x=src_data,y=[ref_data, trunc_ref] if self.truncate else ref_data, validation_split=val_split, 
                              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)

    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        def mask_loss(y_true, y_pred):
            return emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred) if self.mask else emd_loss(y_true, y_pred)
        self.model = load_model(path, custom_objects={'mask_loss': mask_loss, 'Interpolate': Interpolate, 'SampleAndGroup': SampleAndGroup})