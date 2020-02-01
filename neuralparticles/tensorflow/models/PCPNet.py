from .architecture import Network

import numpy as np
import math

import keras
import keras.backend as K
from keras.layers import Input, multiply, concatenate, Conv1D, Lambda, add, Dropout, Dense, Reshape
from keras.models import Model, load_model

from neuralparticles.tensorflow.tools.spatial_transformer import SpatialTransformer, stn_transform, stn_tranform_inv
from neuralparticles.tensorflow.tools.zero_mask import zero_mask, soft_trunc_mask

from neuralparticles.tools.data_helpers import get_data

from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss
from neuralparticles.tensorflow.layers.mult_const_layer import MultConst
from neuralparticles.tensorflow.losses.repulsion_loss import repulsion_loss


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

class PCPNet(Network):
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

        self.norm_factor = kwargs.get("norm_factor")

    def _init_optimizer(self, epochs=1):
        self.optimizer = keras.optimizers.adam(lr=self.learning_rate, decay=self.decay)

    def _build_model(self):            
        input_xyz = Input((self.particle_cnt_src, 3), name="main_input")
        inputs = [input_xyz]

        if self.mask:
            mask = zero_mask(input_xyz, self.pad_val, name="mask")
            input_xyz = multiply([input_xyz, mask])

        stn_input = Input((self.particle_cnt_src,3))
        self.stn = SpatialTransformer(stn_input,self.particle_cnt_src,dropout=self.dropout,quat=True,norm=True)
        stn_model = Model(inputs=stn_input, outputs=self.stn, name="stn")
        self.stn = stn_model(input_xyz)

        input_xyz = stn_transform(self.stn,input_xyz,quat=True, name='trans')     

        if len(self.features) > 0:
            inputs.append(Input((self.particle_cnt_src, len(self.features) + (2 if 'v' in self.features else 0)), name="aux_input"))
            aux_input = MultConst(1./self.norm_factor)(inputs[1])
            if self.mask:
                aux_input = multiply([aux_input, mask]) 
            if 'v' in self.features:
                aux_input = Lambda(lambda a: concatenate([stn_transform(self.stn, a[:,:,:3]/100,quat=True),a[:,:,3:]], axis=-1), name='aux_trans')(aux_input)
            input_xyz = concatenate([input_xyz, aux_input], axis=-1, name='input_concatenation')

        x = Conv1D(self.fac, 1)(input_xyz)
        x = Conv1D(self.fac, 1)(x)

        x = stn_transform(SpatialTransformer(x,particle_cnt_src,fac,1),x)

        x = Conv1D(self.fac, 1)(x)
        x = Conv1D(self.fac*2, 1)(x)
        x = Conv1D(self.fac*4, 1)(x)

        x = multiply([x,mask])

        x = unstack(x,1)
        x = add(x)

        for i in range(self.particle_cnt_dst//self.particle_cnt_src):
            tmp = Conv1D(self.fac*32, 1, name="expansion_1_"+str(i+1), kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
            tmp = Conv1D(self.fac*16, 1, name="expansion_2_"+str(i+1), kernel_regularizer=keras.regularizers.l2(self.l2_reg))(tmp)
            l.append(tmp)
        x = concatenate(l, axis=1, name="pixel_conv") if self.particle_cnt_dst//self.particle_cnt_src > 1 else l[0]

        if self.truncate:
            x_t = Dropout(self.dropout)(x)
            x_t = Dense(self.fac, activation='elu', kernel_regularizer=keras.regularizers.l2(0.02), name="truncation_1")(x_t)
            x_t = Dropout(self.dropout)(x_t)
            b = np.ones(1, dtype='float32')
            W = np.zeros((self.fac, 1), dtype='float32')
            trunc = Dense(1, activation='elu', kernel_regularizer=keras.regularizers.l2(0.02), weights=[W,b], name="cnt")(x_t)
            out_mask = soft_trunc_mask(trunc, self.particle_cnt_dst, name="truncation_mask")

        if self.mask:
            x = Lambda(lambda v: v[0]/K.sum(v[1],axis=1))([x, mask])
        
        x = Dropout(self.dropout)(x)
        x = Dense(self.particle_cnt_dst, kernel_regularizer=keras.regularizers.l2(0.02))(x)
        x = Dropout(self.dropout)(x)
        x = Dense(self.particle_cnt_dst, kernel_regularizer=keras.regularizers.l2(0.02))(x)
        x = Dropout(self.dropout)(x)
        x = Dense(3*self.particle_cnt_dst, kernel_regularizer=keras.regularizers.l2(0.02))(x)
        
        x = Reshape((self.particle_cnt_dst,3))(x)
        out = stn_transform_inv(self.stn,x,quat=True)

        if self.truncate:
            out = multiply([out, Reshape((self.particle_cnt_dst,1))(out_mask)], name="masked_coords")
            self.model = Model(inputs=inputs, outputs=[out,trunc])
            trunc_exp = stack([trunc, trunc, trunc], 2)
            out = concatenate([out, trunc_exp], 1, name='points')
            self.train_model = Model(inputs=inputs, outputs=[out, trunc])
        else:
            self.model = Model(inputs=inputs, outputs=out)
            self.train_model = Model(inputs=inputs, outputs=out)

    def mask_loss(self, y_true, y_pred):
        if y_pred.get_shape()[1] > self.particle_cnt_dst:    
            return (emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred[:,:self.particle_cnt_dst]) if self.mask else emd_loss(y_true, y_pred[:,:self.particle_cnt_dst]))# / (y_pred[:,-1, 0])
        else:
            return emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred * zero_mask(y_true, self.pad_val)) if self.mask else emd_loss(y_true, y_pred)

    def particle_metric(self, y_true, y_pred):
        if y_pred.get_shape()[1] > self.particle_cnt_dst:    
            return (emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred[:,:self.particle_cnt_dst]) if self.mask else emd_loss(y_true, y_pred[:,:self.particle_cnt_dst]))
        elif y_pred.get_shape()[1] < self.particle_cnt_dst:
            return keras.losses.mse(y_true, y_pred)
        else:
            return emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred * zero_mask(y_true, self.pad_val)) if self.mask else emd_loss(y_true, y_pred)

    def compile_model(self):
        if self.truncate:
            self.train_model.compile(loss=[self.mask_loss, 'mse'], optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay), metrics=[self.particle_metric], loss_weights=[1.0,1.0])
        else:
            self.train_model.compile(loss=self.mask_loss, optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay))

    def _train(self, epochs, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        if "generator" in kwargs:
            return self.train_model.fit_generator(generator=kwargs['generator'], validation_data=kwargs.get('val_generator'), use_multiprocessing=True, workers=6, verbose=1, callbacks=callbacks, epochs=epochs)
        else:
            src_data = kwargs.get("src")
            ref_data = kwargs.get("ref")

            val_split = kwargs.get("val_split", 0.1)
            batch_size = kwargs.get("batch_size", 32)

            trunc_ref = np.count_nonzero(ref_data[:,:,:1] != self.pad_val, axis=1)/self.particle_cnt_dst 
            trunc_ref_exp = np.repeat(trunc_ref, 3, axis=-1)
            trunc_ref_exp = np.expand_dims(trunc_ref_exp, axis=1)
            
            return self.train_model.fit(x=src_data,y=[np.concatenate([ref_data, trunc_ref_exp], axis=1), trunc_ref] if self.truncate else ref_data, validation_split=val_split, 
                                epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)

    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path, custom_objects={'mask_loss': self.mask_loss, 'Interpolate': Interpolate, 'SampleAndGroup': SampleAndGroup, 'MultConst': MultConst})
        if self.truncate:
            out, trunc = self.model.outputs
            trunc_exp = stack([trunc, trunc, trunc], 2)
            out = concatenate([out, trunc_exp], 1, name='points')
            self.train_model = Model(inputs=self.model.inputs, outputs=[out, trunc])
        else:
            self.train_model = Model(self.model.inputs, self.model.outputs)
