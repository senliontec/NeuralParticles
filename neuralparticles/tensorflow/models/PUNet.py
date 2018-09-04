from .architecture import Network

import numpy as np
import math

import keras
from keras.layers import Input, multiply, concatenate, Conv1D, Lambda, add, Dropout, Dense, Reshape, RepeatVector, Flatten
from keras.models import Model, load_model


from neuralparticles.tensorflow.tools.pointnet_util import pointnet_sa_module, pointnet_fp_module, Interpolate, SampleAndGroup
from neuralparticles.tensorflow.tools.zero_mask import zero_mask, trunc_mask

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
        inputs = Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0)), name="main_input")
        input_xyz = extract_xyz(inputs)

        input_points = input_xyz

        if len(self.features) > 0:
            aux_input = extract_aux(inputs)
            aux_input = MultConst(1./self.norm_factor)(aux_input)
            input_points = concatenate([input_xyz, aux_input], axis=-1, name='input_concatenation')

        l1_xyz, l1_points = pointnet_sa_module(input_xyz, input_points, self.particle_cnt_src, 0.25, self.fac*4, 
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
            mask = zero_mask(input_points, self.pad_val, name="mask_2")
            input_points = multiply([input_points, mask])
            input_xyz = multiply([input_xyz, mask])

        # interpoliere die features in l2_points auf die Punkte in x
        up_l2_points = pointnet_fp_module(input_xyz, l2_xyz, None, l2_points, [self.fac*8])
        up_l3_points = pointnet_fp_module(input_xyz, l3_xyz, None, l3_points, [self.fac*8])
        up_l4_points = pointnet_fp_module(input_xyz, l4_xyz, None, l4_points, [self.fac*8])

        x = concatenate([up_l4_points, up_l3_points, up_l2_points, l1_points, input_points], axis=-1)
        l = []
        for i in range(self.particle_cnt_dst//self.particle_cnt_src):
            tmp = Conv1D(self.fac*32, 1, name="expansion_1_"+str(i+1), kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
            tmp = Conv1D(self.fac*16, 1, name="expansion_2_"+str(i+1), kernel_regularizer=keras.regularizers.l2(self.l2_reg))(tmp)
            l.append(tmp)
        x_t = concatenate(l, axis=1, name="pixel_conv") if self.particle_cnt_dst//self.particle_cnt_src > 1 else l[0]

        x = Conv1D(self.fac*8, 1, name="coord_reconstruction_1")(x_t)

        b = np.zeros((3,), dtype='float32')
        W = np.zeros((1, self.fac*8, 3), dtype='float32')
        x = Conv1D(3, 1, name="coord_reconstruction_2")(x)#, weights=[W,b])(x)
        
        out = x

        '''x = Flatten()(input_xyz)
        x = RepeatVector(self.particle_cnt_dst//self.particle_cnt_src)(x)
        x = Reshape((self.particle_cnt_dst, 3))(x)

        out = add([x, out])'''

        if self.truncate:
            x_t = unstack(x_t, 1, name='unstack')
            x_t = add(x_t, name='merge_features')
            #x_t = Dropout(self.dropout)(x_t)
            x_t = Dense(self.fac, activation='elu', kernel_regularizer=keras.regularizers.l2(0.02), name="truncation_1")(x_t)
            #x_t = Dropout(self.dropout)(x_t)
            b = np.ones(1, dtype='float32')
            W = np.zeros((self.fac, 1), dtype='float32')
            trunc = Dense(1, activation='elu', kernel_regularizer=keras.regularizers.l2(0.02), weights=[W,b], name="cnt")(x_t)
            #trunc = Input((1,))
            #inputs.append(trunc)
            out_mask = trunc_mask(trunc, self.particle_cnt_dst, name="truncation_mask")

            self.short_model = Model(inputs=inputs, outputs=out)
            out = multiply([out, Reshape((self.particle_cnt_dst,1))(out_mask)], name="masked_coords")
            self.model = Model(inputs=inputs, outputs=[out,trunc])

            inputs = [Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0))),Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0)))]
            out, trunc = self.model(inputs[0])
            trunc_exp = stack([trunc, trunc, trunc], 2)
            out = concatenate([out, self.model(inputs[1])[0], trunc_exp], axis=1, name='points')
            self.train_model = Model(inputs=inputs, outputs=[out, trunc])
        else:
            self.model = Model(inputs=inputs, outputs=out)
            self.train_model = Model(inputs=inputs, outputs=out)

#16287/16287 [==============================] - 930s 57ms/step - loss: 0.1596 - concatenate_2_loss: 0.1523 - model_2_loss: 0.0068 - concatenate_2_particle_metric: 0.1237 - model_2_particle_metric: 0.0068

        
    def mask_loss(self, y_true, y_pred):
        import tensorflow as tf
        if y_pred.get_shape()[1] > self.particle_cnt_dst: 
            pred, pred_t, trunc = tf.split(y_pred, [self.particle_cnt_dst, self.particle_cnt_dst, 1], 1)
            gt, t_loss_w = tf.split(y_true, [self.particle_cnt_dst, 1], 1)
            return ((emd_loss(gt * zero_mask(gt, self.pad_val), pred) if self.mask else emd_loss(gt, pred)) + t_loss_w[:,0,0] * emd_loss(pred, pred_t))# / trunc[:,0,0]
        else:
            return emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred * zero_mask(y_true, self.pad_val)) if self.mask else emd_loss(y_true, y_pred)

    def particle_metric(self, y_true, y_pred):
        if y_pred.get_shape()[1] > self.particle_cnt_dst:    
            pred = y_pred[:,:self.particle_cnt_dst]
            gt = y_true[:,:self.particle_cnt_dst]
            return emd_loss(gt * zero_mask(gt, self.pad_val), pred) if self.mask else emd_loss(gt, pred)
        elif y_pred.get_shape()[1] < self.particle_cnt_dst:
            return keras.losses.mse(y_true, y_pred)
        else:
            return emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred * zero_mask(y_true, self.pad_val)) if self.mask else emd_loss(y_true, y_pred)

    def compile_model(self):
        if self.truncate:
            self.short_model.compile(loss=self.mask_loss, optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay))
            self.train_model.compile(loss=[self.mask_loss, 'mse'], optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay), metrics=[self.particle_metric], loss_weights=[1.0,1.0])
        else:
            self.train_model.compile(loss=self.mask_loss, optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay))
#16287/16287 [==============================] - 904s 55ms/step - loss: 0.1790 - concatenate_2_loss: 0.1658 - model_2_loss: 0.0082 - concatenate_2_particle_metric: 0.1317 - model_2_particle_metric: 0.0082

    def _train(self, epochs, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        if "generator" in kwargs:
            return self.train_model.fit_generator(generator=kwargs['generator'], validation_data=kwargs.get('val_generator'), use_multiprocessing=False, workers=1, verbose=1, callbacks=callbacks, epochs=epochs, shuffle=False)
        else:
            src_data = kwargs.get("src")
            ref_data = kwargs.get("ref")

            val_split = kwargs.get("val_split", 0.1)
            batch_size = kwargs.get("batch_size", 32)

            trunc_ref = np.count_nonzero(ref_data[...,:1] != self.pad_val, axis=1)/self.particle_cnt_dst 
            trunc_ref_exp = np.repeat(trunc_ref, 3, axis=-1)
            trunc_ref_exp = np.expand_dims(trunc_ref_exp, axis=1)

            #src_data.append(np.count_nonzero(src_data[0][:,:,:1] != self.pad_val, axis=1)/self.particle_cnt_dst)

            if self.truncate and False:
                self.short_model.fit(x=src_data,y=ref_data, validation_split=val_split, 
                                    epochs=epochs, batch_size=batch_size, verbose=1)
            
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