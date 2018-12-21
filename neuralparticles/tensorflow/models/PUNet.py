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
        
        self.particle_cnt_src = kwargs.get("par_cnt")
        self.particle_cnt_dst = kwargs.get("par_cnt_ref")

        self.features = kwargs.get("features")
        if kwargs.get("gen_vel"):
            self.features.append('v')
        self.dim = kwargs.get("dim", 2)
        self.factor = kwargs.get("factor")
        self.factor_d = math.pow(self.factor, 1/self.dim)

        self.patch_size = kwargs.get("patch_size") * kwargs.get("res") / self.factor_d

        self.fps = kwargs.get("fps")

        tmp_w = kwargs.get("loss_weights")

        self.loss_weights = [tmp_w[0]]
        self.temp_coh = tmp_w[1] > 0.0
        self.truncate = tmp_w[2] > 0.0

        self.acc_fac = kwargs.get("acc_fac")

        if self.temp_coh:
            self.loss_weights.append(tmp_w[1])
        
        if self.truncate:
            self.loss_weights.append(tmp_w[2])

        self.res = kwargs.get("res")
        self.lres = int(self.res/self.factor_d)

        self.norm_factor = kwargs.get("norm_factor")
        if kwargs.get("gen_vel"):
            self.norm_factor = np.append(self.norm_factor, [0.1,0.1,0.1])

        self.use_temp_emd = kwargs.get("use_temp_emd")
        self.residual = kwargs.get("residual")

    def _init_optimizer(self, epochs=1):
        self.optimizer = keras.optimizers.adam(lr=self.learning_rate, decay=self.decay)

    def _build_model(self):            
        activation = keras.activations.tanh#lambda x: keras.activations.relu(x, alpha=0.1)
        inputs = Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0) + (2 if 'n' in self.features else 0)), name="main_input")
        input_xyz = extract_xyz(inputs, name="extract_pos")
        input_points = input_xyz

        if len(self.features) > 0:
            input_points = extract_aux(inputs, name="extract_aux")
            input_points = MultConst(1./self.norm_factor, name="normalization")(input_points)
            input_points = concatenate([input_xyz, input_points], axis=-1, name='input_concatenation')
        
        if not self.mask:
            mask = zero_mask(input_xyz, self.pad_val, name="mask_1")
            input_points = multiply([input_points, mask])
            input_xyz = multiply([input_xyz, mask])
            input_xyz_m = input_xyz
            input_points_m = input_points


        l1_xyz, l1_points = pointnet_sa_module(input_xyz, input_points, self.particle_cnt_src, 0.25, self.fac*4, 
                                               [self.fac*4,
                                                self.fac*4,
                                                self.fac*8], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        l2_xyz, l2_points = pointnet_sa_module(l1_xyz, l1_points, self.particle_cnt_src//2, 0.5, self.fac*4, 
                                               [self.fac*8,
                                                self.fac*8,
                                                self.fac*16], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        l3_xyz, l3_points = pointnet_sa_module(l2_xyz, l2_points, self.particle_cnt_src//4, 0.6, self.fac*4, 
                                               [self.fac*16,
                                                self.fac*16,
                                                self.fac*32], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        l4_xyz, l4_points = pointnet_sa_module(l3_xyz, l3_points, self.particle_cnt_src//8, 0.7, self.fac*4, 
                                               [self.fac*32,
                                                self.fac*32,
                                                self.fac*64], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))


        if self.mask:
            mask = zero_mask(input_xyz, self.pad_val, name="mask_2")
            input_xyz_m = multiply([input_xyz, mask])
            input_points_m = multiply([input_points, mask])
            #x = multiply([x, mask])

        # interpoliere die features in l2_points auf die Punkte in x
        up_l2_points = pointnet_fp_module(input_xyz_m, l2_xyz, None, l2_points, [self.fac*8], kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)
        up_l3_points = pointnet_fp_module(input_xyz_m, l3_xyz, None, l3_points, [self.fac*8], kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)
        up_l4_points = pointnet_fp_module(input_xyz_m, l4_xyz, None, l4_points, [self.fac*8], kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)

        x = concatenate([up_l4_points, up_l3_points, up_l2_points, l1_points, input_points_m], axis=-1)

        #if self.mask:
        #    mask = zero_mask(input_xyz, self.pad_val, name="mask_2")
            #x = multiply([x, mask])

        x_t = x
        l = []
        for i in range(self.particle_cnt_dst//self.particle_cnt_src):
            tmp = Conv1D(self.fac*32, 1, name="expansion_1_"+str(i+1), kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)(x)
            tmp = Conv1D(self.fac*16, 1, name="expansion_2_"+str(i+1), kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)(tmp)
            l.append(tmp)
        x = concatenate(l, axis=1, name="pixel_conv") if self.particle_cnt_dst//self.particle_cnt_src > 1 else l[0]

        x = Conv1D(self.fac*8, 1, name="coord_reconstruction_1", kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)(x)

        b = np.zeros((3,), dtype='float32')
        W = np.zeros((1, self.fac*8, 3), dtype='float32')
        x = Conv1D(3, 1, name="coord_reconstruction_2")(x)#, weights=[W,b])(x)
        
        out = x
        
        if self.truncate:
            out = Reshape((self.particle_cnt_dst//self.particle_cnt_src, self.particle_cnt_src, 3))(out)
            out = Permute((2,1,3))(out)
            out = Reshape((self.particle_cnt_dst, 3))(out)

        if self.residual:
            x = Flatten()(input_xyz_m)
            x = RepeatVector(self.particle_cnt_dst//self.particle_cnt_src)(x)
            x = Reshape((self.particle_cnt_dst, 3))(x)

            out = add([x, out])

        if self.truncate:
            x_t = Conv1D(self.fac*4, 1)(input_points)
            x_t = Conv1D(self.fac*4, 1)(x_t)

            '''out_mask = []
            for i in range(self.particle_cnt_dst//self.particle_cnt_src): 
                tmp = unstack(l[i], 1, name='unstack_%d'%(i+1))
                tmp = add(tmp, name='merge_features_%d'%(i+1))

                tmp = Dense(self.fac, activation='elu', kernel_regularizer=keras.regularizers.l2(0.02), name="truncation_%d"%(i+1))(tmp)
                #x_t = Dropout(self.dropout)(x_t)
                b = np.zeros(1, dtype='float32')
                W = np.zeros((self.fac, 1), dtype='float32')
                tmp = Dense(1, activation='elu', kernel_regularizer=keras.regularizers.l2(0.02), weights=[W,b], name="cnt_%d"%(i+1))(tmp)
                l[i] = AddConst((self.particle_cnt_src+1)/self.particle_cnt_src)(tmp)
                out_mask.append(soft_trunc_mask(l[i], self.particle_cnt_src, name="truncation_mask_%d"%(i+1)))
            trunc = concatenate(l, 1, name="cnt")
            out_mask = concatenate(out_mask, axis=1, name="truncation_mask")'''

            #if self.mask:
            #    x_t = multiply([x_t, mask])

            x_t = unstack(x_t, 1, name='unstack')
            x_t = add(x_t, name='merge_features')


            """#x_t = Dropout(self.dropout)(x_t)
            x_t = Dense(self.fac, activation='elu', kernel_regularizer=keras.regularizers.l2(0.02))(x_t)
            #x_t = Dropout(self.dropout)(x_t)
            b = np.zeros(1, dtype='float32')
            W = np.zeros((self.fac, 1), dtype='float32')
            trunc = Dense(1, activation='elu', kernel_regularizer=keras.regularizers.l2(0.02), weights=[W,b], name="cnt")(x_t)
            trunc = AddConst((self.particle_cnt_dst+1)/self.particle_cnt_dst, name="truncation")(trunc)"""


            x_t = Dense(self.fac, activation='tanh')(x_t)
            #x_t = Dropout(self.dropout)(x_t)
            b = np.zeros(1, dtype='float32')
            W = np.zeros((self.fac, 1), dtype='float32')
            x_t = Dense(1, activation='tanh', weights=[W,b], name="cnt")(x_t)
            #x_t = AddConst((self.particle_cnt_dst+1)/self.particle_cnt_dst, name="truncation")(trunc)

            trunc = MultConst(1/self.particle_cnt_src)(add(unstack(mask,1)))

            trunc = add([trunc, x_t], name="truncation")

            out_mask = soft_trunc_mask(trunc, self.particle_cnt_dst, name="truncation_mask")

            self.trunc_model = Model(inputs=inputs, outputs=trunc)
            out = multiply([out, Reshape((self.particle_cnt_dst,1))(out_mask)], name="masked_coords")
            self.model = Model(inputs=inputs, outputs=[out,trunc])
        else:
            if self.mask:# or True:
                out_mask = RepeatVector(self.particle_cnt_dst//self.particle_cnt_src)(Flatten()(mask))
                out = multiply([out, Reshape((self.particle_cnt_dst,1))(out_mask)])
            self.model = Model(inputs=inputs, outputs=[out])

        if self.temp_coh:
            inputs = [
                Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0) + (2 if 'n' in self.features else 0))),
                Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0) + (2 if 'n' in self.features else 0))),
                Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0) + (2 if 'n' in self.features else 0)))]
            out = self.model(inputs[0])

            if self.truncate:
                trunc = Lambda(lambda x: x, name="trunc")(out[1])
                out = Lambda(lambda x: x, name="points")(out[0])
                out1 = concatenate([out, self.model(inputs[1])[0], self.model(inputs[2])[0]], axis=1, name='temp')
                self.train_model = Model(inputs=inputs, outputs=[out, out1, trunc])
            else:
                out = Lambda(lambda x: x, name="points")(out)
                out1 = concatenate([out, self.model(inputs[1]), self.model(inputs[2])], axis=1, name='temp')
                self.train_model = Model(inputs=inputs, outputs=[out, out1])
        else:
            self.train_model = self.model
        
    def mask_loss(self, y_true, y_pred):
        loss = get_repulsion_loss4(y_pred)
        return loss + emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred)

    def trunc_loss(self, y_true, y_pred):
        return keras.losses.mse(1, y_pred/y_true)

    def temp_loss(self, y_true, y_pred):
        import tensorflow as tf
        pred, pred_p, pred_n = tf.split(y_pred, [self.particle_cnt_dst, self.particle_cnt_dst, self.particle_cnt_dst], 1)
        gt, gt_p, gt_n = tf.split(y_true, [self.particle_cnt_dst, self.particle_cnt_dst, self.particle_cnt_dst], 1)
        if self.use_temp_emd: 
            return keras.losses.mse(emd_loss(pred, pred_t), emd_loss(gt * zero_mask(gt, self.pad_val), gt_t * zero_mask(gt_t, self.pad_val)))
        else:
            #return keras.losses.mse(pred, pred_n)
            #return keras.losses.mse(pred-pred_p, pred_n-pred)
            pred_v0 = pred - pred_p
            pred_v1 = pred_n - pred
            gt_v0 = gt - gt_p
            gt_v1 = gt_n - gt
            pred_a = pred_v1 - pred_v0
            gt_a = gt_v1 - gt_v0

            match = approx_match(pred, gt)
            return (1 - self.acc_fac) * (match_cost(pred_v0, gt_v0, match) + match_cost(pred_v1, gt_v1, match)) / 2 + self.acc_fac * match_cost(pred_a, gt_a, match)

            #return keras.losses.mse(emd_loss(pred, gt_t * zero_mask(gt_t, self.pad_val)), -emd_loss(pred_t, gt * zero_mask(gt, self.pad_val)))
            #return keras.backend.square(emd_loss(pred_t, gt * zero_mask(gt, self.pad_val)) + emd_loss(pred, gt_t * zero_mask(gt_t, self.pad_val))) + 0.1 * keras.losses.mse(pred, pred_t)#, emd_loss(gt * zero_mask(gt, self.pad_val), gt_t * zero_mask(gt_t, self.pad_val))) 
            #return keras.losses.mse(pred, pred_p) + keras.losses.mse(pred, pred_n)
 
    def trunc_metric(self, y_true, y_pred):
        if y_pred.get_shape()[1] == self.particle_cnt_dst:
            return keras.losses.mse(keras.backend.sum(zero_mask(y_true, self.pad_val),axis=-2)/self.particle_cnt_dst,keras.backend.sum(zero_mask(y_pred, 0.0),axis=-2)/self.particle_cnt_dst)
        return keras.backend.constant(0)

    def particle_metric(self, y_true, y_pred):
        if y_pred.get_shape()[1] == self.particle_cnt_dst:    
            return emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred)
        elif y_pred.get_shape()[1] == self.particle_cnt_dst*2:
            return self.temp_loss(y_true, y_pred)
        else:           
            return keras.losses.mse(y_true, y_pred)

    def repulsion_metric(self, y_true, y_pred):
        if y_pred.get_shape()[1] > self.particle_cnt_dst:    
            pred = y_pred[:,:self.particle_cnt_dst]
            return repulsion_loss(pred)
        elif y_pred.get_shape()[1] < self.particle_cnt_dst:
            return keras.losses.mse(y_true, y_pred)
        else:
            return repulsion_loss(y_pred)

    def compile_model(self):
        loss = [self.mask_loss]
        if self.temp_coh:
            loss.append(self.temp_loss)

        if self.truncate:
            loss.append(self.trunc_loss)
            self.trunc_model.compile(loss=self.trunc_loss, optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay))
            self.train_model.compile(loss=loss, optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay), loss_weights=self.loss_weights, metrics=[self.particle_metric])
        else:
            self.train_model.compile(loss=loss, optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay), loss_weights=self.loss_weights, metrics=[self.particle_metric, self.trunc_metric])

    def _train(self, epochs, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        if "generator" in kwargs:
            '''self.train_model.trainable = False
            self.trunc_model.trainable = True
            tmp_w = self.loss_weights
            self.loss_weights[0] = 0.0
            self.loss_weights[1] = 0.0
            self.compile_model()
            self.trunc_model.summary()
            self.train_model.summary()
            self.train_model.fit_generator(generator=kwargs['generator'], validation_data=kwargs.get('val_generator'), use_multiprocessing=False, workers=1, verbose=1, callbacks=callbacks, epochs=epochs, shuffle=False)

            self.trunc_model.trainable = False
            self.train_model.trainable = True
            self.loss_weights[0] = tmp_w[0]
            self.loss_weights[1] = tmp_w[1]
            self.loss_weights[2] = 0.0
            self.compile_model()'''
            return self.train_model.fit_generator(generator=kwargs['generator'], validation_data=kwargs.get('val_generator'), use_multiprocessing=False, workers=1, verbose=1, callbacks=callbacks, epochs=epochs, shuffle=False)
        else:
            src_data = kwargs.get("src")
            ref_data = [kwargs.get("ref")]
            
            val_split = kwargs.get("val_split", 0.1)
            batch_size = kwargs.get("batch_size", 32)

            if self.temp_coh:
                if kwargs.get("gen_vel"):
                    vel = random_deformation(src_data)
                    src_data = np.concatenate((src_data, vel), axis=-1)
                else:
                    vel = src_data[...,3:6]

                idx = np.argmin(np.linalg.norm(src_data[...,:3], axis=-1), axis=1)
                vel = vel - np.expand_dims(vel[np.arange(len(idx)), idx],axis=1)

                adv_src = src_data[...,:3] + vel / (self.patch_size * self.fps)
                src_data = [src_data, np.concatenate((adv_src, src_data[...,3:]), axis=-1)]

                ref_data.append(np.concatenate((ref_data[0], ref_data[0]), axis=1))

            if self.truncate:
                trunc_ref = np.count_nonzero(ref_data[0][...,:1] != self.pad_val, axis=1)/self.particle_cnt_dst 
                ref_data.append(trunc_ref)
            
            return self.train_model.fit(x=src_data,y=ref_data, validation_split=val_split, 
                                epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)

    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        if self.model:
            self.model.load_weights(path)
            self.train_model.load_weights(path)
        else:
            self.model = load_model(path, custom_objects={'mask_loss': self.mask_loss, 'trunc_loss': self.trunc_loss, 'temp_loss': self.temp_loss, 
                                                          'particle_metric': self.particle_metric, 'trunc_metric': self.trunc_metric,
                                                          'Interpolate': Interpolate, 'SampleAndGroup': SampleAndGroup, 'MultConst': MultConst, 'AddConst': AddConst})
        '''if self.truncate:
            out, trunc = self.model.outputs
            trunc_exp = stack([trunc, trunc, trunc], 2)
            out = concatenate([out, trunc_exp], 1, name='points')
            self.train_model = Model(inputs=self.model.inputs, outputs=[out, trunc])
        else:
            self.train_model = Model(self.model.inputs, self.model.outputs)'''