from .architecture import Network

import numpy as np
import math

import keras
from keras.layers import Input, multiply, concatenate, Conv1D, Lambda, add, Dropout, Dense, Reshape, RepeatVector, Flatten, Permute
from keras.models import Model, load_model
import keras.backend as K

from neuralparticles.tensorflow.tools.pointnet_util import pointnet_sa_module, pointnet_fp_module, Interpolate, SampleAndGroup
from neuralparticles.tensorflow.tools.zero_mask import zero_mask, soft_trunc_mask
from neuralparticles.tools.data_augmentation import *

from neuralparticles.tools.data_helpers import get_data

from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss, approx_match, match_cost
from neuralparticles.tensorflow.losses.tf_nndistance import nn_index, batch_gather
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
        self.mingle = kwargs.get("mingle")
        self.repulsion = kwargs.get("repulsion")

        self.neg_examples = kwargs.get("neg_examples")

        if self.temp_coh:
            self.loss_weights.append(tmp_w[1])
        
        self.pretrain = kwargs.get('pretrain')
        if self.truncate and not self.pretrain:
            self.loss_weights.append(tmp_w[2])

        self.res = kwargs.get("res")
        self.lres = int(self.res/self.factor_d)

        self.norm_factor = kwargs.get("norm_factor")
        if kwargs.get("gen_vel"):
            self.norm_factor = np.append(self.norm_factor, [0.1,0.1,0.1])

        self.use_temp_emd = kwargs.get("use_temp_emd")
        self.residual = kwargs.get("residual")
        self.permutate = kwargs.get("permutate")

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
                                               [self.fac*4, self.fac*4, self.fac*8], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        l2_xyz, l2_points = pointnet_sa_module(l1_xyz, l1_points, self.particle_cnt_src//2, 0.5, self.fac*4, 
                                               [self.fac*8, self.fac*8, self.fac*16], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        l3_xyz, l3_points = pointnet_sa_module(l2_xyz, l2_points, self.particle_cnt_src//4, 0.6, self.fac*4, 
                                               [self.fac*16, self.fac*16, self.fac*32], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        l4_xyz, l4_points = pointnet_sa_module(l3_xyz, l3_points, self.particle_cnt_src//8, 0.7, self.fac*4, 
                                               [self.fac*32, self.fac*32, self.fac*64], activation=activation, kernel_regularizer=keras.regularizers.l2(self.l2_reg))

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
            tmp = Dropout(self.dropout)(x)
            tmp = Conv1D(self.fac*32, 1, name="expansion_1_"+str(i+1), kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)(tmp)
            tmp = Dropout(self.dropout)(tmp)
            tmp = Conv1D(self.fac*16, 1, name="expansion_2_"+str(i+1), kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)(tmp)
            l.append(tmp)
        x = concatenate(l, axis=1, name="pixel_conv") if self.particle_cnt_dst//self.particle_cnt_src > 1 else l[0]
            
        x = Dropout(self.dropout)(x)
        x = Conv1D(self.fac*8, 1, name="coord_reconstruction_1", kernel_regularizer=keras.regularizers.l2(self.l2_reg), activation=activation)(x)

        b = np.zeros((3,), dtype='float32')
        W = np.zeros((1, self.fac*8, 3), dtype='float32')
        x = Conv1D(3, 1, name="coord_reconstruction_2")(x)#, weights=[W,b])(x)
        
        out = x

        if self.residual:
            x = Flatten()(input_xyz_m)
            x = RepeatVector(self.particle_cnt_dst//self.particle_cnt_src)(x)
            x = Reshape((self.particle_cnt_dst, 3))(x)

            out = add([x, out])

        if self.permutate:
            out = Reshape((self.particle_cnt_dst//self.particle_cnt_src*1, self.particle_cnt_src//1, 3))(out)
            out = Permute((2,1,3))(out)
            out = Reshape((self.particle_cnt_dst, 3))(out)

        if self.truncate:
            trunc_input = Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0) + (2 if 'n' in self.features else 0)), name="trunc_input")
            
            input_xyz_t = extract_xyz(trunc_input, name="extract_pos_t")
            input_points_t = input_xyz_t

            if len(self.features) > 0:
                input_points_t = extract_aux(trunc_input, name="extract_aux_t")
                input_points_t = MultConst(1./self.norm_factor, name="normalization_t")(input_points_t)
                input_points_t = concatenate([input_xyz_t, input_points_t], axis=-1, name='input_concatenation_t')
            
            mask_t = zero_mask(input_xyz_t, self.pad_val, name="mask_1_t")
            if not self.mask:
                input_points_t = multiply([input_points_t, mask_t])
                input_xyz_t = multiply([input_xyz_t, mask_t])
                
            x_t = Conv1D(self.fac*8, 1)(input_points_t)
            x_t = Conv1D(self.fac*8, 1)(x_t)
            
            x_t = unstack(x_t, 1, name='unstack')
            x_t = add(x_t, name='merge_features')
            
            """
            x_t = Dense(self.fac*4, activation='elu', kernel_regularizer=keras.regularizers.l2(0.02))(x_t)
            b = np.zeros(1, dtype='float32')
            W = np.zeros((self.fac*4, 1), dtype='float32')

            x_t = Dense(1, activation='elu', kernel_regularizer=keras.regularizers.l2(0.02), weights=[W,b], name="cnt")(x_t)
            trunc = AddConst((self.particle_cnt_dst+1)/self.particle_cnt_dst, name="truncation")(x_t)

            """
            x_t = Dense(self.fac*4, activation='tanh')(x_t)
            b = np.zeros(1, dtype='float32')
            W = np.zeros((self.fac*4, 1), dtype='float32')
            
            x_t = Dense(1, weights=[W,b], activation='tanh', name="cnt")(x_t)

            trunc = MultConst(1/self.particle_cnt_src)(add(unstack(mask_t, 1)))
            trunc = add([trunc, x_t], name="truncation")
            
            self.trunc_model = Model(inputs=trunc_input, outputs=trunc, name="truncation")

            trunc = self.trunc_model(inputs)


            out_mask = soft_trunc_mask(trunc, self.particle_cnt_dst)

            out_mask = Reshape((self.particle_cnt_dst, 1), name="truncation_mask")(out_mask)

            out = multiply([out, out_mask], name="masked_coords")
            self.model = Model(inputs=inputs, outputs=[out, trunc])
        else:
            if self.mask:# or True:
                out_mask = RepeatVector(self.particle_cnt_dst//self.particle_cnt_src)(Flatten()(mask))

                out_mask = Reshape((self.particle_cnt_dst//self.particle_cnt_src, self.particle_cnt_src))(out_mask)
                out_mask = Permute((2,1))(out_mask)
                out_mask = Reshape((self.particle_cnt_dst, 1), name="truncation_mask")(out_mask)

                out = multiply([out, out_mask])
            self.model = Model(inputs=inputs, outputs=[out])


        # gen train model
        inputs = [Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0) + (2 if 'n' in self.features else 0)))]
        
        out0 = self.model(inputs[0])

        if self.truncate:
            trunc = Lambda(lambda x: x, name="trunc")(out0[1])
            out0 = Lambda(lambda x: x)(out0[0])
            out_mask = soft_trunc_mask(trunc, self.particle_cnt_dst)
            out_mask = Reshape((self.particle_cnt_dst, 1), name="truncation_mask")(out_mask)
        else:
            if self.mask:
                out_mask = RepeatVector(self.particle_cnt_dst//self.particle_cnt_src)(Flatten()(zero_mask(inputs[0], self.pad_val)))

                out_mask = Reshape((self.particle_cnt_dst//self.particle_cnt_src, self.particle_cnt_src))(out_mask)
                out_mask = Permute((2,1))(out_mask)
                out_mask = Reshape((self.particle_cnt_dst, 1))(out_mask)                
            else:
                out_mask = Lambda(lambda x: K.ones_like(x)[...,:1])(out0)

        out_m = concatenate([out0, out_mask], axis=-1, name="out_m")

        outputs = [out_m]
        if self.temp_coh:
            inputs.extend([
                Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0) + (2 if 'n' in self.features else 0))),
                Input((self.particle_cnt_src, 3 + len(self.features) + (2 if 'v' in self.features else 0) + (2 if 'n' in self.features else 0)))])
            if self.truncate:
                #out1 = self.model(inputs[1])
                #out2 = self.model(inputs[2])
                #out_mask = multiply([Flatten()(out_mask), soft_trunc_mask(out1[1], self.particle_cnt_dst), soft_trunc_mask(out2[1], self.particle_cnt_dst)])
                #out_mask = Reshape((self.particle_cnt_dst, 1))(out_mask)
                #out1 = concatenate([out0, out1, out2], axis=1, name='temp')
                out1 = concatenate([out0, self.model(inputs[1])[0], self.model(inputs[2])[0]], axis=1, name='temp')
            else:
                out1 = concatenate([out0, self.model(inputs[1]), self.model(inputs[2])], axis=1, name='temp')
            outputs.append(out1)
        
        if self.truncate and not self.pretrain:
            outputs.append(trunc)
            
        self.train_model = Model(inputs=inputs, outputs=outputs)


    def mask_loss(self, y_true, y_pred):
        loss = 0
        mask = y_pred[...,3:]
        y_pred = y_pred[...,:3]
        if self.mingle > 0:
            loss += self.mingle_loss(mask, y_pred) * self.mingle
        if self.repulsion > 0:
            loss += get_repulsion_loss4(y_pred) * self.repulsion
        return loss + emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred)


    def trunc_loss(self, y_true, y_pred):
        return keras.losses.mse(1, y_pred/y_true)

    def temp_loss_raw(self, y_true, y_pred, use_emd, acc_fac, old=True):
        import tensorflow as tf
        pred, pred_p, pred_n = tf.split(y_pred, [self.particle_cnt_dst, self.particle_cnt_dst, self.particle_cnt_dst], 1)
        gt, gt_p, gt_n = tf.split(y_true, [self.particle_cnt_dst, self.particle_cnt_dst, self.particle_cnt_dst], 1)
        if use_emd: 
            pred_v0 = pred - pred_p
            pred_v1 = pred_n - pred
            pred_a = pred_v1 - pred_v0

            if not old:
                match = nn_index(gt, pred)
                gt = batch_gather(gt, match)
                gt_p = batch_gather(gt_p, match)
                gt_n = batch_gather(gt_n, match)

            gt_v0 = gt - gt_p
            gt_v1 = gt_n - gt
            gt_a = gt_v1 - gt_v0

            if not old:
                return (1 - acc_fac) * (K.mean(K.sqrt(K.sum(K.square(pred_v0 - gt_v0), axis=-1)), axis=-1) + K.mean(K.sqrt(K.sum(K.square(pred_v1 - gt_v1), axis=-1)), axis=-1)) / 2 + acc_fac * K.mean(K.sqrt(K.sum(K.square(pred_a - gt_a), axis=-1)), axis=-1)
            else:
                match = approx_match(pred, gt*zero_mask(gt, self.pad_val))
                return ((1 - acc_fac) * (match_cost(pred_v0, gt_v0, match) + match_cost(pred_v1, gt_v1, match)) / 2 + acc_fac * match_cost(pred_a, gt_a, match)) / tf.cast(tf.shape(gt)[1], tf.float32)
        else:
            return (1 - acc_fac) * keras.losses.mse(pred, pred_n) + acc_fac * keras.losses.mse(pred-pred_p, pred_n-pred)

    def temp_loss(self, y_true, y_pred):
        return self.temp_loss_raw(y_true, y_pred, self.use_temp_emd, self.acc_fac)

    def mingle_loss_2(self, mask, y_pred):
        if self.permutate:
            A = K.reshape(y_pred, (-1, self.particle_cnt_src, self.particle_cnt_dst//self.particle_cnt_src, 1, 3))
            B = K.reshape(y_pred, (-1, self.particle_cnt_src, 1, self.particle_cnt_dst//self.particle_cnt_src, 3))
            A = K.repeat_elements(A, self.particle_cnt_dst//self.particle_cnt_src, 3)
            B = K.repeat_elements(B, self.particle_cnt_dst//self.particle_cnt_src, 2)
            
            m0 = K.reshape(mask, (-1, self.particle_cnt_src, self.particle_cnt_dst//self.particle_cnt_src, 1))
            m1 = K.reshape(mask, (-1, self.particle_cnt_src, 1, self.particle_cnt_dst//self.particle_cnt_src))
            m0 = K.repeat_elements(m0, self.particle_cnt_dst//self.particle_cnt_src, 3)
            m1 = K.repeat_elements(m1, self.particle_cnt_dst//self.particle_cnt_src, 2)
            mask = m0 * m1

            cost = K.sum((K.sum(K.square(A-B),axis=-1)*mask), axis=(2,3))/(K.sum(mask, axis=(2,3))+1e-8)
        else:
            A = K.reshape(y_pred, (-1, self.particle_cnt_dst//self.particle_cnt_src, 1, self.particle_cnt_src, 3))
            B = K.reshape(y_pred, (-1, 1, self.particle_cnt_dst//self.particle_cnt_src, self.particle_cnt_src, 3))
            A = K.repeat_elements(A, self.particle_cnt_dst//self.particle_cnt_src, 2)
            B = K.repeat_elements(B, self.particle_cnt_dst//self.particle_cnt_src, 1)

            m0 = K.reshape(mask, (-1, self.particle_cnt_dst//self.particle_cnt_src, 1, self.particle_cnt_src))
            m1 = K.reshape(mask, (-1, 1, self.particle_cnt_dst//self.particle_cnt_src, self.particle_cnt_src))
            m0 = K.repeat_elements(m0, self.particle_cnt_dst//self.particle_cnt_src, 2)
            m1 = K.repeat_elements(m1, self.particle_cnt_dst//self.particle_cnt_src, 1)
            mask = m0 * m1

            cost = K.sum((K.sum(K.square(A-B),axis=-1)*mask), axis=(1,2))/(K.sum(mask, axis=(1,2))+1e-8)

        return 1./(K.mean(cost, axis=-1)+1e-8)

    def mingle_loss(self, mask, y_pred): 
        if self.permutate:
            y_pred = K.reshape(y_pred, (-1, self.particle_cnt_src, self.particle_cnt_dst//self.particle_cnt_src, 3))
            mask = K.reshape(mask, (-1, self.particle_cnt_src, self.particle_cnt_dst//self.particle_cnt_src, 1))
            cnt = K.sum(mask[:,:,1,0], axis=-1)
            group_cnt = K.sum(mask, axis=2)
            group_mean = K.sum(y_pred, axis=2)/(group_cnt+1e-8)
            group_mean = K.expand_dims(group_mean, axis=2)
            group_diff = K.sum(K.sqrt(K.sum(K.square(group_mean - y_pred) * mask, axis=-1)+1e-8), axis=2)
        else:
            y_pred = K.reshape(y_pred, (-1, self.particle_cnt_dst//self.particle_cnt_src, self.particle_cnt_src, 3))
            mask = K.reshape(mask, (-1, self.particle_cnt_dst//self.particle_cnt_src, self.particle_cnt_src, 1))
            cnt = K.sum(mask[:,1,:,0], axis=-1)
            group_cnt = K.sum(mask, axis=1)
            group_mean = K.sum(y_pred, axis=1)/(group_cnt+1e-8)
            group_mean = K.expand_dims(group_mean, axis=1)
            group_diff = K.sum(K.sqrt(K.sum(K.square(group_mean - y_pred) * mask, axis=-1)+1e-8), axis=1)
        group_cnt = K.clip(K.reshape(group_cnt, (-1, self.particle_cnt_src)) - 1, 0, self.particle_cnt_dst//self.particle_cnt_src)
        return K.sum(group_cnt/group_diff, axis=-1)/(cnt+1e-8)


    def trunc_metric(self, y_true, y_pred):
        return keras.losses.mse(K.sum(zero_mask(y_true, self.pad_val),axis=-2)/self.particle_cnt_dst,K.sum(y_pred[...,3:],axis=-2)/self.particle_cnt_dst)


    def particle_metric(self, y_true, y_pred):
        return emd_loss(y_true * zero_mask(y_true, self.pad_val), y_pred[...,:3])

    def mse_vel_metric(self, y_true, y_pred):
        return self.temp_loss_raw(y_true, y_pred, False, 0.0, True)

    def mse_acc_metric(self, y_true, y_pred):
        return self.temp_loss_raw(y_true, y_pred, False, 1.0, True)

    def emd_vel_metric(self, y_true, y_pred):
        return self.temp_loss_raw(y_true, y_pred, True, 0.0, True)

    def emd_acc_metric(self, y_true, y_pred):
        return self.temp_loss_raw(y_true, y_pred, True, 1.0, True)

    def mingle_metric(self, y_true, y_pred):
        return self.mingle_loss(y_pred[...,3:], y_pred[...,:3])


    def compile_model(self):
        loss = [self.mask_loss]
        metrics = {'out_m':[self.trunc_metric, self.particle_metric, self.mingle_metric]}
        if self.temp_coh:
            loss.append(self.temp_loss)
            metrics['temp'] = [self.mse_vel_metric, self.mse_acc_metric, self.emd_vel_metric, self.emd_acc_metric]

        if self.truncate:
            if self.pretrain:
                self.trunc_model.compile(loss=self.trunc_loss, optimizer=keras.optimizers.adam(lr=self.learning_rate*0.1, decay=self.decay*0.1))
                self.trunc_model.trainable = False
            else:
                loss.append(self.trunc_loss)
        self.train_model.compile(loss=loss, optimizer=keras.optimizers.adam(lr=self.learning_rate, decay=self.decay), loss_weights=self.loss_weights, metrics=metrics)

    def _train(self, epochs, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        if "generator" in kwargs:
            if self.truncate and self.pretrain:
                self.trunc_model.fit_generator(generator=kwargs['trunc_generator'], validation_data=kwargs.get("val_trunc_generator"), use_multiprocessing=False, workers=1, verbose=0, callbacks=kwargs.get("trunc_callbacks", []), epochs=3, shuffle=False)
            return self.train_model.fit_generator(generator=kwargs['generator'], validation_data=kwargs.get('val_generator'), use_multiprocessing=False, workers=1, verbose=0, callbacks=callbacks, epochs=epochs, shuffle=False)
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
                                epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)

    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    def eval(self, generator):
        return self.train_model.evaluate_generator(generator=generator, use_multiprocessing=False, workers=1, verbose=1)

    def save_model(self, path):
        self.model.save(path)
        

    def load_checkpoint(self, path):
        self.train_model.load_weights(path)

    def load_model(self, path):
        if self.model:
            self.model.load_weights(path)           
        else:
            self.model = load_model(path, custom_objects={'mask_loss': self.mask_loss, 'trunc_loss': self.trunc_loss, 'temp_loss': self.temp_loss, 
                                                          'particle_metric': self.particle_metric, 'trunc_metric': self.trunc_metric,
                                                          'Interpolate': Interpolate, 'SampleAndGroup': SampleAndGroup, 'MultConst': MultConst, 'AddConst': AddConst})
