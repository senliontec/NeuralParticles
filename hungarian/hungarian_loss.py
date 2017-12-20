import tensorflow as tf

import keras
import keras.backend as K
from keras import losses

import os

class HungarianLoss:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.hungarian_module = tf.load_op_library(os.path.dirname(os.path.abspath(__file__)) + '/hungarian.so')

    def hungarian_loss(self, y_true, y_pred):
        sh = [s for s in y_pred.get_shape()]
        sh[0] = self.batch_size

        y_pred.set_shape(sh)
        y_true.set_shape(sh)

        m_pred = K.expand_dims(y_pred,axis=2)
        m_true = K.expand_dims(y_true,axis=2)

        cnt = y_pred.get_shape()[1]

        m_true = K.permute_dimensions(m_true,(0,2,1,3))

        cost = K.sum(K.square(m_pred - m_true),axis=-1)

        idx = tf.unstack(self.hungarian_module.hungarian(cost))
        y_true = tf.unstack(y_true)
        res = []
        for i in range(len(idx)):
            res.append(K.gather(y_true[i],idx[i]))

        y_true = tf.stack(res)

        return losses.mean_squared_error(y_true, y_pred)