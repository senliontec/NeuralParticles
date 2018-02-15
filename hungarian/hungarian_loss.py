import tensorflow as tf

import keras
import keras.backend as K
from keras import losses

import os

hungarian_module = tf.load_op_library(os.path.dirname(os.path.abspath(__file__)) + '/hungarian.so')
def hungarian_loss(y_true, y_pred):
    m_pred = K.expand_dims(y_pred,axis=2)
    m_true = K.expand_dims(y_true,axis=2)

    m_true = K.permute_dimensions(m_true,(0,2,1,3))

    cost = K.sum(K.square(m_pred - m_true),axis=-1)

    idx = hungarian_module.hungarian(cost)

    bs = tf.shape(y_pred)[0]
    l = tf.shape(y_pred)[1]

    batch_idx = tf.range(0, bs)
    batch_idx = tf.reshape(batch_idx, (bs, 1))
    b = tf.tile(batch_idx, (1, l))

    idx = tf.stack([b,idx],2)

    y_true = tf.gather_nd(y_true,idx)

    return losses.mean_squared_error(y_true, y_pred)