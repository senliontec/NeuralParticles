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

    return K.mean(K.sum(K.square(y_true - y_pred), axis=-1), axis=-1)

if __name__ == '__main__':
    import numpy as np

    x = np.random.rand(1,50,2)
    y = np.random.rand(1,50,2)

    m_x = np.expand_dims(x,axis=2)
    m_y = np.expand_dims(y,axis=2)

    m_y = np.transpose(m_y,(0,2,1,3))

    cost = np.sum(np.square(m_x - m_y),axis=-1)

    idx = K.eval(hungarian_module.hungarian(cost))[0]

    import matplotlib.pyplot as plt

    summe = 0
    for i in range(len(idx)):
        plt.plot([x[0,i,0],y[0,idx[i],0]],[x[0,i,1],y[0,idx[i],1]], 'k-')
        summe += np.sum(np.square(x[0,i] - y[0,idx[i]]))
    print(summe)

    plt.scatter(x[0,:,0],x[0,:,1],c='r')
    plt.scatter(y[0,:,0],y[0,:,1],c='g')

    #plt.show()
    plt.savefig("hungarian.pdf")