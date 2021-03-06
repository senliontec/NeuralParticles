import keras
import numpy as np
from keras import backend as K
import tensorflow as tf

def quaternion_conj(quat):
    return quat * np.array([1,-1,-1,-1])/K.expand_dims(K.sum(quat*quat, axis=1), axis=1)

def quaternion_norm(quat):
    import tensorflow as tf
    return quat/K.expand_dims(tf.norm(quat,axis=1),axis=1)
    
def quaternion_rot(vec, quat):
    import tensorflow as tf
    def quat_mul(q0,q1):
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1

        return [-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]

    def quat_vec_mul(q,v):
        w0, x0, y0, z0 = q
        x1, y1, z1 = v

        return [-x1 * x0 - y1 * y0 - z1 * z0,
                x1 * w0 + y1 * z0 - z1 * y0,
                -x1 * z0 + y1 * w0 + z1 * x0,
                x1 * y0 - y1 * x0 + z1 * w0]

    dim_diff = K.ndim(vec) - K.ndim(quat)
    v = tf.unstack(vec,axis=-1)
    q = tf.unstack(quat,axis=-1)

    q_inv = quaternion_conj(quat)
    q_inv = tf.unstack(q_inv,axis=-1)

    for i in range(dim_diff):
        q = list(map(lambda x: K.expand_dims(x,axis=-1), q))
        q_inv = list(map(lambda x: K.expand_dims(x,axis=-1), q_inv))

    v = quat_vec_mul(q, v)
    v = quat_mul(v, q_inv)
    return tf.stack([v[1],v[2],v[3]],axis=-1)