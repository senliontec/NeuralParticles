import keras
from keras.layers.core import Layer
from keras.models import Model
from keras.layers import Input, Lambda
import numpy as np
from keras import backend as K

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

class QuaternionMul(Layer):
    def __init__(self, **kwargs): 
        super(QuaternionMul, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, X, mask=None):
        return quaternion_rot(X[0],X[1])

class MatrixMul(Layer):
    def __init__(self, **kwargs): 
        super(QuaternionMul, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, X, mask=None):
        return K.batch_dot(X[0],X[1])

if __name__ == "__main__":
    pos = np.array([[[1,0,0],[0,1,0],[0,0,1]],[[-1,0,0],[0,-1,0],[0,-1,0]]])
    quat = np.array([[0,1,0,0],[0,0,1,0]])

    p = K.variable(pos)
    q = K.variable(quat)
    print(pos)
    print(quat)
    print(K.eval(quaternion_rot(p,q)))

    inputs = Input((3,3))
    quat_inputs = Input((4,))
    x = QuaternionMul()([inputs,quat_inputs])
    m = Model(inputs=[inputs,quat_inputs], outputs=x)
    
    qi = Lambda(quaternion_conj)(quat_inputs)
    x = QuaternionMul()([x,qi])
    m2 = Model(inputs=[inputs,quat_inputs], outputs=x)
    m.summary()
    print(m.predict(x=[pos,quat]))
    print(m2.predict(x=[pos,quat]))
    print(Model(inputs=[inputs,quat_inputs], outputs=qi).predict(x=[pos,quat]))