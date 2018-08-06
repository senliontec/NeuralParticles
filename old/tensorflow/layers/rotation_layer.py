import keras
from keras.layers.core import Layer
from keras.models import Model
from keras.layers import Input, Lambda
import numpy as np
from keras import backend as K

from neuralparticles.tensorflow.tools.quaternion import *

class QuaternionMul(Layer):
    def __init__(self, **kwargs): 
        super(QuaternionMul, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, X, mask=None):
        return quaternion_rot(X[0],X[1])

class MatrixMul(Layer):
    def __init__(self, **kwargs): 
        super(MatrixMul, self).__init__(**kwargs)

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