import keras
from keras.layers.core import Layer
from keras.layers import Dense

class SplitDense(Layer):
    def __init__(self,
                 units, 
                 activation=None, 
                 use_bias=True, 
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', 
                 kernel_regularizer=None, 
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None, 
                 **kwargs):
                 
        self.dense = Dense(
                 units=units, 
                 activation=activation, 
                 use_bias=use_bias, 
                 kernel_initializer=kernel_initializer, 
                 bias_initializer=bias_initializer, 
                 kernel_regularizer=kernel_regularizer, 
                 bias_regularizer=bias_regularizer, 
                 activity_regularizer=activity_regularizer, 
                 kernel_constraint=kernel_constraint, 
                 bias_constraint=bias_constraint)
        super(SplitDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense.build(input_shape[0])
        self.trainable_weights = self.dense.trainable_weights

    def compute_output_shape(self, input_shape):
        return [self.dense.output_shape] * len(input_shape)

    def call(self, X, mask=None):
        Y = []
        for x in X:
            Y.append(self.dense(x))
        return Y

    def get_config(self):
        config = self.dense.get_config()
        return config