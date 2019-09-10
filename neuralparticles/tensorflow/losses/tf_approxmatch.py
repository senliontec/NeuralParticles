import tensorflow as tf
from tensorflow.python.framework import ops
import os.path as osp

base_dir = osp.dirname(osp.abspath(__file__))

approxmatch_module = tf.load_op_library(osp.join(base_dir, 'tf_approxmatch_so.so'))


def approx_match(xyz1,xyz2,n=None,m=None):
	'''
input:
	xyz1 : batch_size * #dataset_points * 3
	xyz2 : batch_size * #query_points * 3
	n : batch_size * 1
	m : batch_size * 1
returns:
	match : batch_size * #query_points * #dataset_points
	'''
	if n is None: 
		n = tf.tile(tf.shape(xyz1)[1:2], tf.shape(xyz1)[:1])
	if m is None: 
		m = tf.tile(tf.shape(xyz2)[1:2], tf.shape(xyz2)[:1])
	return approxmatch_module.approx_match(xyz1,xyz2,n,m)
ops.NoGradient('ApproxMatch')
#@tf.RegisterShape('ApproxMatch')
@ops.RegisterShape('ApproxMatch')
def _approx_match_shape(op):
	shape1=op.inputs[0].get_shape().with_rank(3)
	shape2=op.inputs[1].get_shape().with_rank(3)
	return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[1]])]

def match_cost(xyz1,xyz2,match):
	'''
input:
	xyz1 : batch_size * #dataset_points * 3
	xyz2 : batch_size * #query_points * 3
	match : batch_size * #query_points * #dataset_points
returns:
	cost : batch_size
	'''
	return approxmatch_module.match_cost(xyz1,xyz2,match)
#@tf.RegisterShape('MatchCost')
@ops.RegisterShape('MatchCost')
def _match_cost_shape(op):
	shape1=op.inputs[0].get_shape().with_rank(3)
	shape2=op.inputs[1].get_shape().with_rank(3)
	shape3=op.inputs[2].get_shape().with_rank(3)
	return [tf.TensorShape([shape1.dims[0]])]
@tf.RegisterGradient('MatchCost')
def _match_cost_grad(op,grad_cost):
	xyz1=op.inputs[0]
	xyz2=op.inputs[1]
	match=op.inputs[2]
	grad_1,grad_2=approxmatch_module.match_cost_grad(xyz1,xyz2,match)
	return [grad_1*tf.expand_dims(tf.expand_dims(grad_cost,1),2),grad_2*tf.expand_dims(tf.expand_dims(grad_cost,1),2),None]

def emd_loss(y_true, y_pred, n=None, m=None):
	if n is None: 
		n = tf.tile(tf.shape(y_true)[1:2], tf.shape(y_true)[:1])
	if m is None: 
		m = tf.tile(tf.shape(y_pred)[1:2], tf.shape(y_pred)[:1])
	match = approx_match(y_true, y_pred, n, m)
	return match_cost(y_true, y_pred, match)/tf.cast(tf.maximum(n,m), tf.float32)

def approx_vel(pos_0, pos_1, n=None, m=None):
	vel = tf.expand_dims(pos_1, axis=2) - tf.expand_dims(pos_0, axis=1)
	match = tf.expand_dims(approx_match(pos_0, pos_1, n, m), axis=-1)
	return tf.reduce_sum(vel*match, axis=1)

if __name__=='__main__':
	import numpy as np
	import keras
	import os

	#os.environ["CUDA_VISIBLE_DEVICES"] = ""

	np.random.seed(2)
	a = np.random.rand(10000,10,3)
	b = np.random.rand(1000,10,3)

	inputs = keras.layers.Input((10,3))
	x = keras.layers.Flatten()(inputs)
	x = keras.layers.Dense(30)(x)
	x = keras.layers.Reshape((10,3))(x)
	m = keras.models.Model(inputs, x)

	m.compile(optimizer=keras.optimizers.adam(lr=0.001), loss=emd_loss)

	t0 = keras.backend.constant(a)
	t1 = keras.backend.constant(b)

	print(m.fit(a,a,epochs=20))#, epochs=10)
	print(b[0])
	print(m.predict(b[:1]))