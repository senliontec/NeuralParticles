import tensorflow as tf
import keras
import keras.backend as K
from tensorflow.python.framework import ops
import os.path as osp

base_dir = osp.dirname(osp.abspath(__file__))

approxmatch_module = tf.load_op_library(osp.join(base_dir, 'tf_approxmatch_so.so'))


def approx_match(xyz1,xyz2):
	'''
input:
	xyz1 : batch_size * #dataset_points * 3
	xyz2 : batch_size * #query_points * 3
returns:
	match : batch_size * #query_points * #dataset_points
	'''
	return approxmatch_module.approx_match(xyz1,xyz2)
ops.NoGradient('ApproxMatch')
#@tf.RegisterShape('ApproxMatch')
@ops.RegisterShape('ApproxMatch')
def _approx_match_shape(op):
	shape1=op.inputs[0].get_shape().with_rank(3)
	shape2=op.inputs[1].get_shape().with_rank(3)
	return [tf.TensorShape([shape1.dims[0],shape1.dims[1],shape2.dims[1]])]

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
	return [tf.TensorShape([shape1.dims[0], shape1.dims[1]])]

@tf.RegisterGradient('MatchCost')
def _match_cost_grad(op,grad_cost):
	xyz1=op.inputs[0]
	xyz2=op.inputs[1]
	match=op.inputs[2]
	grad_1,grad_2=approxmatch_module.match_cost_grad(xyz1,xyz2,match)
	return [grad_1*tf.expand_dims(grad_cost,axis=-1),grad_2*tf.expand_dims(grad_cost,axis=-1),None]

def emd_loss(y_true, y_pred):
	match = approx_match(y_true, y_pred)
	return match_cost(y_true, y_pred, match)

if __name__=='__main__':
	import numpy as np
	import keras
	import os

	os.environ["CUDA_VISIBLE_DEVICES"] = ""

	np.random.seed(2)
	a = np.random.rand(1000,100,3)
	b = np.random.rand(1000,100,3)

	inputs = keras.layers.Input((100,3))
	x = keras.layers.Flatten()(inputs)
	x = keras.layers.Dense(300)(x)
	x = keras.layers.Reshape((100,3))(x)
	m = keras.models.Model(inputs, x)

	m.compile(optimizer=keras.optimizers.adam(lr=0.001), loss=emd_loss)

	t0 = keras.backend.constant(a)
	t1 = keras.backend.constant(b)

	print(m.fit(a,b,epochs=10))#, epochs=10)