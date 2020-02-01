import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops
import os.path as osp

base_dir = osp.dirname(osp.abspath(__file__))

nn_distance_module = tf.load_op_library(osp.join(base_dir, 'tf_nndistance_so.so'))


def nn_distance(xyz1, xyz2):
	'''
	Computes the distance of nearest neighbors for a pair of point clouds
	input: xyz1: (batch_size,#points_1,3)  the first point cloud
	input: xyz2: (batch_size,#points_2,3)  the second point cloud
	output: dist1: (batch_size,#point_1)   distance from first to second
	output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
	output: dist2: (batch_size,#point_2)   distance from second to first
	output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
	'''
	
	return nn_distance_module.nn_distance(xyz1,xyz2)

#@tf.RegisterShape('NnDistance')
@ops.RegisterShape('NnDistance')
def _nn_distance_shape(op):
	shape1=op.inputs[0].get_shape().with_rank(3)
	shape2=op.inputs[1].get_shape().with_rank(3)
	return [tf.TensorShape([shape1.dims[0],shape1.dims[1]]),tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
		tf.TensorShape([shape2.dims[0],shape2.dims[1]]),tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op,grad_dist1,grad_idx1,grad_dist2,grad_idx2):
	xyz1=op.inputs[0]
	xyz2=op.inputs[1]
	idx1=op.outputs[1]
	idx2=op.outputs[3]
	return nn_distance_module.nn_distance_grad(xyz1,xyz2,grad_dist1,idx1,grad_dist2,idx2)

def batch_gather(params, indices, name=None):
  """Gather slices from `params` according to `indices` with leading batch dims.
  This operation assumes that the leading dimensions of `indices` are dense,
  and the gathers on the axis corresponding to the last dimension of `indices`.
  More concretely it computes:
  result[i1, ..., in] = params[i1, ..., in-1, indices[i1, ..., in]]
  Therefore `params` should be a Tensor of shape [A1, ..., AN, B1, ..., BM],
  `indices` should be a Tensor of shape [A1, ..., AN-1, C] and `result` will be
  a Tensor of size `[A1, ..., AN-1, C, B1, ..., BM]`.
  In the case in which indices is a 1D tensor, this operation is equivalent to
  `tf.gather`.
  See also `tf.gather` and `tf.gather_nd`.
  Args:
    params: A Tensor. The tensor from which to gather values.
    indices: A Tensor. Must be one of the following types: int32, int64. Index
        tensor. Must be in range `[0, params.shape[axis]`, where `axis` is the
        last dimension of `indices` itself.
    name: A name for the operation (optional).
  Returns:
    A Tensor. Has the same type as `params`.
  Raises:
    ValueError: if `indices` has an unknown shape.
  """

  with ops.name_scope(name):
    indices = ops.convert_to_tensor(indices, name="indices")
    params = ops.convert_to_tensor(params, name="params")
    indices_shape = tf.shape(indices)
    params_shape = tf.shape(params)
    ndims = indices.shape.ndims
    if ndims is None:
      raise ValueError("batch_gather does not allow indices with unknown "
                       "shape.")
    batch_indices = indices
    accum_dim_value = 1
    for dim in range(ndims-1, 0, -1):
      dim_value = params_shape[dim-1]
      accum_dim_value *= params_shape[dim]
      dim_indices = gen_math_ops._range(0, dim_value, 1)
      dim_indices *= accum_dim_value
      dim_shape = tf.stack([1] * (dim - 1) + [dim_value] + [1] * (ndims - dim),
                        axis=0)
      batch_indices += tf.cast(tf.reshape(dim_indices, dim_shape), tf.int64)

    flat_indices = tf.reshape(batch_indices, [-1])
    outer_shape = params_shape[ndims:]
    flat_inner_shape = gen_math_ops.prod(
        params_shape[:ndims], [0], False)

    flat_params = tf.reshape(
        params, tf.concat([[flat_inner_shape], outer_shape], axis=0))
    flat_result = tf.gather(flat_params, flat_indices)
    result = tf.reshape(flat_result, tf.concat([indices_shape, outer_shape], axis=0))
    final_shape = indices.get_shape()[:ndims-1].merge_with(
        params.get_shape()[:ndims -1])
    final_shape = final_shape.concatenate(indices.get_shape()[ndims-1])
    final_shape = final_shape.concatenate(params.get_shape()[ndims:])
    result.set_shape(final_shape)
    return result

def nn_index(y_true, y_pred):
	distance = tf.reduce_sum(tf.square(tf.expand_dims(y_true,2) - tf.expand_dims(y_pred,1)), -1)
	return tf.argmin(distance, -1)

def chamfer_loss(y_true, y_pred):
    cost_p1_p2, _, cost_p2_p1, _ = nn_distance(y_pred, y_true)
    return tf.reduce_mean(cost_p1_p2, axis=-1) + tf.reduce_mean(cost_p2_p1, axis=-1)

if __name__=='__main__':
	import numpy as np
	import random
	import time
	from tensorflow.python.kernel_tests.gradient_checker import compute_gradient
	random.seed(100)
	np.random.seed(100)
	with tf.Session('') as sess:
		xyz1=np.random.randn(32,16384,3).astype('float32')
		xyz2=np.random.randn(32,1024,3).astype('float32')
		with tf.device('/gpu:0'):
			inp1=tf.Variable(xyz1)
			inp2=tf.constant(xyz2)
			reta,retb,retc,retd=nn_distance(inp1,inp2)
			loss=tf.reduce_sum(reta)+tf.reduce_sum(retc)
			train=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
		sess.run(tf.initialize_all_variables())
		t0=time.time()
		t1=t0
		best=1e100
		for i in xrange(100):
			trainloss,_=sess.run([loss,train])
			newt=time.time()
			best=min(best,newt-t1)
			print(i,trainloss,(newt-t0)/(i+1),best)
			t1=newt
		#print sess.run([inp1,retb,inp2,retd])
		#grads=compute_gradient([inp1,inp2],[(16,32,3),(16,32,3)],loss,(1,),[xyz1,xyz2])
		#for i,j in grads:
			#print i.shape,j.shape,np.mean(np.abs(i-j)),np.mean(np.abs(i)),np.mean(np.abs(j))
		#for i in xrange(10):
			#t0=time.time()
			#a,b,c,d=sess.run([reta,retb,retc,retd],feed_dict={inp1:xyz1,inp2:xyz2})
			#print 'time',time.time()-t0
		#print a.shape,b.shape,c.shape,d.shape
		#print a.dtype,b.dtype,c.dtype,d.dtype
		#samples=np.array(random.sample(range(xyz2.shape[1]),100),dtype='int32')
		#dist1=((xyz1[:,samples,None,:]-xyz2[:,None,:,:])**2).sum(axis=-1).min(axis=-1)
		#idx1=((xyz1[:,samples,None,:]-xyz2[:,None,:,:])**2).sum(axis=-1).argmin(axis=-1)
		#print np.abs(dist1-a[:,samples]).max()
		#print np.abs(idx1-b[:,samples]).max()
		#dist2=((xyz2[:,samples,None,:]-xyz1[:,None,:,:])**2).sum(axis=-1).min(axis=-1)
		#idx2=((xyz2[:,samples,None,:]-xyz1[:,None,:,:])**2).sum(axis=-1).argmin(axis=-1)
		#print np.abs(dist2-c[:,samples]).max()
		#print np.abs(idx2-d[:,samples]).max()

