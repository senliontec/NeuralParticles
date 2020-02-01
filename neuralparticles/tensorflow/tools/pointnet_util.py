""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
import tensorflow as tf
from .tf_sampling import farthest_point_sample, gather_point
from .tf_grouping import query_ball_point, group_point, knn_point
from .tf_interpolate import three_nn, three_interpolate
from .zero_mask import zero_mask
import tensorflow as tf
import numpy as np

import keras
import keras.backend as K

from keras.layers import Conv1D, Conv2D, Lambda, multiply, MaxPool2D, concatenate, Reshape, Dropout

def sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec=None, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        tnet_spec: dict (keys: mlp, mlp2), if None do not apply tnet
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        if np.isscalar(radius):
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
        else:
            idx_list = []
            for radius_one, xyz_one, new_xyz_one in zip(tf.unstack(radius,axis=0), tf.unstack(xyz, axis=0),tf.unstack(new_xyz, axis=0)):
                idx_one, _ = query_ball_point(radius_one, nsample, tf.expand_dims(xyz_one, axis=0), tf.expand_dims(new_xyz_one, axis=0))
                idx_list.append(idx_one)
            idx = tf.stack(idx_list, axis=0)
            idx = tf.squeeze(idx, axis=1)

    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if tnet_spec is not None:
        grouped_xyz = tnet(grouped_xyz, tnet_spec)
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            # new_points = tf.concat([grouped_xyz, tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]),grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
            new_points = tf.concat([grouped_xyz, grouped_points],axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        # new_points =  tf.concat([grouped_xyz, tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])], axis=-1)
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

class SampleAndGroup(keras.layers.Layer):
    def __init__(self, npoint, radius, nsample, **kwargs):
        super(SampleAndGroup, self).__init__(**kwargs)

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

    def compute_output_shape(self, input_shape):
        if type(input_shape) is list:
            bs = input_shape[0][0]
            fs0 = input_shape[0][-1]
            fs1 = fs0 + input_shape[1][-1]
        else:
            bs = input_shape[0]
            fs0 = fs1 = input_shape[-1]
        return [(bs, self.npoint, fs0), (bs, self.npoint, self.nsample, fs1)]
    
    def call(self, X, mask=None):
        if type(X) is list:
            return list(sample_and_group(self.npoint, self.radius, self.nsample, X[0], X[1])[:2])
        else:
            return list(sample_and_group(self.npoint, self.radius, self.nsample, X, None)[:2])    

    def get_config(self):
        config = super(SampleAndGroup, self).get_config()
        config['npoint'] = self.npoint
        config['radius'] = self.radius
        config['nsample'] = self.nsample
        return config

def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mask_val=None, dropout=0.0, **kwargs):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            batch_radius: the size of each object
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    new_xyz, new_points = SampleAndGroup(npoint, radius, nsample)(xyz if points is None else [xyz, points])

    if mask_val is not None:
        mask = zero_mask(new_points, mask_val, name="mask_1")

    for num_out_channel in mlp:
        new_points = Dropout(dropout)(new_points)
        new_points = Conv2D(num_out_channel, 1, **kwargs)(new_points)

    if mask_val is not None:
        new_points = multiply([new_points, mask])

    new_points = MaxPool2D([1,nsample])(new_points)
    new_points = Reshape((npoint, mlp[-1]))(new_points) # (batch_size, npoints, mlp[-1])
    return new_xyz, new_points

class Interpolate(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Interpolate, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], input_shape[0][-1])

    def call(self, X, mask=None):
        dist, idx = three_nn(X[1], X[2])
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        return three_interpolate(X[0], idx, weight)

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, dropout=0.0, **kwargs):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    interpolated_points = Interpolate()([points2, xyz1, xyz2])
    if points1 is not None:
        new_points1 = concatenate([interpolated_points, points1], axis=2) # B,ndataset1,nchannel1+nchannel2
    else:
        new_points1 = interpolated_points

    for num_out_channel in mlp:
        if dropout > 0.0:
            new_points1 = Dropout(dropout)(new_points1)
        new_points1 = Conv1D(num_out_channel, 1, **kwargs)(new_points1)

    return new_points1
'''
def sample_and_group_all(xyz, points, use_xyz=True):
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def _pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all,
                       pooling='max', tnet_spec=None, knn=False, use_xyz=True, mask_val=None):

    if group_all:
        nsample = xyz.get_shape()[1].value
        new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
    else:
        new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec, knn, use_xyz)

    if mlp2 is None: mlp2 = []

    if mask_val is not None:
        mask = zero_mask(new_points, mask_val)

    for i, num_out_channel in enumerate(mlp):
        new_points = tf.layers.conv2d(new_points, num_out_channel, [1,1],
                                    padding='VALID', strides=[1,1]) 
    if mask_val is not None:
        new_points = tf.multiply(new_points, mask)

    if pooling=='avg':
        new_points = tf.layers.average_pooling2d(new_points, [1,nsample], [1,1], padding='VALID')
    elif pooling=='weighted_avg':
        dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
        exp_dists = tf.exp(-dists * 5)
        weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
        new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
        new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
    elif pooling=='max':
        new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
    elif pooling=='min':
        new_points = tf.layers.max_pooling2d(-1 * new_points, [1, nsample], [1, 1], padding='VALID')
    elif pooling=='max_and_avg':
        avg_points = tf.layers.max_pooling2d(new_points, [1,nsample], [1,1], padding='VALID')
        max_points = tf.layers.average_pooling2d(new_points, [1,nsample],[1,1], padding='VALID')
        new_points = tf.concat([avg_points, max_points], axis=-1)
        
    if mlp2 is None: mlp2 = []
    for i, num_out_channel in enumerate(mlp2):
        new_points = tf.layers.conv2d(new_points, num_out_channel, [1,1],
                                    padding='VALID', strides=[1,1]) 
    
    new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
    return new_xyz, new_points, idx

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, use_xyz=True):
    '' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    ''
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
    new_points_list = []
    for i in range(len(radius_list)):
        radius = radius_list[i]
        nsample = nsample_list[i]
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = group_point(xyz, idx)
        grouped_xyz -= tf.expand_dims(new_xyz, 2)
        if points is not None:
            grouped_points = group_point(points, idx)
            if use_xyz:
                grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
        else:
            grouped_points = grouped_xyz
        for j,num_out_channel in enumerate(mlp_list[i]):
            grouped_points = tf.layers.conv2d(grouped_points, num_out_channel, [1,1],
                                            padding='VALID', strides=[1,1])
        new_points = tf.reduce_max(grouped_points, axis=[2])
        new_points_list.append(new_points)
    new_points_concat = tf.concat(new_points_list, axis=-1)
    return new_xyz, new_points_concat


def _pointnet_fp_module(xyz1, xyz2, points1, points2, mlp):
    dist, idx = three_nn(xyz1, xyz2)
    dist = tf.maximum(dist, 1e-10)
    norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
    norm = tf.tile(norm,[1,1,3])
    weight = (1.0/dist) / norm
    interpolated_points = three_interpolate(points2, idx, weight)

    if points1 is not None:
        new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
    else:
        new_points1 = interpolated_points
    new_points1 = tf.expand_dims(new_points1, 2)
    for i, num_out_channel in enumerate(mlp):
        new_points1 = tf.layers.conv2d(new_points1, num_out_channel, [1,1],
                                        padding='VALID', strides=[1,1])
    new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
    return new_points1
'''