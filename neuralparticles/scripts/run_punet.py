import numpy as np
import h5py
import keras
from glob import glob

import json

import math

import time
from collections import OrderedDict

from keras.layers import Input, multiply, concatenate, Conv1D, Lambda, add, Dropout, Dense, Reshape, RepeatVector, Flatten, Permute
from keras.models import Model, load_model
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw
from neuralparticles.tools.param_helpers import *
from neuralparticles.tensorflow.tools.pointnet_util import pointnet_sa_module, pointnet_fp_module, Interpolate, SampleAndGroup
from neuralparticles.tensorflow.layers.mult_const_layer import MultConst

def extract_xyz(X, **kwargs):
    return Lambda(lambda x: x[...,:3], **kwargs)(X)

def extract_aux(X, **kwargs):
    return Lambda(lambda x: x[...,3:], **kwargs)(X)

data_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")

temp_coh_dt = float(getParam("temp_coh_dt", 0))

checkUnusedParams()

dst_path = data_path + "result/"
if not os.path.exists(dst_path):
	os.makedirs(dst_path)

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

dst_path += "%s_%s-%s_%s" % (data_config['prefix'], data_config['id'], pre_config['id'], train_config['id']) + "_d%03d_var%02d/"

features = train_config['features']
par_cnt = pre_config['par_cnt']
par_cnt_dst = pre_config['par_cnt_ref']
norm_factor = 1.
fac = train_config['fac']
l2_reg = train_config['l2_reg']
dim = data_config['dim']
factor_d = math.pow(pre_config['factor'], 1/dim)
hres = data_config['res']
res = int(hres/factor_d)

point_cnt = 5000

activation = keras.activations.relu
inputs = Input((point_cnt, 3 + len(features) + (2 if 'v' in features or 'n' in features else 0)), name="main_input")
input_xyz = extract_xyz(inputs, name="extract_pos")
input_points = input_xyz

if len(features) > 0:
    input_points = extract_aux(inputs, name="extract_aux")
    input_points = MultConst(1./norm_factor, name="normalization")(input_points)
    input_points = concatenate([input_xyz, input_points], axis=-1, name='input_concatenation')

l1_xyz, l1_points = pointnet_sa_module(input_xyz, input_points, point_cnt, 0.05, fac*4, 
                                        [fac*4,
                                        fac*4,
                                        fac*8], kernel_regularizer=keras.regularizers.l2(l2_reg), activation=activation)
l2_xyz, l2_points = pointnet_sa_module(l1_xyz, l1_points, point_cnt//2, 0.1, fac*4, 
                                        [fac*8,
                                        fac*8,
                                        fac*16], activation=activation, kernel_regularizer=keras.regularizers.l2(l2_reg))
l3_xyz, l3_points = pointnet_sa_module(l2_xyz, l2_points, point_cnt//4, 0.2, fac*4, 
                                        [fac*16,
                                        fac*16,
                                        fac*32], activation=activation, kernel_regularizer=keras.regularizers.l2(l2_reg))
l4_xyz, l4_points = pointnet_sa_module(l3_xyz, l3_points, point_cnt//8, 0.3, fac*4, 
                                        [fac*32,
                                        fac*32,
                                        fac*64], activation=activation, kernel_regularizer=keras.regularizers.l2(l2_reg))

# interpoliere die features in l2_points auf die Punkte in x
up_l2_points = pointnet_fp_module(input_xyz, l2_xyz, None, l2_points, [fac*8], kernel_regularizer=keras.regularizers.l2(l2_reg), activation=activation)
up_l3_points = pointnet_fp_module(input_xyz, l3_xyz, None, l3_points, [fac*8], kernel_regularizer=keras.regularizers.l2(l2_reg), activation=activation)
up_l4_points = pointnet_fp_module(input_xyz, l4_xyz, None, l4_points, [fac*8], kernel_regularizer=keras.regularizers.l2(l2_reg), activation=activation)

x = concatenate([up_l4_points, up_l3_points, up_l2_points, l1_points, input_xyz], axis=-1)
x_t = x
l = []
for i in range(par_cnt_dst//par_cnt):
    tmp = Conv1D(fac*32, 1, name="expansion_1_"+str(i+1), kernel_regularizer=keras.regularizers.l2(l2_reg), activation=activation)(x)
    tmp = Conv1D(fac*16, 1, name="expansion_2_"+str(i+1), kernel_regularizer=keras.regularizers.l2(l2_reg), activation=activation)(tmp)
    l.append(tmp)
x = concatenate(l, axis=1, name="pixel_conv") if par_cnt_dst//par_cnt > 1 else l[0]

x = Conv1D(fac*8, 1, name="coord_reconstruction_1", kernel_regularizer=keras.regularizers.l2(l2_reg), activation=activation)(x)

x = Conv1D(3, 1, name="coord_reconstruction_2")(x)

out = x

m = Model(inputs=inputs, outputs=out)
m.load_weights(data_path + "models/%s_%s_trained.h5" % (data_config['prefix'], config['id']))

samples = glob(data_config['test'] + "*.xyz")
samples.sort()

for i,item in enumerate(samples):
    if not os.path.exists(dst_path%(i,0)):
        os.makedirs(dst_path%(i,0))
    input = np.loadtxt(item)
    input = np.expand_dims(input, axis=0)
    pred = (m.predict(input)[0]+1) * hres/2
    input = (input[0,:,:3]+1) * res/2
    
    hdr = OrderedDict([ ('dim',len(pred)),
                        ('dimX',hres),
                        ('dimY',hres),
                        ('dimZ',1 if dim == 2 else hres),
                        ('elementType',0),
                        ('bytesPerElement',16),
                        ('info',b'\0'*256),
                        ('timestamp',(int)(time.time()*1e6))])

    writeParticlesUni((dst_path + "result_%03d.uni")%(i,0,0), hdr, pred)
    hdr['dim'] = len(pred)
    writeParticlesUni((dst_path + "reference_%03d.uni")%(i,0,0), hdr, pred)
    hdr['dim'] = len(input)
    hdr['dimX'] = res
    hdr['dimY'] = res
    if dim == 3: hdr['dimZ'] = res
    writeParticlesUni((dst_path + "source_%03d.uni")%(i,0,0), hdr, input)