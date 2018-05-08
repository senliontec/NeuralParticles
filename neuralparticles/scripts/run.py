import os

import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Model, load_model

from neuralparticles.tools.data_helpers import PatchExtractor, get_data_pair
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.uniio import writeParticles

import json
#from uniio import *

import math
import numpy as np

src_path = getParam("src", "")
t_start = int(getParam("t_start", -1))
t_end = int(getParam("t_end", -1))
dst_path = getParam("dst", "")

data_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0
dataset = int(getParam("dataset", -1))
var = int(getParam("var", 0))

checkpoint = int(getParam("checkpoint", -1))

checkUnusedParams()

if dst_path == "" and not os.path.exists(data_path + "result"):
	os.makedirs(data_path + "result")

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

if verbose:
    print("Config Loaded:")
    print(config)
    print(data_config)
    print(pre_config)
    print(train_config)

if dataset < 0:
    dataset = int(data_config['data_count']*train_config['train_split'])

if src_path == "":
    src_path = data_path + "source/%s_%s-%s_d%03d_var%02d" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var) + "_%03d"
if dst_path == "":
    dst_path = data_path + "result/%s_result.uni" % os.path.basename(src_path)
if t_start < 0:
    t_start = train_config['t_start']
if t_end < 0:
    t_end = train_config['t_end']

if verbose:
    print(src_path)
    print(dst_path)
    print(t_start)
    print(t_end)

loss_mode = train_config['loss']

particle_loss = keras.losses.mse

if loss_mode == 'hungarian_loss':
    from neuralparticles.tensorflow.losses.hungarian_loss import hungarian_loss
    particle_loss = hungarian_loss
elif loss_mode == 'emd_loss':
    from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss
    particle_loss = emd_loss
elif loss_mode == 'chamfer_loss':
    from neuralparticles.tensorflow.losses.tf_nndistance import chamfer_loss
    particle_loss = chamfer_loss
else:
    print("No matching loss specified! Fallback to MSE loss.")

factor_2D = math.sqrt(pre_config['factor'])
patch_size = pre_config['patch_size']
ref_patch_size = pre_config['patch_size_ref']
par_cnt = pre_config['par_cnt']
par_cnt_dst = pre_config['par_cnt_ref']

res = data_config['res']
low_res = int(res/factor_2D)

half_ps = ref_patch_size//2
#border = int(math.ceil(half_ps-(patch_size//2*factor_2D)))

features = train_config['features'][1:]

if checkpoint > 0:
    model_path = data_path + "models/checkpoints/%s_%s_%04d.h5" % (data_config['prefix'], config['id'], checkpoint)
else:
    model_path = data_path + "models/%s_%s.h5" % (data_config['prefix'], config['id'])

model = load_model(model_path, custom_objects={loss_mode: particle_loss})

for t in range(t_start, t_end):
    (src_data, sdf_data), (ref_data, ref_sdf_data) = get_data_pair(data_path, config_path, dataset, t, var) 

    patch_extractor = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pre_config['surf'], pre_config['stride'], 4)
    while(True):
        src = patch_extractor.get_patch()
        if src is None:
            break
        patch_extractor.set_patch(model.predict(x=np.array([src]))[0])
    result = patch_extractor.data
    
    hdr = OrderedDict([ ('dim',len(result)),
                        ('dimX',res),
                        ('dimY',res),
                        ('dimZ',1),
                        ('elementType',0),
                        ('bytesPerElement',16),
                        ('info',b'\0'*256),
                        ('timestamp',(int)(time.time()*1e6))])
    writeParticles(dst_path%t, hdr, result)