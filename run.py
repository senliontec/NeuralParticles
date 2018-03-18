import sys, os
sys.path.append("manta/scenes/tools")
sys.path.append("hungarian/")

import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Model, load_model

from gen_patches import gen_patches

from subpixel import *
from spatial_transformer import *

import json
from helpers import *
from uniio import *

import scipy.ndimage.filters as fi
import math
import numpy as np

from hungarian_loss import hungarian_loss

paramUsed = []

src_path = getParam("src", "", paramUsed)
t_start = int(getParam("t_start", -1, paramUsed))
t_end = int(getParam("t_end", -1, paramUsed))
dst_path = getParam("dst", "", paramUsed)

data_path = getParam("data", "data/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0
dataset = int(getParam("dataset", -1, paramUsed))
var = int(getParam("var", 0, paramUsed))

checkpoint = int(getParam("checkpoint", -1, paramUsed))

checkUnusedParam(paramUsed)

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

def filter2D(kernlen, s, fac):
    dirac = np.zeros((kernlen, kernlen))
    dirac[kernlen//2, kernlen//2] = 1
    return np.clip(fi.gaussian_filter(dirac, s) * fac, a_min=None, a_max=1.0)

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

use_particles = train_config['explicit'] != 0

factor_2D = math.sqrt(pre_config['factor'])
patch_size = pre_config['patch_size']
high_patch_size = int(patch_size*factor_2D)
par_cnt = pre_config['par_cnt']

res = data_config['res']
low_res = int(res/factor_2D)

half_ps = high_patch_size//2
#border = int(math.ceil(half_ps-(patch_size//2*factor_2D)))

elem_min = np.vectorize(lambda x,y: min(x,y))
circular_filter = np.reshape(filter2D(high_patch_size, high_patch_size*0.2, 500), (high_patch_size, high_patch_size))

features = train_config['features'][1:]

if checkpoint > 0:
    model_path = data_path + "models/checkpoints/%s_%s_%04d.h5" % (data_config['prefix'], config['id'], checkpoint)
else:
    model_path = data_path + "models/%s_%s.h5" % (data_config['prefix'], config['id'])

model = load_model(model_path, custom_objects={'Subpixel': Subpixel, 'hungarian_loss': hungarian_loss})

for t in range(t_start, t_end):
    data, ref, rot_data, positions = gen_patches(data_path, config_path, dataset+1, t+1, 1, 1, dataset, t)
    prediction = model.predict(x=data, batch_size=train_config['batch_size'])
    result = np.empty((0, 3)) if use_particles else np.ones((res, res, 1))
    for i in range(len(prediction)):
        if use_particles:
            result = np.append(result, (prediction[i] * high_patch_size/2 + np.array([[positions[i,0]*factor_2D,positions[i,1]*factor_2D,0.0]])), axis=0)
        else:
            tmp = prediction[i]
            if pre_config['use_tanh'] != 0:
                tmp = np.arctanh(np.clip(tmp,-.999999,.999999))
            tmp = tmp * circular_filter/pre_config['h_fac']
            insert_patch(result, tmp, factor_2D*positions[i], elem_min)

    if use_particles:
        hdr = OrderedDict([ ('dim',len(result)),
                            ('dimX',res),
                            ('dimY',res),
                            ('dimZ',1),
                            ('elementType',0),
                            ('bytesPerElement',16),
                            ('info',b'\0'*256),
                            ('timestamp',(int)(time.time()*1e6))])
        writeParticles(dst_path%t, hdr, result)
    else:
        hdr = OrderedDict([	('dimX',res),
                            ('dimY',res),
                            ('dimZ',1),
                            ('gridType',17),
                            ('elementType',1),
                            ('bytesPerElement',4),
                            ('info',b'\0'*252),
                            ('dimT',0),
                            ('timestamp',(int)(time.time()*1e6))])
        writeUni(dst_path%t, hdr, result)
    