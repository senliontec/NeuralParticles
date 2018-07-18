import os

import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from neuralparticles.tensorflow.models.PUNet import PUNet

from neuralparticles.tools.data_helpers import PatchExtractor, get_data_pair, extract_particles, in_bound
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw

from neuralparticles.tools.plot_helpers import plot_particles, write_csv

from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss

from neuralparticles.tensorflow.tools.eval_helpers import eval_frame

import json

import math
import numpy as np

t_start = int(getParam("t_start", -1))
t_end = int(getParam("t_end", -1))
dst_path = getParam("dst", "")

data_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0
dataset = int(getParam("dataset", -1))
var = int(getParam("var", 0))
gpu = getParam("gpu", "")

checkpoint = int(getParam("checkpoint", -1))

checkUnusedParams()

if dst_path == "":
    dst_path = data_path + "result/"

if not os.path.exists(dst_path):
	os.makedirs(dst_path)

if not gpu is "":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
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

dst_path += "%s_%s-%s_d%03d_var%02d/" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var)
if not os.path.exists(dst_path):
	os.makedirs(dst_path)

if t_start < 0:
    t_start = min(train_config['t_start'], data_config['frame_count']-1)
if t_end < 0:
    t_end = min(train_config['t_end'], data_config['frame_count'])

def write_out_particles(particles, t, suffix, xlim=None, ylim=None, s=1, z=None):
    writeNumpyRaw((dst_path + suffix + "_%03d")%t, particles)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.png")%t, z=z)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.svg")%t, z=z)
    write_csv((dst_path + suffix + "_%03d.csv")%t, particles)

def write_out_vel(particles, vel, t, suffix, xlim=None, ylim=None, s=1, z=None):
    writeNumpyRaw((dst_path + suffix + "_%03d")%t, vel)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.png")%t, src=particles, vel=vel, z=z)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.svg")%t, src=particles, vel=vel, z=z)
    write_csv((dst_path + suffix + "_%03d.csv")%t, vel)

if verbose:
    print(dst_path)
    print(t_start)
    print(t_end)

pad_val = pre_config['pad_val']

dim = data_config['dim']
factor_d = math.pow(pre_config['factor'], 1/dim)
patch_size = pre_config['patch_size']
ref_patch_size = pre_config['patch_size_ref']
par_cnt = pre_config['par_cnt']
par_cnt_dst = pre_config['par_cnt_ref']

hres = data_config['res']
res = int(hres/factor_d)

bnd = data_config['bnd']/factor_d

half_ps = ref_patch_size//2
#border = int(math.ceil(half_ps-(patch_size//2*factor_2D)))

features = train_config['features'][1:]

if checkpoint > 0:
    model_path = data_path + "models/checkpoints/%s_%s_%04d.h5" % (data_config['prefix'], config['id'], checkpoint)
else:
    model_path = data_path + "models/%s_%s_trained.h5" % (data_config['prefix'], config['id'])

config_dict = {**data_config, **pre_config, **train_config}
punet = PUNet(**config_dict)
punet.load_model(model_path)

for t in range(t_start, t_end):
    (src_data, sdf_data, par_aux), (ref_data, ref_sdf_data) = get_data_pair(data_path, config_path, dataset, t, var) 

    src_data = src_data[in_bound(src_data[:,:dim], bnd, res - bnd)]

    patch_extractor = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pre_config['surf'], pre_config['stride'], aux_data=par_aux, features=features, pad_val=pad_val, bnd=bnd)

    write_out_particles(patch_extractor.positions, t, "patch_centers", [0,res], [0,res], 0.1, res//2 if dim == 3 else None)

    result = eval_frame(punet, patch_extractor, factor_d, dst_path + "result_%03d"%t, src_data, par_aux, ref_data, hres, z=None if dim == 2 else hres//2, verbose=3 if verbose else 1)

    hdr = OrderedDict([ ('dim',len(result)),
                        ('dimX',hres),
                        ('dimY',hres),
                        ('dimZ',1 if dim == 2 else hres),
                        ('elementType',0),
                        ('bytesPerElement',16),
                        ('info',b'\0'*256),
                        ('timestamp',(int)(time.time()*1e6))])
    writeParticlesUni(dst_path + "result_%03d.uni"%t, hdr, result)
    
    print("particles: %d -> %d (fac: %.2f)" % (len(src_data), len(result), (len(result)/len(src_data))))
