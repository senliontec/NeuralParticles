import os

import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from neuralparticles.tensorflow.models.PUNet import PUNet

from neuralparticles.tools.data_helpers import PatchExtractor, get_data_pair, extract_particles, in_bound, get_data
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw

from neuralparticles.tools.plot_helpers import plot_particles, write_csv

from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss

from neuralparticles.tensorflow.tools.eval_helpers import eval_frame

import json

import math
import numpy as np

t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", -1))
dst_path = getParam("dst", "")

data_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0
dataset = int(getParam("dataset", -1))
var = int(getParam("var", -1))
gpu = getParam("gpu", "")
real = int(getParam("real", 0)) != 0

temp_coh_dt = float(getParam("temp_coh_dt", 0))
 
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
    d_start = 0 if real else data_config['data_count']
    d_end = d_start + data_config['test_count']
else:
    d_start = dataset
    d_end = d_start + 1

if var < 0:
    var = pre_config['var']

dst_path += "%s_%s-%s_%s" % (data_config['prefix'], data_config['id'], pre_config['id'], train_config['id']) + "_d%03d_var%02d" + ("_real/" if real else "/")
if t_end < 0:
    t_end = data_config['frame_count']

def write_out_particles(particles, d, v, t, suffix, xlim=None, ylim=None, s=1, z=None):
    writeNumpyRaw((dst_path + suffix + "_%03d")%(d,v,t), particles)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.png")%(d,v,t), z=z)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.svg")%(d,v,t), z=z)
    write_csv((dst_path + suffix + "_%03d.csv")%(d,v,t), particles)

def write_out_vel(particles, vel, d, v, t, suffix, xlim=None, ylim=None, s=1, z=None):
    writeNumpyRaw((dst_path + suffix + "_%03d")%(d,v,t), vel)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.png")%(d,v,t), src=particles, vel=vel, z=z)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.svg")%(d,v,t), src=particles, vel=vel, z=z)
    write_csv((dst_path + suffix + "_%03d.csv")%(d,v,t), vel)

if verbose:
    print(dst_path)
    print(t_start)
    print(t_end)

pad_val = pre_config['pad_val']

dim = data_config['dim']
factor_d = math.pow(pre_config['factor'], 1/dim)
patch_size = pre_config['patch_size'] * data_config['res'] / factor_d
patch_size_ref = pre_config['patch_size_ref'] * data_config['res']
par_cnt = pre_config['par_cnt']
par_cnt_dst = pre_config['par_cnt_ref']

hres = data_config['res']
res = int(hres/factor_d)

bnd = data_config['bnd']/factor_d

half_ps = patch_size_ref//2
#border = int(math.ceil(half_ps-(patch_size//2*factor_2D)))

features = train_config['features']

if checkpoint > 0:
    model_path = data_path + "models/checkpoints/%s_%s_%02d.h5" % (data_config['prefix'], config['id'], checkpoint)
else:
    model_path = data_path + "models/%s_%s_trained.h5" % (data_config['prefix'], config['id'])

config_dict = {**data_config, **pre_config, **train_config}
punet = PUNet(**config_dict)
punet.load_model(model_path)

src_data = None
positions = None
for d in range(d_start, d_end):
    for v in range(var):
        if not os.path.exists(dst_path%(d,v)):
            os.makedirs(dst_path%(d,v))
        for t in range(t_start, t_end):
            if temp_coh_dt == 0 or src_data is None:
                if real:
                    path_src = "%sreal/%s_%s_d%03d_%03d" % (data_path, data_config['prefix'], data_config['id'], d, t)
                    src_data, sdf_data, par_aux = get_data(path_src, par_aux=train_config['features'])
                else:
                    (src_data, sdf_data, par_aux), (ref_data, ref_sdf_data) = get_data_pair(data_path, config_path, d, t, v) 
            else:
                src_data = src_data + par_aux['v'] * temp_coh_dt / data_config['fps']

            #src_data = src_data[in_bound(src_data[:,:dim], bnd, res - bnd)]

            patch_extractor = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pre_config['surf'], 2, aux_data=par_aux, features=features, pad_val=pad_val, bnd=bnd, positions=positions)

            if temp_coh_dt != 0 and positions is None:
                positions = patch_extractor.positions

            write_out_particles(patch_extractor.positions, d, v, t, "patch_centers", [0,res], [0,res], 0.1, res//2 if dim == 3 else None)

            result = eval_frame(punet, patch_extractor, factor_d, dst_path%(d,v) + "result_%s" + "_%03d"%t, src_data, par_aux, None if real else ref_data, hres, z=None if dim == 2 else hres//2, verbose=3 if verbose else 1)

            hdr = OrderedDict([ ('dim',len(result)),
                                ('dimX',hres),
                                ('dimY',hres),
                                ('dimZ',1 if dim == 2 else hres),
                                ('elementType',0),
                                ('bytesPerElement',16),
                                ('info',b'\0'*256),
                                ('timestamp',(int)(time.time()*1e6))])

            writeParticlesUni((dst_path + "result_%03d.uni")%(d,v,t), hdr, result)

            if not real:
                hdr['dim'] = len(ref_data)
                writeParticlesUni((dst_path + "reference_%03d.uni")%(d,v,t), hdr, ref_data)

            hdr['dim'] = len(src_data)
            hdr['dimX'] = res
            hdr['dimY'] = res
            if dim == 3: hdr['dimZ'] = res
            writeParticlesUni((dst_path + "source_%03d.uni")%(d,v,t), hdr, src_data)

            print("particles: %d -> %d (fac: %.2f)" % (len(src_data), len(result), (len(result)/len(src_data))))
