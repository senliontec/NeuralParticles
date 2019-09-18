import os

import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from neuralparticles.tensorflow.models.PUNet import PUNet

from neuralparticles.tools.data_helpers import PatchExtractor, get_data_pair, extract_particles, in_bound, get_data, get_nearest_idx
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw, writeNumpyOBJ

from neuralparticles.tools.plot_helpers import plot_particles, write_csv

from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss

from neuralparticles.tensorflow.tools.eval_helpers import eval_frame, eval_patch

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
gpu = getParam("gpu", "")
real = int(getParam("real", 0)) != 0
out_res = int(getParam("res", -1))
 
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

dst_path += "%s_%s-%s_%s" % (data_config['prefix'], data_config['id'], pre_config['id'], train_config['id']) + "_d%03d_var%02d" + ("_real/" if real else "/")
if t_end < 0:
    t_end = data_config['frame_count']

if verbose:
    print(dst_path)
    print(t_start)
    print(t_end)

pad_val = pre_config['pad_val']

dim = data_config['dim']
factor_d = math.pow(pre_config['factor'], 1/dim)
factor_d = np.array([factor_d, factor_d, 1 if dim == 2 else factor_d])
patch_size = pre_config['patch_size'] * data_config['res'] / factor_d[0]
patch_size_ref = pre_config['patch_size_ref'] * data_config['res']
par_cnt = pre_config['par_cnt']
par_cnt_dst = pre_config['par_cnt_ref']

hres = data_config['res']
res = int(hres/factor_d[0])

if out_res < 0:
    out_res = hres

bnd = data_config['bnd']/factor_d[0]

half_ps = patch_size_ref//2

features = train_config['features']


def write_out_particles(particles, d, v, t, suffix, xlim=None, ylim=None, s=5, z=None):
    writeNumpyRaw((dst_path + suffix + "_%03d")%(d,v,t), particles)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.png")%(d,v,t), z=z)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.svg")%(d,v,t), z=z)
    write_csv((dst_path + suffix + "_%03d.csv")%(d,v,t), particles)

def write_out_vel(particles, vel, d, v, t, suffix, xlim=None, ylim=None, s=5, z=None):
    writeNumpyRaw((dst_path + suffix + "_%03d")%(d,v,t), vel)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.png")%(d,v,t), src=particles, vel=vel, z=z)
    plot_particles(particles, xlim, ylim, s, (dst_path + suffix + "_%03d.svg")%(d,v,t), src=particles, vel=vel, z=z)
    write_csv((dst_path + suffix + "_%03d.csv")%(d,v,t), vel)


if checkpoint > 0:
    model_path = data_path + "models/checkpoints/%s_%s_%02d.h5" % (data_config['prefix'], config['id'], checkpoint)
else:
    model_path = data_path + "models/%s_%s_trained.h5" % (data_config['prefix'], config['id'])

config_dict = {**data_config, **pre_config, **train_config}
punet = PUNet(**config_dict)
punet.load_model(model_path)

for d in range(d_start, d_end):
    tmp_path = dst_path%(d,0)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    if real:
        path_src = "%sreal/%s_%s_d%03d_%03d" % (data_path, data_config['prefix'], data_config['id'], d, t_start)
        src_data, sdf_data, par_aux = get_data(path_src, par_aux=train_config['features'])
    else:
        (src_data, sdf_data, par_aux), (ref_data, ref_sdf_data, _) = get_data_pair(data_path, config_path, d, t_start, 0) 

    patch_extractor = [None, None, None]
    patch_extractor[0] = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pre_config['surf'], 0 if len(patch_pos) == 3 else 2, aux_data=par_aux, features=features, pad_val=pad_val, bnd=bnd, last_pos=positions, stride_hys=1.0, shuffle=True)
    positions = patch_extractor[0].positions - par_aux['v'][patch_extractor[0].pos_idx] / data_config['fps']
    prev_src = src_data - par_aux['v'] / data_config['fps']
    patch_extractor[1] = PatchExtractor(prev_src, sdf_data, patch_size, par_cnt, pre_config['surf'], 0 if len(patch_pos) == 3 else 2, aux_data=par_aux, features=features, pad_val=pad_val, bnd=bnd, last_pos=positions, stride_hys=1.0, shuffle=True)
                
    for t in range(t_start+1, t_end+1):        
        print("Dataset: %d, Frame: %d" % (d,t))
        positions = patch_extractor[0].positions + par_aux['v'][patch_extractor[0].pos_idx] / data_config['fps']

        if t == t_end:
            src_data = src_data + par_aux['v'] / data_config['fps']
        else:
            if real:
                path_src = "%sreal/%s_%s_d%03d_%03d" % (data_path, data_config['prefix'], data_config['id'], d, t)
                src_data, sdf_data, par_aux = get_data(path_src, par_aux=train_config['features'])
            else:
                (src_data, sdf_data, par_aux), (ref_data, ref_sdf_data, _) = get_data_pair(data_path, config_path, d, t, 0) 

        patch_extractor[2] = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pre_config['surf'], 0 if len(patch_pos) == 3 else 2, aux_data=par_aux, features=features, pad_val=pad_val, bnd=bnd, last_pos=positions, stride_hys=1.0, shuffle=True)


        write_out_particles(patch_extractor[0].positions, d, 0, t, "patch_centers", [0,int(out_res/factor_d[0])], [0,int(out_res/factor_d[0])], 5, int(out_res/factor_d[0])//2 if dim == 3 else None)

        result = eval_frame(punet, patch_extractor, factor_d[0], tmp_path + "result_%s" + "_%03d"%t, src_data, par_aux, None if real else ref_data, out_res, z=None if dim == 2 else out_res//2, verbose=3 if verbose else 1)

        hdr = OrderedDict([ ('dim',len(result)),
                            ('dimX',hres),
                            ('dimY',hres),
                            ('dimZ',1 if dim == 2 else hres),
                            ('elementType',0),
                            ('bytesPerElement',16),
                            ('info',b'\0'*256),
                            ('timestamp',(int)(time.time()*1e6))])

        writeParticlesUni(tmp_path + "result_%03d.uni"%t, hdr, result*hres/out_res)

        writeNumpyOBJ(tmp_path + "result_%03d.obj"%t, result*hres/out_res)

        if not real:
            hdr['dim'] = len(ref_data)
            writeParticlesUni(tmp_path + "reference_%03d.uni"%t, hdr, ref_data*hres/out_res)
            writeNumpyOBJ(tmp_path + "reference_%03d.obj"%t, ref_data*hres/out_res)

        hdr['dim'] = len(src_data)
        hdr['dimX'] = res
        hdr['dimY'] = res
        if dim == 3: hdr['dimZ'] = res
        writeParticlesUni(tmp_path + "source_%03d.uni"%t, hdr, src_data*hres/out_res)
        writeNumpyOBJ(tmp_path + "source_%03d.obj"%t, src_data*hres/out_res)

        print("particles: %d -> %d (fac: %.2f)" % (len(src_data), len(result), (len(result)/len(src_data))))

        patch_extractor[1] = patch_extractor[0]
        patch_extractor[0] = patch_extractor[2]

