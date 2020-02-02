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
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw

from neuralparticles.tools.plot_helpers import plot_particles, write_csv

from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss

from neuralparticles.tensorflow.tools.eval_helpers import eval_frame, eval_patch

import json

import math
import numpy as np

t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", -1))
dst_path = getParam("dst", "")
t_int = int(getParam("t_int", 1))

data_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0
dataset = int(getParam("dataset", -1))
var = int(getParam("var", -1))
gpu = getParam("gpu", "")
real = int(getParam("real", 0)) != 0
out_res = int(getParam("res", -1))

patch_pos = np.fromstring(getParam("patch", ""),sep=",")

if len(patch_pos) == 2:
    patch_pos = np.append(patch_pos, [0.5])

temp_coh_dt = float(getParam("temp_coh_dt", 1))
 
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
#border = int(math.ceil(half_ps-(patch_size//2*factor_2D)))

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

print(model_path)
'''test_src = np.ones((1,par_cnt,8)) * (-2)
test_src[0,:1] = np.random.random((1,8))*0.1 + [0.5,0.5,0.5,0,0,0,0,0]
test = punet.predict([test_src])
if type(test) is list:
    print(test[0][0])
    print(test[1][0][0]*par_cnt_dst)
    print(test_src[0][:1])
    plot_particles(test[0][0][:int(test[1][0][0]*par_cnt_dst)], [-1,1], [-1,1], 5, src=test_src[0])
else:
    print(test)
    plot_particles(test[0], [-1,1], [-1,1], 5, src=test_src[0])'''

for d in range(d_start, d_end):
    for v in range(var):
        tmp_path = dst_path%(d,v)
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        if len(patch_pos) == 3:
            tmp_path += "patch_%d-%d-%d/" % (patch_pos[0],patch_pos[1],patch_pos[2])
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

        src_data = None
        positions = None    

        #patch = None
        for t in range(t_start, t_end, t_int):
            print("Dataset: %d, Frame: %d" % (d,t))
            if temp_coh_dt == 1 or src_data is None:
                if real:
                    path_src = "%sreal/%s_%s_d%03d_%03d" % (data_path, data_config['prefix'], data_config['id'], d, t)
                    src_data, sdf_data, par_aux = get_data(path_src, par_aux=train_config['features'])
                else:
                    (src_data, sdf_data, par_aux), (ref_data, ref_sdf_data, _) = get_data_pair(data_path, config_path, d, t, v) 
            else:
                src_data = src_data + par_aux['v'] * temp_coh_dt / data_config['fps']

            #TODO: fix temp coh dt (sdf is not moving with the particles!!!)
            #src_data = src_data[in_bound(src_data[:,:dim], bnd, res - bnd)]
            
            print(np.mean(np.linalg.norm(par_aux['v'],axis=-1)))
            print(np.max(np.linalg.norm(par_aux['v'],axis=-1)))

            if real and positions is not None:
                positions = src_data[positions]
            patch_extractor = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pre_config['surf'], 0 if len(patch_pos) == 3 else 2, aux_data=par_aux, features=features, pad_val=pad_val, bnd=bnd, last_pos=positions, stride_hys=1.0, shuffle=True)

            positions = patch_extractor.pos_idx if real else (patch_extractor.positions + temp_coh_dt * par_aux['v'][patch_extractor.pos_idx] / data_config['fps'])

            if len(patch_pos) == 3:
                idx = get_nearest_idx(patch_extractor.positions, patch_pos)
                np.random.seed(45)
                patch = patch_extractor.get_patch(idx, False)

                plot_particles(patch_extractor.positions, [0,int(out_res/factor_d[0])], [0,int(out_res/factor_d[0])], 5, tmp_path + "patch_centers_%03d.png"%t, np.array([patch_extractor.positions[idx]]), np.array([patch_pos]), z=patch_pos[2] if dim == 3 else None)
                patch_pos = patch_extractor.positions[idx] + temp_coh_dt * par_aux['v'][patch_extractor.pos_idx[idx]] / data_config['fps']
                if real:
                    result = eval_patch(punet, [np.array([patch])], tmp_path + "result_%s" + "_%03d"%t, z=None if dim == 2 else 0, verbose=3 if verbose else 1)
                else:
                    ref_patch = extract_particles(ref_data, patch_pos * factor_d, par_cnt_dst, half_ps, pad_val)[0]
                    result = eval_patch(punet, [np.array([patch])], tmp_path + "result_%s" + "_%03d"%t, ref_patch, z=None if dim == 2 else 0, verbose=3 if verbose else 1)

                hdr = OrderedDict([ ('dim',len(result)),
                                    ('dimX',int(patch_size_ref)),
                                    ('dimY',int(patch_size_ref)),
                                    ('dimZ',1 if dim == 2 else int(patch_size_ref)),
                                    ('elementType',0),
                                    ('bytesPerElement',16),
                                    ('info',b'\0'*256),
                                    ('timestamp',(int)(time.time()*1e6))])

                result = (result + 1) * 0.5 * patch_size_ref
                if dim == 2:
                    result[..., 2] = 0.5
                writeParticlesUni(tmp_path + "result_%03d.uni"%t, hdr, result)

                if not real:
                    hdr['dim'] = len(ref_patch)
                    ref_patch = (ref_patch + 1) * 0.5 * patch_size_ref
                    if dim == 2:
                        ref_patch[..., 2] = 0.5
                    writeParticlesUni(tmp_path + "reference_%03d.uni"%t, hdr, ref_patch)

                src = (patch[...,:3] + 1) * 0.5 * patch_size
                if dim == 2:
                    src[..., 2] = 0.5

                hdr['dim'] = len(src)
                hdr['dimX'] = int(patch_size)
                hdr['dimY'] = int(patch_size)
                
                writeParticlesUni(tmp_path + "source_%03d.uni"%t, hdr, src)

                print("particles: %d -> %d (fac: %.2f)" % (np.count_nonzero(patch[...,0] != pre_config['pad_val']), len(result), (len(result)/np.count_nonzero(patch[...,0] != pre_config['pad_val']))))
            else:
                write_out_particles(patch_extractor.positions, d, v, t, "patch_centers", [0,int(out_res/factor_d[0])], [0,int(out_res/factor_d[0])], 5, int(out_res/factor_d[0])//2 if dim == 3 else None)

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

                if not real:
                    hdr['dim'] = len(ref_data)
                    writeParticlesUni(tmp_path + "reference_%03d.uni"%t, hdr, ref_data*hres/out_res)

                hdr['dim'] = len(src_data)
                hdr['dimX'] = res
                hdr['dimY'] = res
                if dim == 3: hdr['dimZ'] = res
                writeParticlesUni(tmp_path + "source_%03d.uni"%t, hdr, src_data*hres/out_res)

                print("particles: %d -> %d (fac: %.2f)" % (len(src_data), len(result), (len(result)/len(src_data))))

