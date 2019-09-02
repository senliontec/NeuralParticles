import numpy as np
import h5py
import keras
import keras.backend as K
from glob import glob

import json

import math, scipy
from scipy.optimize import linear_sum_assignment

import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from neuralparticles.tensorflow.models.PUNet import PUNet

from neuralparticles.tools.data_helpers import PatchExtractor, get_data_pair, extract_particles, in_bound, get_data, get_nearest_idx

from keras.layers import Input, multiply, concatenate, Conv1D, Lambda, add, Dropout, Dense, Reshape, RepeatVector, Flatten, Permute
from keras.models import Model, load_model
from neuralparticles.tools.uniio import writeParticlesUni, readNumpyOBJ
from neuralparticles.tools.plot_helpers import plot_particles, write_csv
from neuralparticles.tensorflow.tools.eval_helpers import eval_frame, eval_patch
from neuralparticles.tools.param_helpers import *
from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss, approx_match

#python -m neuralparticles.scripts.run_punet data data/3D_data/ test data/Teddy/ config config_3d/version_00.txt real 1 corr 0 res 120

dst_path = getParam("dst", "")

data_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")
test_path = getParam("test", "test/")
real = int(getParam("real", 0)) != 0
corr = int(getParam("corr", 1)) != 0
verbose = int(getParam("verbose", 0)) != 0
gpu = getParam("gpu", "")

t_int = int(getParam("t_int", 1))

temp_coh_dt = float(getParam("temp_coh_dt", 0))
out_res = int(getParam("res", -1))

checkpoint = int(getParam("checkpoint", -1))

patch_pos = np.fromstring(getParam("patch", ""),sep=",")
if len(patch_pos) == 2:
    patch_pos = np.append(patch_pos, [0.5])

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

dst_path += "%s_%s/" % (test_path.split("/")[-2], config['id'])

if verbose:
    print(dst_path)

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

bnd = data_config['bnd']

half_ps = patch_size_ref//2

features = train_config['features']


if checkpoint > 0:
    model_path = data_path + "models/checkpoints/%s_%s_%02d.h5" % (data_config['prefix'], config['id'], checkpoint)
else:
    model_path = data_path + "models/%s_%s_trained.h5" % (data_config['prefix'], config['id'])

config_dict = {**data_config, **pre_config, **train_config}
punet = PUNet(**config_dict)
punet.load_model(model_path)

print(model_path)

if real:
    src_samples = glob(test_path + "real/*.obj")
    src_samples.sort()
else:
    src_samples = glob(test_path + "source/*.obj")
    src_samples.sort()

    ref_samples = glob(test_path + "reference/*.obj")
    ref_samples.sort()

positions = None

tmp_path = dst_path
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
if len(patch_pos) == 3:
    tmp_path += "patch_%d-%d-%d/" % (patch_pos[0],patch_pos[1],patch_pos[2])
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

if corr:
    data = None
    ref_data = None
else:
    data = []
    ref_data = []

plot_z = 0
for i,item in enumerate(src_samples):
    d = readNumpyOBJ(item)[0]
    if not real:
        d_ref = readNumpyOBJ(ref_samples[i])[0]
    plot_z += np.mean(d[:,2])
    if data is None:
        data = np.empty((len(src_samples), d.shape[0], 6))
        if not real:
            ref_data = np.empty((len(ref_samples), d_ref.shape[0], 3))
    
    if corr:
        data[i,:,:3] = d
    else:
        data.append(d)

    if not real:
        if corr:
            ref_data[i] = d_ref
        else:
            ref_data.append(d_ref)
plot_z/=len(data)
print(plot_z)
'''data[...,:3] -= np.min(data[...,:3],axis=(0,1))
data[...,:3] *= (res - 2 * data_config['bnd']) / np.max(data[...,:3])
data[...,:3] += data_config['bnd']'''
"""
def scale_data(data, min_v, max_v, res, bnd):
    data -= min_v
    data *= (res - 2 * bnd) / max_v
    data += bnd
    return data
min_v = np.min(ref_data[...,:3],axis=(0,1))
max_v = np.max(ref_data[...,:3])
"""
src_data_n = None
vel = None
for i,item in enumerate(data):
    if i % t_int != 0:
        continue
    print("Frame: %d" % i)

    src_data = item[...,:3]
    par_aux = {}

    if i+1 < len(data):
        src_data_n = data[i+1][...,:3]
        if corr:
            par_aux['v'] = (src_data_n - src_data) * data_config['fps']
        else:
            par_aux['v'] = np.expand_dims(src_data_n, axis=0) - np.expand_dims(src_data, axis=1)
            match = K.eval(approx_match(K.constant(np.expand_dims(src_data_n, 0)), K.constant(np.expand_dims(src_data, 0))))[0]

            par_aux['v'] = np.sum(np.expand_dims(match, -1) * par_aux['v'], axis=1) * data_config['fps']
            #print(par_aux['v'].shape)
            #print(np.mean(np.sqrt(np.linalg.norm(src_data - np.dot(match, src_data_n), axis=-1))))
            #print(K.eval(emd_loss(K.constant(np.expand_dims(src_data_n, 0)), K.constant(np.expand_dims(src_data + par_aux['v']/data_config['fps'], 0)))))
            #print(K.eval(emd_loss(K.constant(np.expand_dims(src_data_n, 0)), K.constant(np.expand_dims(src_data, 0)))))
    else:
        src_data_n  = data[i-1][...,:3]
        if corr:
            par_aux['v'] = (src_data - src_data_n) * data_config['fps']
        else:
            par_aux['v'] = -np.expand_dims(src_data_n, axis=0) + np.expand_dims(src_data, axis=1)
            match = K.eval(approx_match(K.constant(np.expand_dims(src_data_n, 0)), K.constant(np.expand_dims(src_data, 0))))[0]

            par_aux['v'] = np.sum(np.expand_dims(match, -1) * par_aux['v'], axis=1) * data_config['fps']

        #match = approx_match(pred, gt*zero_mask(gt, self.pad_val))
        #cost = np.linalg.norm(np.expand_dims(src_data_n, axis=0) - np.expand_dims(src_data, axis=1), axis=-1)
        #row_ind, col_ind = linear_sum_assignment(cost)
        #print(row_ind.shape)
        #print(col_ind.shape)

    vel = par_aux['v']
    par_aux['d'] = np.ones((item.shape[0],1))*1000
    par_aux['p'] = np.ones((item.shape[0],1))*1000

    print(np.mean(par_aux['v'],axis=0))
    print(np.mean(np.linalg.norm(par_aux['v'],axis=-1)))
    print(np.max(np.linalg.norm(par_aux['v'],axis=-1)))
    
    patch_extractor = PatchExtractor(src_data, np.zeros((1 if dim == 2 else int(out_res/factor_d[0]), int(out_res/factor_d[0]), int(out_res/factor_d[0]),1)), patch_size, par_cnt, pre_config['surf'], 0 if len(patch_pos) == 3 else 2, aux_data=par_aux, features=features, pad_val=pad_val, bnd=bnd, last_pos=positions, stride_hys=1.0)

    if len(patch_pos) == 3:
        idx = get_nearest_idx(patch_extractor.positions, patch_pos)
        patch = patch_extractor.get_patch(idx, False)

        plot_particles(patch_extractor.positions, [0,int(out_res/factor_d[0])], [0,int(out_res/factor_d[0])], 5, tmp_path + "patch_centers_%03d.png"%i, np.array([patch_extractor.positions[idx]]), np.array([patch_pos]), z=patch_pos[2] if dim == 3 else None)
        patch_pos = patch_extractor.positions[idx] + par_aux['v'][patch_extractor.pos_idx[idx]] / data_config['fps']
        result = eval_patch(punet, [np.array([patch])], tmp_path + "result_%s" + "_%03d"%i, z=None if dim == 2 else 0, verbose=3 if verbose else 1)
       
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
        writeParticlesUni(tmp_path + "result_%03d.uni"%i, hdr, result)

        src = (patch[...,:3] + 1) * 0.5 * patch_size
        if dim == 2:
            src[..., 2] = 0.5

        hdr['dim'] = len(src)
        hdr['dimX'] = int(patch_size)
        hdr['dimY'] = int(patch_size)
        
        writeParticlesUni(tmp_path + "source_%03d.uni"%i, hdr, src)

        if not real:
            ref_patch = extract_particles(ref_data[i], patch_pos * factor_d, par_cnt_dst, half_ps, pad_val)[0]
            hdr['dim'] = len(ref_patch)
            ref_patch = (ref_patch + 1) * 0.5 * patch_size_ref
            if dim == 2:
                ref_patch[..., 2] = 0.5
            writeParticlesUni(tmp_path + "reference_%03d.uni"%i, hdr, ref_patch)

        print("particles: %d -> %d (fac: %.2f)" % (np.count_nonzero(patch[...,0] != pre_config['pad_val']), len(result), (len(result)/np.count_nonzero(patch[...,0] != pre_config['pad_val']))))
                
    else:    
        positions = (patch_extractor.positions + par_aux['v'][patch_extractor.pos_idx] / data_config['fps'])
                
        plot_particles(patch_extractor.positions, [0,int(out_res/factor_d[0])], [0,int(out_res/factor_d[0])], 5, tmp_path + "patch_centers_%03d.png"%i, z=plot_z if dim == 3 else None)

        result = eval_frame(punet, patch_extractor, factor_d[0], tmp_path + "result_%s" + "_%03d"%i, src_data, par_aux, None, out_res, z=None if dim == 2 else out_res//2, verbose=3 if verbose else 1)

        hdr = OrderedDict([ ('dim',len(result)),
                            ('dimX',hres),
                            ('dimY',hres),
                            ('dimZ',1 if dim == 2 else hres),
                            ('elementType',0),
                            ('bytesPerElement',16),
                            ('info',b'\0'*256),
                            ('timestamp',(int)(time.time()*1e6))])
                            
        writeParticlesUni(tmp_path + "result_%03d.uni"%i, hdr, result * hres / out_res)

        if not real:
            hdr['dim'] = len(ref_data[i])
            writeParticlesUni(tmp_path + "reference_%03d.uni"%i, hdr, ref_data[i] * hres / out_res)

        hdr['dim'] = len(src_data)
        hdr['dimX'] = res
        hdr['dimY'] = res
        if dim == 3: hdr['dimZ'] = res
        writeParticlesUni(tmp_path + "source_%03d.uni"%i, hdr, src_data * hres / out_res)


        print("particles: %d -> %d (fac: %.2f)" % (len(src_data), len(result), (len(result)/len(src_data))))
