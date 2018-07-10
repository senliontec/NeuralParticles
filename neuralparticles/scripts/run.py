import os

import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Model, load_model
import keras.backend as K

from neuralparticles.tools.data_helpers import PatchExtractor, get_data_pair, extract_particles
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw

from neuralparticles.tools.plot_helpers import plot_particles, write_csv

from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss
from neuralparticles.tensorflow.losses.tf_nndistance import chamfer_loss

from neuralparticles.tensorflow.tools.zero_mask import zero_mask, trunc_mask

from neuralparticles.tensorflow.tools.eval_helpers import eval_frame, eval_patch

import json
#from uniio import *

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

checkpoint = int(getParam("checkpoint", -1))

checkUnusedParams()

if not os.path.exists(data_path + "result"):
	os.makedirs(data_path + "result")

if not os.path.exists(data_path + "result/npy"):
	os.makedirs(data_path + "result/npy")

if not os.path.exists(data_path + "result/csv"):
	os.makedirs(data_path + "result/csv")

if not os.path.exists(data_path + "result/pdf"):
	os.makedirs(data_path + "result/pdf")

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

file_name = "%s_%s-%s_d%03d_var%02d" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var)
if dst_path == "":
    dst_path = data_path + "result/%s_result"%file_name + "_%03d.uni"
if t_start < 0:
    t_start = train_config['t_start']
if t_end < 0:
    t_end = train_config['t_end']

npy_path = data_path + "result/npy/%s"%file_name
pdf_path = data_path + "result/pdf/%s"%file_name
csv_path = data_path + "result/csv/%s"%file_name

def write_out_particles(particles, t, suffix, xlim=None, ylim=None, s=1, z=None):
    writeNumpyRaw((npy_path + suffix + "_%03d")%t, particles)
    plot_particles(particles, xlim, ylim, s, (pdf_path + suffix + "_%03d.png")%t, z=z)
    plot_particles(particles, xlim, ylim, s, (pdf_path + suffix + "_%03d.pdf")%t, z=z)
    write_csv((csv_path + suffix + "_%03d.csv")%t, particles)

def write_out_vel(particles, vel, t, suffix, xlim=None, ylim=None, s=1, z=None):
    writeNumpyRaw((npy_path + suffix + "_%03d")%t, vel)
    plot_particles(particles, xlim, ylim, s, (pdf_path + suffix + "_%03d.png")%t, src=particles, vel=vel, z=z)
    plot_particles(particles, xlim, ylim, s, (pdf_path + suffix + "_%03d.pdf")%t, src=particles, vel=vel, z=z)
    write_csv((csv_path + suffix + "_%03d.csv")%t, vel)

if verbose:
    print(dst_path)
    print(t_start)
    print(t_end)

pad_val = pre_config['pad_val']
use_mask = train_config['mask']
truncate = train_config['truncate']

loss_mode = train_config['loss']

particle_loss = keras.losses.mse

if loss_mode == 'hungarian_loss':
    from neuralparticles.tensorflow.losses.hungarian_loss import hungarian_loss
    particle_loss = hungarian_loss
elif loss_mode == 'emd_loss':
    particle_loss = emd_loss
elif loss_mode == 'chamfer_loss':
    particle_loss = chamfer_loss
else:
    print("No matching loss specified! Fallback to MSE loss.")

def mask_loss(y_true, y_pred):
   return particle_loss(y_true * zero_mask(y_true, pad_val), y_pred) if use_mask else particle_loss(y_true, y_pred)

dim = data_config['dim']
factor_d = math.pow(pre_config['factor'], 1/dim)
patch_size = pre_config['patch_size']
ref_patch_size = pre_config['patch_size_ref']
par_cnt = pre_config['par_cnt']
par_cnt_dst = pre_config['par_cnt_ref']

res = data_config['res']
low_res = int(res/factor_d)

half_ps = ref_patch_size//2
#border = int(math.ceil(half_ps-(patch_size//2*factor_2D)))

features = train_config['features'][1:]

if checkpoint > 0:
    model_path = data_path + "models/checkpoints/%s_%s_%04d.h5" % (data_config['prefix'], config['id'], checkpoint)
else:
    model_path = data_path + "models/%s_%s_trained.h5" % (data_config['prefix'], config['id'])

model = load_model(model_path, custom_objects={'mask_loss': mask_loss})

avg_chamfer_loss = 0
avg_emd_loss = 0

for t in range(t_start, t_end):
    (src_data, sdf_data, par_aux), (ref_data, ref_sdf_data) = get_data_pair(data_path, config_path, dataset, t, var) 

    patch_extractor = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pre_config['surf'], pre_config['stride'], aux_data=par_aux, features=features, pad_val=pad_val, bnd=data_config['bnd']/factor_d)

    write_out_particles(patch_extractor.positions, t, "_pos", [0,low_res], [0,low_res], 0.1, low_res//2 if dim == 3 else None)

    src_accum = np.empty((0,3))
    ref_accum = np.empty((0,3))
    res_accum = np.empty((0,3))
    
    src_patches = np.empty((0,par_cnt,3))
    ref_patches = np.empty((0,par_cnt_dst,3))
    res_patches = np.empty((0,par_cnt_dst,3))
    vel_patches = np.empty((0,par_cnt_dst,3))

    patch_pos = np.empty((0,3))

    avg_trunc = 0
    avg_ref_trunc = 0
    
    '''src = patch_extractor.get_patches()
    result = model.predict(x=src)
    patch_extractor.set_patches(result)'''

    
    while(True):
        src = patch_extractor.get_patch()
        if src is None:
            break

        ref = extract_particles(ref_data, patch_extractor.last_pos*factor_d, par_cnt_dst, ref_patch_size/2, pad_val)[0]

        if truncate:
            result, trunc = model.predict(x=src)
            trunc = int(trunc[0]*par_cnt_dst)
            raw_result = result[0]
            result = raw_result[:trunc]
            avg_trunc += trunc
            avg_ref_trunc += np.count_nonzero(ref != pad_val)/3
        else:
            result = model.predict(x=src)[0]
            raw_result = result
        
        patch_extractor.set_patch(result)

        src_accum = np.concatenate((src_accum, patch_extractor.transform_patch(src[0][0])))
        ref_accum = np.concatenate((ref_accum, patch_extractor.transform_patch(ref)*factor_d))
        res_accum = np.concatenate((res_accum, patch_extractor.transform_patch(result)*factor_d))

        src_patches = np.concatenate((src_patches, np.array([src[0][0]])))
        ref_patches = np.concatenate((ref_patches, np.array([ref])))
        res_patches = np.concatenate((res_patches, np.array([raw_result])))

        if 'v' in features:
            vel_patches = np.concatenate((vel_patches, np.array([src[1][features.index('v')]])))

        patch_pos = np.concatenate((patch_pos, np.array([patch_extractor.last_pos])))

    if truncate:
        print("Avg truncation position: %.1f" % (avg_trunc/len(src_patches)))
        print("Avg truncation position ref: %.1f" % (avg_ref_trunc/len(src_patches)))

    result = patch_extractor.data*factor_d
    result = result[np.where(np.all([np.all(4<=result,axis=-1),np.all(result<=res-4,axis=-1)],axis=0))]
    
    hdr = OrderedDict([ ('dim',len(result)),
                        ('dimX',res),
                        ('dimY',res),
                        ('dimZ',1 if dim == 2 else res),
                        ('elementType',0),
                        ('bytesPerElement',16),
                        ('info',b'\0'*256),
                        ('timestamp',(int)(time.time()*1e6))])
    writeParticlesUni(dst_path%t, hdr, result)

    writeNumpyRaw(npy_path + "_src_patches_%03d"%t, src_patches)
    writeNumpyRaw(npy_path + "_ref_patches_%03d"%t, ref_patches)
    writeNumpyRaw(npy_path + "_res_patches_%03d"%t, res_patches)
    writeNumpyRaw(npy_path + "_patch_pos_%03d"%t, patch_pos)

    write_out_particles(src_data, t, "_src", [0,low_res], [0,low_res], 0.1, low_res//2 if dim == 3 else None)
    write_out_particles(ref_data, t, "_ref", [0,res], [0,res], 0.1, low_res//2 if dim == 3 else None)
    write_out_particles(result, t, "_res", [0,res], [0,res], 0.1, low_res//2 if dim == 3 else None)
    if 'v' in features:
        writeNumpyRaw(npy_path + "_vel_patches_%03d"%t, vel_patches)
        write_out_vel(src_data, par_aux['v']/1000, t, "_vel", [0,low_res],[0,low_res], 0.1, low_res//2 if dim == 3 else None)

    min_cnt = min(len(ref_accum), len(res_accum))
    np.random.shuffle(ref_accum)
    np.random.shuffle(res_accum)
    ref_accum = K.constant(np.array([ref_accum[:min_cnt]]))
    res_accum = K.constant(np.array([res_accum[:min_cnt]]))

    print("particles: %d -> %d (fac: %.2f)" % (len(src_data), len(patch_extractor.data), (len(patch_extractor.data)/len(src_data))))

    '''call_f = lambda f,x,y: K.eval(f(x,y))[0]
    loss = call_f(chamfer_loss, ref_accum, res_accum)
    avg_chamfer_loss += loss
    print("global chamfer loss: %f" % loss)
    loss = call_f(emd_loss, ref_accum, res_accum)
    avg_emd_loss += loss
    print("global emd loss: %f" % loss)'''

print("avg chamfer loss: %f" % (avg_chamfer_loss/(t_end-t_start)))
print("avg emd loss: %f" % (avg_emd_loss/(t_end-t_start)))


