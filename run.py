import sys, os
sys.path.append("manta/scenes/tools")

import keras
from keras.models import Model, load_model

from subpixel import *

import json
from helpers import *
from uniio import *

import scipy.ndimage.filters as fi
import math
import numpy as np

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
    dst_path = data_path + "result/%s_%s-%s_d%03d_var%02d" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var) + "_%03d_result.uni"
if t_start < 0:
    t_start = train_config['t_start']
if t_end < 0:
    t_end = train_config['t_end']

if verbose:
    print(src_path)
    print(dst_path)
    print(t_start)
    print(t_end)

factor_2D = math.sqrt(pre_config['factor'])
patch_size = pre_config['patch_size']
high_patch_size = int(patch_size*factor_2D)

res = data_config['res']
low_res = int(res/factor_2D)

result = np.ndarray(shape=(1,res,res,1), dtype=float)

half_ps = high_patch_size//2
border = int(math.ceil(half_ps-(patch_size//2*factor_2D)))

result=np.pad(result,((0,0),(border,border),(border,border),(0,0)),mode="edge")

elem_min = np.vectorize(lambda x,y: min(x,y))
circular_filter = np.reshape(filter2D(high_patch_size, high_patch_size*0.2, 500), (high_patch_size, high_patch_size,1))

model = load_model(data_path + "models/%s_%s_trained.h5" % (data_config['prefix'], config['id']), custom_objects={'Subpixel': Subpixel})

for t in range(t_start, t_end):
    result.fill(1)
    hdr, source = readUni(src_path%t+"_sdf.uni")
    aux = []
    for f in train_config['features']:
        if f != "sdf":
            aux.append(readUni(src_path%t+"_"+f+".uni")[1])
    
    if len(aux) > 0:
        aux = np.concatenate(aux, axis=3)
    
    # pre_param.stride instead of "1"!
    patch_pos = get_patches(source, patch_size, low_res, low_res, 1, pre_config['surf'])

    for pos in patch_pos:
        data = [np.array([extract_patch(source, pos, patch_size) * pre_config['l_fac']])]

        if pre_config['use_tanh'] != 0:
            data[0] = np.tanh(data[0])
        if len(aux) > 0:
            data.append(np.array([extract_patch(aux, pos, patch_size)]))

        predict = model.predict(x=data, batch_size=1)

        if pre_config['use_tanh'] != 0:
            predict = np.arctanh(np.clip(predict,-.999999,.999999))

        predict = predict * circular_filter/pre_config['h_fac']
        
        insert_patch(result, predict[0], (factor_2D*pos).astype(int)+border, elem_min)

    hdr['dimX'] = res
    hdr['dimY'] = res

    writeUni(dst_path%t, hdr, result[0,border:res+border,border:res+border,0])

