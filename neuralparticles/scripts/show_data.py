import os

import json

from neuralparticles.tools.shell_script import *
from neuralparticles.tools.param_helpers import *

import random
import math
import numpy as np

data_path = getParam("data", "data/")
manta_path = getParam("manta", "neuralparticles/build/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0

t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", -1))

dataset = int(getParam("dataset", -1))
var = int(getParam("var", 0))

show_results = int(getParam("res", 1)) != 0
real = int(getParam("real", 0)) != 0

patch_pos = np.fromstring(getParam("patch", ""),sep=",")

if len(patch_pos) == 2:
    patch_pos = np.append(patch_pos, [0.5])

show_temp_coh = int(getParam("temp_coh", 0)) != 0

scr = getParam("scr", "")

checkUnusedParams()

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

param = {}

if dataset < 0:
    dataset = data_config['data_count'] if show_results and not real else 0

dim = data_config['dim']
res = data_config['res']

if show_results:
    data_path += ("result/%s_%s-%s_%s_d%03d_var%02d" + ("_real/" if real else "/")) % (data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], dataset, var)
    if len(patch_pos) == 3:
        data_path += "patch_%d-%d-%d/" % (patch_pos[0],patch_pos[1],patch_pos[2])
        res = int(res * pre_config['patch_size'])
        param['bnd'] = 0

    param['in'] = data_path + "result_%03d.uni"
    if not show_temp_coh:
        param['src'] = data_path + "source_%03d.uni"
        if not real: param['ref'] = data_path + "reference_%03d.uni"
        
        #param['src_sdf'] = data_path + "source/%s_%s-%s_d%03d_var%02d" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var) + "_%03d_sdf.uni"
        #param['ref_sdf'] = data_path + "reference/%s_%s_d%03d" % (data_config['prefix'], data_config['id'], dataset) + "_%03d_sdf.uni"
else:
    param['src'] = data_path + "source/%s_%s-%s_d%03d_var%02d" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var) + "_%03d_ps.uni"
    param['in'] = data_path + "reference/%s_%s_d%03d" % (data_config['prefix'], data_config['id'], dataset) + "_%03d_ps.uni"
    
    param['src_sdf'] = data_path + "source/%s_%s-%s_d%03d_var%02d" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var) + "_%03d_sdf.uni"
    param['sdf'] = data_path + "reference/%s_%s_d%03d" % (data_config['prefix'], data_config['id'], dataset) + "_%03d_sdf.uni"

if t_end < 0:
    t_end = data_config['frame_count']

param['t_start'] = t_start
param['t_end'] = t_end
param['res'] = res
param['scr'] = scr
param['dim'] = dim
param['fac'] = math.pow(pre_config['factor'],1/dim)

run_manta(manta_path, "scenes/show_particles.py", param, verbose)