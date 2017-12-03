import sys, os
sys.path.append("manta/scenes/tools")

import json
from shell_script import *
from helpers import *

import math

paramUsed = []

data_path = getParam("data", "data/", paramUsed)
manta_path = getParam("manta", "manta/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0

timestep = int(getParam("timestep", -1, paramUsed))

dataset = int(getParam("dataset", 0, paramUsed))
var = int(getParam("var", 0, paramUsed))

l_scr = getParam("l_scr", "", paramUsed)
h_scr = getParam("h_scr", "", paramUsed)

checkUnusedParam(paramUsed)

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

if timestep < 0:
    timestep = train_config['t_end']-1

param = {}

param['src'] = data_path + "patches/source/%s_%s-%s_d%03d_var%02d_%03d_sdf"%(data_config['prefix'], data_config['id'], pre_config['id'], dataset, var, timestep)
param['vel'] = data_path + "patches/source/%s_%s-%s_d%03d_var%02d_%03d_vel"%(data_config['prefix'], data_config['id'], pre_config['id'], dataset, var, timestep)
param['ps'] =  data_path + "patches/source/%s_%s-%s_d%03d_var%02d_%03d_ps"%(data_config['prefix'], data_config['id'], pre_config['id'], dataset, var, timestep)
param['ref'] = data_path + "patches/reference/%s_%s-%s_d%03d_var%02d_%03d_sdf"%(data_config['prefix'], data_config['id'], pre_config['id'], dataset, var, timestep)
param['psize'] = pre_config['patch_size']
param['hpsize'] = int(pre_config['patch_size']*math.sqrt(pre_config['factor']))
param['t'] = 1
param['l_scr'] = l_scr
param['h_scr'] = h_scr

#param['scr'] = create_curr_date_folder(data_loc+'screenshots/') + "sph_patch_%03d_sdf_ref.png"

run_manta(manta_path, "scenes/show_patches.py", param, verbose)