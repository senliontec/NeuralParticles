import sys, os
sys.path.append("manta/scenes/tools")

import json
from shell_script import *
from helpers import *

import random
import math

paramUsed = []

data_path = getParam("data", "data/", paramUsed)
manta_path = getParam("manta", "manta/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0

checkUnusedParam(paramUsed)

src_path_src = data_path + "source/"
src_path_ref = data_path + "reference/"

dst_path = data_path + "patches/"
if not os.path.exists(dst_path):
	os.makedirs(dst_path)

dst_path_src = dst_path + "source/"
if not os.path.exists(dst_path_src):
	os.makedirs(dst_path_src)

dst_path_ref = dst_path + "reference/"
if not os.path.exists(dst_path_ref):
	os.makedirs(dst_path_ref)

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

param = {}

param["t"] = data_config['frame_count']

# patch size
param["psize"] = pre_config['patch_size']
param["stride"] = pre_config['stride']

param["hpsize"] = int(pre_config['patch_size']*math.sqrt(pre_config['factor']))

param["l_fac"] = pre_config['l_fac']
param["h_fac"] = pre_config['h_fac']
param["tanh"] = pre_config['use_tanh']

param['par_cnt'] = pre_config['par_cnt']

# tolerance of surface
param["surface"] = pre_config['surf']

src_path_src = "%s%s_%s-%s" % (src_path_src, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d"
src_path_ref = "%s%s_%s" % (src_path_ref, data_config['prefix'], data_config['id']) + "_d%03d"
print(src_path_src)
print(src_path_ref)

dst_path_src = "%s%s_%s-%s" % (dst_path_src, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d"
dst_path_ref = "%s%s_%s-%s" % (dst_path_ref, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d"
print(dst_path_src)
print(dst_path_ref)

for i in range(train_config['train_data_count'] + train_config['test_data_count']):
    for j in range(pre_config['var']):
        param["h_in"] = src_path_ref%(i) + "_%03d"
        param["l_in"] = src_path_src%(i,j) + "_%03d"
        param["h_out"] = dst_path_ref%(i,j) + "_%03d"
        param["l_out"] = dst_path_src%(i,j) + "_%03d"
        run_manta(manta_path, "scenes/extract_patches.py", param, verbose)