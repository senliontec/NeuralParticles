import sys, os
sys.path.append("manta/scenes/tools")

import json
from shell_script import *
from helpers import *

import random

paramUsed = []

data_path = getParam("data", "data/", paramUsed)
manta_path = getParam("manta", "manta/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0
gui = int(getParam("gui", 0, paramUsed))

checkUnusedParam(paramUsed)

src_path = data_path + "reference/"

dst_path = data_path + "source/"
if not os.path.exists(dst_path):
	os.makedirs(dst_path)

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

param = {}

param['res'] = data_config['res']
param['sres'] = data_config['sub_res']
 
param['factor'] = pre_config['factor']
param['gui'] = gui
param['t'] = data_config['frame_count']
param['min_n'] = pre_config['min_n']

random.seed(data_config['seed'])

src_path = "%s%s_%s" % (src_path, data_config['prefix'], data_config['id']) + "_d%03d"
output_path = "%s%s_%s-%s" % (dst_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d"
print(src_path)
print(output_path)

for i in range(train_config['train_data_count']+train_config['test_data_count']):
    for j in range(pre_config['var']):
        param['seed'] = random.randint(0,45820438204)
        param['in'] = src_path%(i) + "_%03d"
        param['out'] = output_path%(i,j) + "_%03d"
        run_manta(manta_path, "scenes/down_scale.py", param, verbose)