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

t_start = int(getParam("t_start", -1, paramUsed))
t_end = int(getParam("t_end", -1, paramUsed))

dataset = int(getParam("dataset", 0, paramUsed))
var = int(getParam("var", 0, paramUsed))

data_type = getParam("type", "ref", paramUsed)

scr = getParam("scr", "", paramUsed)

checkUnusedParam(paramUsed)

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())


param = {}

res = data_config['res']
if data_type == "ref":
    data_path += "reference/%s_%s_d%03d" % (data_config['prefix'], data_config['id'], dataset)
    param['in'] = data_path + "_%03d_ps.uni"
    param['sdf'] = data_path + "_%03d_sdf.uni"
elif data_type == "src":
    data_path += "source/%s_%s-%s_d%03d_var%02d" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var)
    res = int(res/math.sqrt(pre_config['factor']))
    param['in'] = data_path + "_%03d_ps.uni"
    param['sdf'] = data_path + "_%03d_sdf.uni"
elif data_type == "res":
    data_path += "result/%s_%s-%s_d%03d_var%02d" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var)
    param['sdf'] = data_path + "_%03d_result.uni"
else:
    print("data type not supported!")
    exit()

if t_start < 0:
    t_start = train_config['t_start']
if t_end < 0:
    t_end = train_config['t_end']

param['t_start'] = t_start
param['t_end'] = t_end
param['res'] = res
param['scr'] = scr

run_manta(manta_path, "scenes/show_particles.py", param, verbose)