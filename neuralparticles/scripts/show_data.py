import os

import json

from neuralparticles.tools.shell_script import *
from neuralparticles.tools.param_helpers import *

import random
import math

data_path = getParam("data", "data/")
manta_path = getParam("manta", "neuralparticles/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0

src = getParam("src", "")

t_start = int(getParam("t_start", -1))
t_end = int(getParam("t_end", -1))

dataset = int(getParam("dataset", -1))
var = int(getParam("var", 0))

data_type = getParam("type", "ref")

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
    dataset = int(data_config['data_count']*train_config['train_split']) if data_type == 'res' else 0

dim = data_config['dim']
res = data_config['res']
if data_type == "ref":
    data_path += "reference/%s_%s_d%03d" % (data_config['prefix'], data_config['id'], dataset)
    param['in'] = data_path + "_%03d_ps.uni"
    param['sdf'] = data_path + "_%03d_sdf.uni"
elif data_type == "real":
    data_path += "real/%s_%s_d%03d" % (data_config['prefix'], data_config['id'], dataset)
    res = int(res/math.pow(pre_config['factor'], 1/dim))
    param['in'] = data_path + "_%03d_ps.uni"
    param['sdf'] = data_path + "_%03d_sdf.uni"
elif data_type == "src":
    data_path += "source/%s_%s-%s_d%03d_var%02d" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var)
    res = int(res/math.pow(pre_config['factor'], 1/dim))
    param['in'] = data_path + "_%03d_ps.uni"
    param['sdf'] = data_path + "_%03d_sdf.uni"
elif data_type == "res":
    data_path += "result/%s_%s-%s_d%03d_var%02d" % (data_config['prefix'], data_config['id'], pre_config['id'], dataset, var)
    param['in' if train_config['explicit'] else 'sdf'] = data_path + "_result_%03d.uni"
elif data_type == "res_real":
    data_path += "result/%s_%s_d%03d" % (data_config['prefix'], data_config['id'], dataset)
    param['in' if train_config['explicit'] else 'sdf'] = data_path + "_result_%03d.uni"
else:
    print("data type not supported!")
    exit()

if t_start < 0:
    t_start = min(train_config['t_start'], data_config['frame_count']-1)
if t_end < 0:
    t_end = min(train_config['t_end'], data_config['frame_count'])


param['t_start'] = t_start
param['t_end'] = t_end
param['res'] = res
param['scr'] = scr
param['dim'] = dim

run_manta(manta_path, "scenes/show_particles.py", param, verbose)