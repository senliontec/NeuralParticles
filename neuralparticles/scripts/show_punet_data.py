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

dataset = int(getParam("dataset", 0))

show_results = int(getParam("res", 1)) != 0

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

dim = data_config['dim']
res = data_config['res']

if show_results:
    param['in'] = data_path + "result/%s_%s-%s_%s_d%03d_var%02d/result" % (data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], dataset, 0) + "_%03d.uni"
    if not show_temp_coh:
        param['src'] = data_path + "result/%s_%s-%s_%s_d%03d_var%02d/source" % (data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], dataset, 0) + "_%03d.uni"
        param['ref'] = data_path + "result/%s_%s-%s_%s_d%03d_var%02d/reference" % (data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], dataset, 0) + "_%03d.uni"
else:
    pass

param['t_start'] = 0
param['t_end'] = 1
param['res'] = res
param['scr'] = scr
param['dim'] = dim
param['fac'] = math.pow(pre_config['factor'],1/dim)

run_manta(manta_path, "scenes/show_particles.py", param, verbose)