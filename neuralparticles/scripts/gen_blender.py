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
test_path = getParam("test", "")

surface_fac = float(getParam("surface", 0))

t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", -1))

dataset = int(getParam("dataset", -1))
var = int(getParam("var", 0))

gui = int(getParam("gui", 0))
pause = int(getParam("pause", 0))

real = int(getParam("real", 0)) != 0

patch_pos = np.fromstring(getParam("patch", ""),sep=",")

checkUnusedParams()

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

if dataset < 0:
    dataset = 0 if real else data_config['data_count']

if var < 0:
    var = pre_config['var']

if t_end < 0:
    t_end = data_config['frame_count']

if test_path != "":
    data_path += "result/%s_%s/" % (test_path[:-1], config['id'])
else:   
    data_path += ("result/%s_%s-%s_%s_d%03d_var%02d" + ("_real/" if real else "/")) % (data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], dataset, var)

if len(patch_pos) == 3:
    data_path += "patch_%d-%d-%d/" % (patch_pos[0],patch_pos[1],patch_pos[2])
    
prefix_surface = data_path + ("surface_%d/" % int(surface_fac*100) if surface_fac > 0 else "surface/")
prefix_foam = data_path + ("foam_%d/" % int((1-surface_fac)*100) if surface_fac > 0 else "foam/")

if not os.path.exists(prefix_surface):
    os.makedirs(prefix_surface)
if not os.path.exists(prefix_foam):
    os.makedirs(prefix_foam)
if not os.path.exists(prefix_surface + "source/"):
    os.makedirs(prefix_surface + "source/")
if not os.path.exists(prefix_surface + "reference/"):
    os.makedirs(prefix_surface + "reference/")
if not os.path.exists(prefix_surface + "result/"):
    os.makedirs(prefix_surface + "result/")
if not os.path.exists(prefix_foam + "source/"):
    os.makedirs(prefix_foam + "source/")
if not os.path.exists(prefix_foam + "reference/"):
    os.makedirs(prefix_foam + "reference/")
if not os.path.exists(prefix_foam + "result/"):
    os.makedirs(prefix_foam + "result/")

param = {}

dim = data_config['dim']
res = data_config['res']
bnd = data_config['bnd']
param['t_start'] = t_start
param['t_end'] = t_end
param['res'] = res
param['bnd'] = bnd
param['dim'] = dim
param['gui'] = gui
param['pause'] = pause
param['surface'] = surface_fac

if len(patch_pos) == 3:
    res = int(res * pre_config['patch_size_ref'])

param['in'] = data_path + "result_%03d.uni"
param['out_surface'] = prefix_surface + "result/fluidsurface_final_%04d.bobj.gz"
param['out_foam'] = prefix_foam + "result/fluidsurface_final_%04d.bobj.gz"
run_manta(manta_path, "scenes/blender.py", param, verbose)

param['in'] = data_path + "reference_%03d.uni"
param['out_surface'] = prefix_surface + "reference/fluidsurface_final_%04d.bobj.gz"
param['out_foam'] = prefix_foam + "reference/fluidsurface_final_%04d.bobj.gz"
run_manta(manta_path, "scenes/blender.py", param, verbose)

param['res'] = int(param['res'] / math.pow(pre_config['factor'],1/dim))
param['bnd'] = int(math.ceil(param['bnd'] / math.pow(pre_config['factor'],1/dim)))
if len(patch_pos) == 3:
    res = int(res * pre_config['patch_size'])
param['in'] = data_path + "source_%03d.uni"
param['out_surface'] = prefix_surface + "source/fluidsurface_final_%04d.bobj.gz"
param['out_foam'] = prefix_foam + "source/fluidsurface_final_%04d.bobj.gz"
run_manta(manta_path, "scenes/blender.py", param, verbose)
