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

surface_fac = float(getParam("surface", 0))

t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", -1))

dataset = int(getParam("dataset", -1))
var = int(getParam("var", 0))

gui = int(getParam("gui", 0))
pause = int(getParam("pause", 0))

checkUnusedParams()

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

data_path += "result/%s_%s-%s_%s_d%03d_var%02d/" % (data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], dataset, var)

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

if dataset < 0:
    dataset = data_config['data_count']

if t_end < 0:
    t_end = data_config['frame_count']

dim = data_config['dim']
res = data_config['res']
param['t_start'] = t_start
param['t_end'] = t_end
param['res'] = res
param['dim'] = dim
param['gui'] = gui
param['pause'] = pause
param['surface'] = surface_fac

param['in'] = data_path + "result_%03d.uni"
param['out_surface'] = prefix_surface + "result/fluidsurface_final_%04d.bobj.gz"
param['out_foam'] = prefix_foam + "result/fluidsurface_final_%04d.bobj.gz"
run_manta(manta_path, "scenes/blender.py", param, verbose)

param['in'] = data_path + "reference_%03d.uni"
param['out_surface'] = prefix_surface + "reference/fluidsurface_final_%04d.bobj.gz"
param['out_foam'] = prefix_foam + "reference/fluidsurface_final_%04d.bobj.gz"
run_manta(manta_path, "scenes/blender.py", param, verbose)

param['res'] = int(res / math.pow(pre_config['factor'],1/dim))
param['in'] = data_path + "source_%03d.uni"
param['out_surface'] = prefix_surface + "source/fluidsurface_final_%04d.bobj.gz"
param['out_foam'] = prefix_foam + "source/fluidsurface_final_%04d.bobj.gz"
run_manta(manta_path, "scenes/blender.py", param, verbose)
