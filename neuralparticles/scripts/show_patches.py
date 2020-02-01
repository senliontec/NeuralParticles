import os

import json

from neuralparticles.tools.param_helpers import * 
from neuralparticles.tools.uniio import NPZBuffer

import random
import math

import matplotlib.pyplot as plt

data_path = getParam("data", "data/")
manta_path = getParam("manta", "neuralparticles/build/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0

timestep = int(getParam("timestep", -1))

dataset = int(getParam("dataset", 0))
var = int(getParam("var", 0))
pvar = int(getParam("pvar", 0))

l_scr = getParam("l_scr", "low_patch_%03d.uni")
h_scr = getParam("h_scr", "high_patch_%03d.uni")

checkUnusedParams()

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

filename = "%s_%s-%s_d%03d_var%02d_pvar%02d_%03d"%(data_config['prefix'], data_config['id'], pre_config['id'], dataset, var, pvar, timestep)

patch_buffer = NPZBuffer(data_path + "patches/source/" + filename + "_ps")

i = 0
while True:
    patch = patch_buffer.next()
    if patch is None:
        break
    plt.scatter(patch[:,0],patch[:,1])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.savefig(l_scr%i)
    plt.clf()
    i+=1


patch_buffer = NPZBuffer(data_path + "patches/reference/" + filename + "_ps")

i = 0
while True:
    patch = patch_buffer.next()
    if patch is None:
        break
    
    plt.scatter(patch[:,0],patch[:,1])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.savefig(h_scr%i)
    plt.clf()
    i+=1
