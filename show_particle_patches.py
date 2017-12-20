import sys, os
sys.path.append("manta/scenes/tools")

import json
from shell_script import *
from helpers import *
from uniio import *

import random
import math

import matplotlib.pyplot as plt
paramUsed = []

data_path = getParam("data", "data/", paramUsed)
manta_path = getParam("manta", "manta/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0

timestep = int(getParam("timestep", -1, paramUsed))

dataset = int(getParam("dataset", 0, paramUsed))
var = int(getParam("var", 0, paramUsed))

l_scr = getParam("l_scr", "low_patch_%03d.uni", paramUsed)
h_scr = getParam("h_scr", "high_patch_%03d.uni", paramUsed)

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

patch_size = 5
stride = 1
surface = 0.5
particle_cnt = 50

filename = "%s_%s-%s_d%03d_var%02d_%03d"%(data_config['prefix'], data_config['id'], pre_config['id'], dataset, var, timestep)

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
