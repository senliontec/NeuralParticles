import numpy as np
from glob import glob
import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import particle_radius
from neuralparticles.tools.shell_script import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw, readNumpy, writeNumpyOBJ, readParticlesUni
import random
import math
from collections import OrderedDict
import time

point_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")
res = int(getParam("res", -1))

checkUnusedParams()

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

sub_res = data_config['sub_res']
dim = data_config['dim']

res /= pre_config['factor'] ** (1/dim)

bnd = data_config['bnd']

random.seed(data_config['seed'])
np.random.seed(data_config['seed'])

real_path = "%sreal/%s_%s_" % (point_path, data_config['prefix'], data_config['id']) + "%03d"

if not os.path.exists(point_path + "real/"):
    os.makedirs(point_path + "real/")

samples = glob(point_path + "*.npz")
samples.sort()

points = []
min_v = np.array([1000,1000,1000])
max_v = np.array([-1000,-1000,-1000])
for i,item in enumerate(samples):
    d = readNumpy(item[:-4])
    points.append(d)
    min_v = np.min((min_v, np.min(d, axis=0)), axis=0)
    max_v = np.max((max_v, np.max(d, axis=0)), axis=0)
    if res < 0:
        res = d.shape[0]**(1/dim)/sub_res + 2 * bnd
print("Resolution: %d" % res)
scale = max_v - min_v

for i in range(len(points)):
    points[i] -= min_v + [0.5,0,0.5] * scale 
    points[i] *= (res - 4 * bnd) / np.max(scale)
    points[i] += [res/2, bnd*2, res/2]

    writeNumpyRaw(real_path%i, points[i])
    writeNumpyOBJ(real_path%i +".obj", points[i])