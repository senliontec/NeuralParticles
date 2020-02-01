import glob
import os
import shutil
import numpy as np
import re
import json
from neuralparticles.tools.param_helpers import getParam, checkUnusedParams
from neuralparticles.tools.data_helpers import *
from neuralparticles.tools.uniio import writeNumpyRaw

data_path = getParam("data", "data/")
dest_path = getParam("dest", "dest/")
config_path = getParam("config", "config/version_00.txt")
real = getParam("real", 0) != 0
checkUnusedParams()

if not os.path.isdir(dest_path):
    os.mkdir(dest_path)

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())


sim_path = dest_path + ("sim_2%03d/" if real else "sim_1%03d/")

dens_path = sim_path+"density_%s_%04d"
sdf_path = sim_path+"levelset_%s_%04d"
vel_path = sim_path+"velocity_%s_%04d"

cnt = data_config['test_count'] if real else data_config['data_count']+data_config['test_count']
timesteps = data_config['frame_count']
res = data_config['res']
dim = data_config['dim']

if real:
    src_path = "%sreal/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"
else:
    ref_path = "%sreference/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"
    src_path = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var00_%03d"


def convert(path, res, postfix):    
    densities = np.empty((cnt, timesteps, res if dim == 3 else 1, res, res, 1))
    velocities = np.empty((cnt, timesteps, res if dim == 3 else 1, res, res, 3))
    sdfs = np.empty((cnt, timesteps, res if dim == 3 else 1, res, res, 1))

    for d in range(cnt):
        for t in range(timesteps):
            densities[d, t] = readNumpy(path%(d,t) + "_dens")
            velocities[d, t] = readNumpy(path%(d,t) + "_vel")
            sdfs[d, t] = readNumpy(path%(d,t) + "_sdf")

    print("Density")
    print("min: %.2f, max: %.2f, avg: %.2f" % (np.min(densities), np.max(densities), np.mean(densities)))
    print("Velocity")
    print("min: %.2f, max: %.2f, avg: %.2f" % (np.min(np.sum(velocities*velocities, axis=-1)), np.max(np.sum(velocities*velocities, axis=-1)), np.mean(np.sum(velocities*velocities, axis=-1))))
    print("SDF")
    print("min: %.2f, max: %.2f, avg: %.2f" % (np.min(sdfs), np.max(sdfs), np.mean(sdfs)))

    maxDens = np.max(densities)

    densities /= maxDens
    velocities /= data_config['fps']
    sdfs = np.tanh(sdfs * 5.0)

    print("normalized:")
    print("Density")
    print("min: %.2f, max: %.2f, avg: %.2f" % (np.min(densities), np.max(densities), np.mean(densities)))
    print("Velocity")
    print("min: %.2f, max: %.2f, avg: %.2f" % (np.min(np.sum(velocities*velocities, axis=-1)), np.max(np.sum(velocities*velocities, axis=-1)), np.mean(np.sum(velocities*velocities, axis=-1))))
    print("SDF")
    print("min: %.2f, max: %.2f, avg: %.2f" % (np.min(sdfs), np.max(sdfs), np.mean(sdfs)))

    for d in range(cnt):
        if not os.path.isdir(sim_path%d):
            os.mkdir(sim_path%d)
        for t in range(timesteps):
            writeNumpy(dens_path%(d,postfix,t), densities[d,t])
            writeNumpy(sdf_path%(d,postfix,t), sdfs[d,t])
            writeNumpy(vel_path%(d,postfix,t), velocities[d,t])

if not real:
    print("_____ high ______")
    convert(ref_path, res, "high")

print("_____ low ______")
convert(src_path, int(res/(pre_config['factor']**(1/dim))), "low")