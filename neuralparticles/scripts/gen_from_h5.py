import json
from neuralparticles.tools.param_helpers import getParam, checkUnusedParams
from neuralparticles.tools.uniio import writeNumpyH5, readNumpyRaw
from glob import glob
from collections import OrderedDict
import time
from neuralparticles.tools.uniio import writeParticlesUni

from neuralparticles.tools.shell_script import *

import h5py

import numpy as np

import os

if __name__ == "__main__":
    data_path = getParam("data", "data/")
    manta_path = getParam("manta", "neuralparticles/")
    config_path = getParam("config", "config/version_00.txt")
    verbose = int(getParam("verbose", 0)) != 0
    gui = int(getParam("gui", 0))
    pause = int(getParam("pause", 0))
    checkUnusedParams()

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    ref_path = data_path + "reference/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    output_path = "%s%s_%s" % (ref_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"

    res = data_config['res']
    dim = data_config['dim']

    print(data_path + data_config["h5"])
    f = h5py.File(data_path + data_config["h5"])
    gt = f['poisson_4096'][:]
    
    min_pos = np.amin(gt[...,0:3], axis=1, keepdims=True)
    gt[...,0:3] = gt[...,0:3] - min_pos
    max_v = np.amax(gt[...,0:3], axis=(1,2),keepdims=True)
    gt[...,0:3] = gt[...,0:3] / max_v

    gt[...,0:3] = gt[...,0:3] * res

    for d in range(len(gt)):
        hdr = OrderedDict([ 
            ('dim',len(gt[d])),
            ('dimX',res),
            ('dimY',res),
            ('dimZ',1 if dim == 2 else res),
            ('elementType',0),
            ('bytesPerElement',16),
            ('info',b'\0'*256),
            ('timestamp',(int)(time.time()*1e6))])
        writeParticlesUni(output_path%(d,0) + "_ps.uni", hdr, gt[d,:,:3])

    param = {}

    param['gui'] = gui
    param['pause'] = pause

    param['in'] = output_path + "_ps.uni"
    param['out'] = output_path + "_sdf.uni"

    param['cnt'] = len(gt)
    param['t'] = 1

    param['dim'] = dim
    param['res'] = res

    run_manta(manta_path, "scenes/gen_levelset.py", dict(param), verbose) 