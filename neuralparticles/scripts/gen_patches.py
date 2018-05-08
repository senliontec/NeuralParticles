import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import *
from neuralparticles.tools.uniio import writeNumpyBuf, finalizeNumpyBufs
import numpy as np

import os

import random

import math

if __name__ == "__main__":
    data_path = getParam("data", "data/")
    config_path = getParam("config", "config/version_00.txt")
    checkUnusedParams()

    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
        train_config = json.loads(f.read())

    t_start = train_config['t_start']
    t_end = train_config['t_end']
    var = pre_config['var']
    repetitions = pre_config['par_var']
    data_cnt = int(data_config['data_count'] * train_config['train_split'])

    features = train_config['features'][1:]

    if not os.path.exists(data_path + "patches/"):
        os.makedirs(data_path + "patches/")

    src_path = data_path + "patches/source/"
    if not os.path.exists(src_path):
        os.makedirs(src_path)

    dst_path = data_path + "patches/reference/"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    src, dst, rotated_src, rotated_dst, positions = gen_patches(data_path, config_path, data_cnt, t_end, var, repetitions, t_start=t_start)

    path = "%s%s_%s-%s_p" % (src_path, data_config['prefix'], data_config['id'], pre_config['id'])
    print(path)
    for s in src[0]:
        writeNumpyBuf(path + "s", s)

    for i in range(len(features)):
        for s in src[i+1]:
            writeNumpyBuf(path + features[i], s)
    
    path = "%s%s_%s-%s_p" % (dst_path, data_config['prefix'], data_config['id'], pre_config['id'])
    print(path)
    for d in dst:
        writeNumpyBuf(path + "s", d)
    
    path = "%s%s_%s-%s_rot_p" % (src_path, data_config['prefix'], data_config['id'], pre_config['id'])
    print(path)
    for s in rotated_src:
        writeNumpyBuf(path + "s", s)

    path = "%s%s_%s-%s_rot_p" % (dst_path, data_config['prefix'], data_config['id'], pre_config['id'])
    print(path)
    for d in rotated_dst:
        writeNumpyBuf(path + "s", d)

    finalizeNumpyBufs()
