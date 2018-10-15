import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import *
from neuralparticles.tools.uniio import writeNumpyRaw
import numpy as np

import os

import random

import math

if __name__ == "__main__":
    data_path = getParam("data", "data/")
    config_path = getParam("config", "config/version_00.txt")
    checkUnusedParams()

    if not os.path.exists(data_path + "patches/"):
        os.makedirs(data_path + "patches/")

    src_path = data_path + "patches/source/"
    if not os.path.exists(src_path):
        os.makedirs(src_path)

    dst_path = data_path + "patches/reference/"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    data_cnt = data_config['data_count']
    test_cnt = data_config['test_count']
    frame_cnt = data_config['frame_count']
    features = ['v','d','p']

    src_path = "%s%s_%s-%s_p" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"
    dst_path = "%s%s_%s-%s_ps" % (dst_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"

    for d in range(data_cnt+test_cnt):
        print("extract patches of dataset %d" % d)
        for t in range(frame_cnt):
            src, dst, positions = gen_patches(data_path, config_path, d_start=d, d_stop=d+1, t_start=t, t_stop=t+1)
            writeNumpyRaw(src_path % ('s',d,t), src[0])
            i = 0
            for f in features:
                if f == 'v':
                    writeNumpyRaw(src_path % (f,d,t), src[1][:,:,i:i+3])
                    i+=3
                else:
                    writeNumpyRaw(src_path % (f,d,t), src[1][:,:,i:i+1])
                    i+=1
            writeNumpyRaw(dst_path % (d,t), dst)
