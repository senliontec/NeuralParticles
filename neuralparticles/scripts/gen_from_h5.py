import json
from neuralparticles.tools.param_helpers import getParam, checkUnusedParams
from neuralparticles.tools.data_helpers import *
from neuralparticles.tools.uniio import writeNumpyRaw

import h5py

import numpy as np

import os


def nonuniform_sampling(num = 4096, sample_num = 1024):
    sample = set()
    loc = np.random.rand()*0.8+0.1
    while(len(sample)<sample_num):
        a = int(np.random.normal(loc=loc,scale=0.3)*num)
        if a<0 or a>=num:
            continue
        sample.add(a)
    return list(sample)

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
    fac = pre_config['factor']
    
    src_path = "%s%s_%s-%s_p" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"
    dst_path = "%s%s_%s-%s_p" % (dst_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"

    f = h5py.File(data_path + data_config["train"])
    gt = f['poisson_4096'][:]
    center = np.mean(gt[...,:3], axis=1, keepdims=True)
    gt[...,0:3] = gt[...,0:3] - center
    radius = np.amax(np.sqrt(np.sum(gt[...,:3] ** 2, axis=-1)),axis=1,keepdims=True)
    gt[...,0:3] = gt[...,0:3] / np.expand_dims(radius,axis=-1)

    np.random.seed(10)
    for d in range(data_cnt+test_cnt):
        for t in range(frame_cnt):
            src = gt[d:d+1,nonuniform_sampling(gt.shape[1], int(gt.shape[1]/fac))]
            writeNumpyRaw(src_path % ('s',d,t), src[...,:3])
            writeNumpyRaw(src_path % ('n',d,t), src[...,3:6])

            writeNumpyRaw(dst_path % ('s',d,t), gt[d:d+1,:,:3])
            writeNumpyRaw(dst_path % ('n',d,t), gt[d:d+1,:,3:6])