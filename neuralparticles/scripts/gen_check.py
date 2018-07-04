import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import *
from neuralparticles.tools.uniio import writeNumpyRaw
from neuralparticles.tools.particle_grid import RandomParticles, ParticleGrid
import numpy as np

import os

import random

import math

if __name__ == "__main__":
    data_path = getParam("data", "data/")
    config_path = getParam("config", "config/version_00.txt")
    skip_gen = int(getParam("skip_gen", 0)) != 0
    checkUnusedParams()

    src_path = data_path + "source/"
    if not os.path.exists(src_path):
        os.makedirs(src_path)

    ref_path = data_path + "reference/"
    if not os.path.exists(ref_path):
        os.makedirs(ref_path)

    if not os.path.exists(data_path + "patches/"):
        os.makedirs(data_path + "patches/")

    patch_src_path = data_path + "patches/source/"
    if not os.path.exists(patch_src_path):
        os.makedirs(patch_src_path)

    patch_ref_path = data_path + "patches/reference/"
    if not os.path.exists(patch_ref_path):
        os.makedirs(patch_ref_path)

    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    data_cnt = data_config['data_count']
    frame_cnt = data_config['frame_count']
    res = data_config['res']
    fac = pre_config['factor']
    dim = data_config['dim']
    fac_d = math.pow(fac, 1/dim)
    low_res = int(res/fac_d)
    sres = data_config['sub_res']
    random.seed(data_config['seed'])
    np.random.seed(data_config['seed'])

    features = []#['v','d','p']


    ref_path = "%s%s_%s" % (ref_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"
    src_path = "%s%s_%s-%s" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
    patch_rot_src_path = "%s%s_%s-%s_rot_ps" % (patch_src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"
    patch_src_path = "%s%s_%s-%s_p" % (patch_src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"
    patch_rot_dst_path = "%s%s_%s-%s_rot_ps" % (patch_ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"
    patch_dst_path = "%s%s_%s-%s_ps" % (patch_ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"

    src_gen = RandomParticles(low_res, low_res, low_res if dim == 3 else 1, sres, fac_d, low_res * 0.3, 5, 0.5 if data_config['cos_displace'] == 0 else 0, data_config['cos_displace'])
    for d in range(data_cnt):
        for t in range(frame_cnt):
            print("dataset: %d/%d" % (d+1,data_cnt))
            print("timestep: %d/%d" % (t+1,frame_cnt))
            if not skip_gen:
                src_gen.gen_random()
                src_grid, ref_grid = src_gen.get_grid()
                writeNumpyRaw(ref_path % (d, t) + "_ps", ref_grid.particles)
                writeNumpyRaw(ref_path % (d, t) + "_sdf", ref_grid.cells)
                writeNumpyRaw(src_path % (d,0,t) + "_ps", src_grid.particles)
                writeNumpyRaw(src_path % (d,0,t) + "_sdf", src_grid.cells)
                for v in range(1,pre_config['var']):
                    src_grid = src_gen.get_grid()[0]
                    writeNumpyRaw(src_path % (d,v,t) + "_ps", src_grid.particles)
                    writeNumpyRaw(src_path % (d,v,t) + "_sdf", src_grid.cells)

            src, dst, rotated_src, rotated_dst, positions = gen_patches(data_path, config_path, d_start=d, d_stop=d+1, t_start=t, t_stop=t+1, features=features)
            writeNumpyRaw(patch_src_path % ('s',d,t), src[0])
            i = 0
            for f in features:
                if f == 'v':
                    writeNumpyRaw(src_path % (f,d,t), src[1][:,:,i:i+3])
                    i+=3
                else:
                    writeNumpyRaw(src_path % (f,d,t), src[1][:,:,i:i+1])
                    i+=1
            writeNumpyRaw(patch_rot_src_path % (d,t), rotated_src)
            writeNumpyRaw(patch_dst_path % (d,t), dst)
            writeNumpyRaw(patch_rot_dst_path % (d,t), rotated_dst)