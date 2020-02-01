import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import *
from neuralparticles.tools.shell_script import *
from neuralparticles.tools.uniio import writeNumpyRaw
from neuralparticles.tools.particle_grid import RandomParticles, ParticleGrid
import numpy as np

import os

import random

import math

if __name__ == "__main__":
    data_path = getParam("data", "data/")
    manta_path = getParam("manta", "neuralparticles/build/")
    config_path = getParam("config", "config/version_00.txt")
    gui = int(getParam("gui", 0))
    verbose = int(getParam("verbose", 0)) != 0
    pause = int(getParam("pause", 0))
    checkUnusedParams()

    ref_path = data_path + "reference/"
    if not os.path.exists(ref_path):
        os.makedirs(ref_path)

    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    param = {}

    #disable gui
    param['gui'] = gui
    param['pause'] = pause

    param['sres'] = data_config['sub_res']

    param['bnd'] = data_config['bnd']

    # write only every 30th frame -> 30 frames are one timestep
    param['fps'] = 1

    # simulation time (how many frames)
    param['t_end'] = 0

    param['res'] = data_config['res']

    param['dim'] = data_config['dim']

    # run random training setups
    random.seed(data_config['seed'])

    if data_config['trans_mode'] != "":
        output_path = "%s%s_%s" % (ref_path, data_config['prefix'], data_config['id']) + "_d%03d_id"
        trans_path = "%s%s_%s" % (ref_path, data_config['prefix'], data_config['id']) + "_d%03d"
    else:
        output_path = "%s%s_%s" % (ref_path, data_config['prefix'], data_config['id']) + "_d%03d"

    print(output_path)

    def run_gen(cubes, spheres, cnt):
        param['c_cnt'] = len(cubes)
        param['s_cnt'] = len(spheres)
        
        param['out'] = output_path % cnt + "_%03d"
        run_manta(manta_path, "scenes/2D_sph.py", dict(param, **cubes, **spheres), verbose) 
    
    for i in range(data_config['data_count']):
        cubes = {}
        spheres = {}
        for c in range(random.randint(1,data_config['max_cnt'])):    
            if random.random() < data_config['cube_prob']:
                scx = random.uniform(data_config['min_scale'], data_config['max_scale'])
                scy = random.uniform(data_config['min_scale'], data_config['max_scale'])
                px = random.uniform(data_config['min_pos']+scx/2, data_config['max_pos']-scx/2)
                py = random.uniform(data_config['min_pos']+scy/2, data_config['max_pos']-scy/2)
                if param['dim'] == 3:
                    scz = random.uniform(data_config['min_scale'], data_config['max_scale'])
                    pz = random.uniform(data_config['min_pos']+scz/2, data_config['max_pos']-scz/2)
                    cubes['c%d'%len(cubes)] = "%f,%f,%f,%f,%f,%f"%(px,py,pz,scx,scy,scz)
                else:
                    cubes['c%d'%len(cubes)] = "%f,%f,%f,%f"%(px,py,scx,scy)
            else:
                rad = random.uniform(data_config['min_scale'], data_config['max_scale'])
                px = random.uniform(data_config['min_pos']+rad/2, data_config['max_pos']-rad/2)
                py = random.uniform(data_config['min_pos']+rad/2, data_config['max_pos']-rad/2)
                if param['dim'] == 3:
                    pz = random.uniform(data_config['min_pos']+rad/2, data_config['max_pos']-rad/2)
                    spheres['s%d'%len(spheres)] = "%f,%f,%f,%f"%(px,py,pz,rad)
                else:
                    spheres['s%d'%len(spheres)] = "%f,%f,%f"%(px,py,rad)

        run_gen(cubes, spheres, i)

    if data_config['trans_mode'] != "":
        param = {}
        param['gui'] = gui
        param['pause'] = pause
        param['dim'] = data_config['dim']
        param['res'] = data_config['res']
        param['curv'] = int(data_config['use_curv'])
        param['peaks'] = data_config['peaks']
        param['t'] = 1
        param['mode'] = data_config['trans_mode']
        param['fac'] = data_config['disp_factor']
        param['bnd'] = data_config['bnd']

        for i in range(data_config['data_count']):
            param['in'] = output_path % i + "_%03d"
            param['out'] = trans_path % i + "_%03d"
            run_manta(manta_path, "scenes/transform_particles.py", param)
