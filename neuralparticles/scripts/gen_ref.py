from neuralparticles.tools.param_helpers import *
import json
from neuralparticles.tools.shell_script import *
import math

import random

data_path = getParam("data", "data/")
manta_path = getParam("manta", "neuralparticles/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0
gui = int(getParam("gui", 0))
pause = int(getParam("pause", 0))

lowres = int(getParam("lowres", 0)) != 0

checkUnusedParams()

if not os.path.exists(data_path):
	os.makedirs(data_path)

data_path += "real/" if lowres else "reference/"
if not os.path.exists(data_path):
	os.makedirs(data_path)

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

if verbose:
    print("Config Loaded:")
    print(config)
    print(data_config)

param = {}

#disable gui
param['gui'] = gui
param['pause'] = pause

param['sres'] = data_config['sub_res']

param['bnd'] = data_config['bnd']

# write only every 30th frame -> 30 frames are one timestep
param['fps'] = data_config['fps']

# simulation time (how many frames)
param['t'] = data_config['frame_count']

param['res'] = int(data_config['res']/math.pow(pre_config['factor'], 1/data_config['dim'])) if lowres else data_config['res']

param['dim'] = int(data_config['dim'])

# run random training setups
random.seed(data_config['seed'])

if 'transform' in data_config:
    output_path = "%s%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_id"
    trans_path = "%s%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d"
else:
    output_path = "%s%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d"

print(output_path)

def run_gen(cubes,spheres,cnt):
    param['c_cnt'] = len(cubes)
    param['s_cnt'] = len(spheres)
    
    param['out'] = output_path % cnt + "_%03d"
    run_manta(manta_path, "scenes/2D_sph.py", dict(param, **cubes, **spheres), verbose) 
    
data_cnt = data_config['data_count']
modes = data_config['modes']

m_idx = -1
n_idx = -1
for i in range(data_cnt):
    if i > n_idx:
        m_idx += 1
        n_idx = i + modes[m_idx]['prop'] * data_cnt
        
        param['circ'] = modes[m_idx]['circ_vel']
        param['wlt'] = modes[m_idx]['wlt_vel']
        param['grav'] = modes[m_idx]['grav']

    cubes = {}
    spheres = {}
    for c in range(random.randint(modes[m_idx]['cnt'][0],modes[m_idx]['cnt'][1])):    
        if random.random() < modes[m_idx]['cube_prob']:
            scx = random.uniform(modes[m_idx]['scale_x'][0], modes[m_idx]['scale_x'][1])
            scy = random.uniform(modes[m_idx]['scale_y'][0], modes[m_idx]['scale_y'][1])
            px = random.uniform(modes[m_idx]['pos_x'][0]+scx/2, modes[m_idx]['pos_x'][1]-scx/2)
            py = random.uniform(modes[m_idx]['pos_y'][0]+scy/2, modes[m_idx]['pos_y'][1]-scy/2)
            if param['dim'] == 3:
                scz = random.uniform(modes[m_idx]['scale_z'][0], modes[m_idx]['scale_z'][1])
                pz = random.uniform(modes[m_idx]['pos_z'][0]+scz/2, modes[m_idx]['pos_z'][1]-scz/2)
                cubes['c%d'%len(cubes)] = "%f,%f,%f,%f,%f,%f"%(px,py,pz,scx,scy,scz)
            else:
                cubes['c%d'%len(cubes)] = "%f,%f,%f,%f"%(px,py,scx,scy)
        else:
            rad = random.uniform(modes[m_idx]['rad'][0], modes[m_idx]['rad'][1])
            px = random.uniform(modes[m_idx]['pos_x'][0]+rad/2, modes[m_idx]['pos_x'][1]-rad/2)
            py = random.uniform(modes[m_idx]['pos_y'][0]+rad/2, modes[m_idx]['pos_y'][1]-rad/2)
            if param['dim'] == 3:
                pz = random.uniform(modes[m_idx]['pos_z'][0]+rad/2, modes[m_idx]['pos_z'][1]-rad/2)
                spheres['s%d'%len(spheres)] = "%f,%f,%f,%f"%(px,py,pz,rad)
            else:
                spheres['s%d'%len(spheres)] = "%f,%f,%f"%(px,py,rad)

    run_gen(cubes, spheres, i)
    param['seed'] = random.randint(0,1000000000)

if "transform" in data_config:
    trans_config = data_config["transform"]
    param = {}
    param['gui'] = gui
    param['pause'] = pause
    param['dim'] = data_config['dim']
    param['res'] = data_config['res']
    param['bnd'] = data_config['bnd']
    param['t'] = data_config['frame_count']

    if "wavelet" == trans_config["mode"]:
        pass
    else:
        param['mode'] = trans_config['mode']
        param['curv'] = int(trans_config['use_curv'])
        param['peaks'] = trans_config['peaks']
        param['fac'] = trans_config['disp_fac']

        for i in range(data_cnt):
            param['in'] = output_path % i + "_%03d"
            param['out'] = trans_path % i + "_%03d"
            run_manta(manta_path, "scenes/transform_particles.py", param)
