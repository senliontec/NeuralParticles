from neuralparticles.tools.param_helpers import *
import json
from neuralparticles.tools.shell_script import *
import math

import random

data_path = getParam("data", "data/")
manta_path = getParam("manta", "neuralparticles/build/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0
gui = int(getParam("gui", 0))
pause = int(getParam("pause", 0))

checkUnusedParams()

if not os.path.exists(data_path):
	os.makedirs(data_path)

data_path += "reference/"
if not os.path.exists(data_path):
	os.makedirs(data_path)

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

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

param['res'] = data_config['res']

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
    
def get_scalar(v, bnd=0.0):
    if type(v) is list:
        return random.uniform(v[0]+bnd,v[1]-bnd)
    else:
        return v

def gen_data(count,off=0):
    modes = data_config['modes']
    m_idx = -1
    n_idx = 0

    for i in range(count):
        if i >= n_idx:
            m_idx += 1
            n_idx = i + int(modes[m_idx]['prop'] * count)
            
            param['circ'] = modes[m_idx]['circ_vel']
            param['wlt'] = modes[m_idx]['wlt_vel']
            param['grav'] = modes[m_idx]['grav']
        cubes = {}
        spheres = {}

        basin_h = modes[m_idx]['basin']
        if basin_h > 0.0:
            if param['dim'] == 2:
                cubes['c0'] = "0.5,%f,0.5,%f" % (basin_h/2, basin_h)
            else:
                cubes['c0'] = "0.5,%f,0.5,0.5,%f,0.5" % (basin_h/2, basin_h)

        for c in range(random.randint(modes[m_idx]['cnt'][0],modes[m_idx]['cnt'][1])):    
            if random.random() < modes[m_idx]['cube_prob']:
                scx = get_scalar(modes[m_idx]['scale_x'])
                scy = get_scalar(modes[m_idx]['scale_y'])
                px = get_scalar(modes[m_idx]['pos_x'],scx/2)
                py = get_scalar(modes[m_idx]['pos_y'],scy/2)
                if param['dim'] == 3:
                    scz = get_scalar(modes[m_idx]['scale_z'])
                    pz = get_scalar(modes[m_idx]['pos_z'],scz/2)
                    cubes['c%d'%len(cubes)] = "%f,%f,%f,%f,%f,%f"%(px,py,pz,scx,scy,scz)
                else:
                    cubes['c%d'%len(cubes)] = "%f,%f,%f,%f"%(px,py,scx,scy)
            else:
                rad = get_scalar(modes[m_idx]['rad'])
                px = get_scalar(modes[m_idx]['pos_x'],rad/2)
                py = get_scalar(modes[m_idx]['pos_y'],rad/2)
                if param['dim'] == 3:
                    pz = get_scalar(modes[m_idx]['pos_z'],rad/2)
                    spheres['s%d'%len(spheres)] = "%f,%f,%f,%f"%(px,py,pz,rad)
                else:
                    spheres['s%d'%len(spheres)] = "%f,%f,%f"%(px,py,rad)

        param['seed'] = random.randint(0,1000000000)
        run_gen(cubes, spheres, i+off)

data_cnt = data_config['data_count']
test_cnt = data_config['test_count']

gen_data(data_cnt)
gen_data(test_cnt,data_cnt)

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

        for i in range(data_cnt + test_cnt):
            param['in'] = output_path % i + "_%03d"
            param['out'] = trans_path % i + "_%03d"
            run_manta(manta_path, "scenes/transform_particles.py", param)
