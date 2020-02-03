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

data_path += "real/"
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

factor_d = math.pow(pre_config['factor'], 1/data_config['dim'])
param['bnd'] = int(math.ceil((data_config['bnd'] / factor_d)))

# write only every 30th frame -> 30 frames are one timestep
param['fps'] = data_config['fps']

# simulation time (how many frames)
param['t'] = data_config['frame_count']

param['res'] = int(data_config['res'] / factor_d)

param['dim'] = int(data_config['dim'])

# run random training setups
random.seed(data_config['seed']+45)

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

data_cnt = data_config['test_count']
modes = data_config['modes']

m_idx = -1
n_idx = 0
for i in range(data_cnt):
    if i >= n_idx:
        m_idx += 1
        n_idx = i + modes[m_idx]['prop'] * data_cnt
        
        param['circ'] = modes[m_idx]['circ_vel']/factor_d
        param['wlt'] = modes[m_idx]['wlt_vel']/factor_d
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
                print(scx, scy, scz)
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

    run_gen(cubes, spheres, i)
    param['seed'] = random.randint(0,1000000000)
