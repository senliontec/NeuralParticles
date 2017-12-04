import sys, os
sys.path.append("manta/scenes/tools")

import json
from shell_script import *
from helpers import *

import random

paramUsed = []

data_path = getParam("data", "data/", paramUsed)
manta_path = getParam("manta", "manta/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0
gui = int(getParam("gui", 0, paramUsed))

checkUnusedParam(paramUsed)

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

param['sres'] = data_config['sub_res']

# write only every 30th frame -> 30 frames are one timestep
param['fps'] = data_config['fps']

# simulation time (how many frames)
param['t_end'] = float(data_config['frame_count']) / data_config['fps']

# run random training setups
random.seed(data_config['seed'])

output_path = "%s%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d"
print(output_path)

def call_dataset_gen(var0,var1,var2,off):
    def run_gen(cubes,cnt):
        param['c_cnt'] = len(cubes)
        
        param['res'] = data_config['res']
        param['out'] = output_path % cnt + "_%03d"
        run_manta(manta_path, "scenes/2D_sph.py", dict(param, **cubes), verbose)
        
        #param['res'] = low_res
        #param['out'] = (test_prefix + "_d%03d")%cnt + "_%03d"
        #run_manta(manta_path, "scenes/2D_sph.py", dict(param, **cubes), verbose)      
        
    param['circ'] = 0
    for i in range(var0):
        # generate different cubes with dataformat "pos_x,pos_y,scale_x,scale_y"
        cubes = {}
        for c in range(random.randint(1,data_config['max_cnt'])):    
            scx = random.uniform(data_config['min_scale'], data_config['max_scale'])
            scy = random.uniform(data_config['min_scale'], data_config['max_scale'])
            px = random.uniform(data_config['min_pos']+scx/2, data_config['max_pos']-scx/2)
            py = random.uniform(0, data_config['max_h']) + scy/2
            cubes['c%d'%c] = "%f,%f,%f,%f"%(px,py,scx,scy)
        run_gen(cubes, off)
        off+=1
        

    for i in range(var1):
        cubes = {}
        scy = data_config['max_h']
        cubes['c0'] = "%f,%f,%f,%f"%(0, scy/2, 1, scy)
        for c in range(1,random.randint(2,data_config['max_cnt'])):    
            scx = random.uniform(data_config['min_scale'], data_config['max_scale'])*0.5
            scy = random.uniform(data_config['min_scale'], data_config['max_scale'])*0.5
            px = random.uniform(data_config['min_pos']+scx/2, data_config['max_pos']-scx/2)
            py = random.uniform(data_config['min_pos']+scy/2, data_config['max_pos']*0.5-scy/2)
            cubes['c%d'%c] = "%f,%f,%f,%f"%(px,py,scx,scy)
        run_gen(cubes, off)
        off+=1

    param['circ'] = data_config['circ_vel']
    for i in range(var2):
        cubes = {}
        for c in range(random.randint(2,data_config['max_cnt'])):    
            scx = random.uniform(data_config['min_scale'], data_config['max_scale'])*0.5
            scy = random.uniform(data_config['min_scale'], data_config['max_scale'])*0.5
            px = random.uniform(data_config['min_pos']+scx/2, data_config['max_pos']-scx/2)
            py = random.uniform(data_config['min_pos']+scy/2, data_config['max_pos']-scy/2)
            cubes['c%d'%c] = "%f,%f,%f,%f"%(px,py,scx,scy)
        run_gen(cubes, off)
        off+=1
    
var1 = int(data_config['data_count'] * data_config['var1'])
var2 = int(data_config['data_count'] * data_config['var2'])
var0 = data_config['data_count'] - var1 - var2
 
call_dataset_gen(var0,var1,var2,0)
