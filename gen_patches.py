import sys, os, warnings
sys.path.append("manta/scenes/tools")

import json
from helpers import *
import numpy as np

import random

import math

from patch_helper import *

class PatchExtractor:
    def __init__(self, src_data, sdf_data, patch_size, cnt, surface=1.0, stride=0, bnd=0):
        self.src_data = src_data
        self.data = src_data.copy()
        self.positions = get_positions(src_data, np.squeeze(sdf_data), patch_size, surface, bnd)
        np.random.shuffle(self.positions)
        self.last_pos = None
        self.radius = patch_size/2
        self.cnt = cnt
        self.stride = stride if stride > 0 else self.radius
    
    def transform_patch(self, patch):
        return np.add(patch * self.radius, self.last_pos)

    def inv_transform_patch(self, patch):
        return np.subtract(patch, self.last_pos) / self.radius

    def get_patch(self):
        if len(self.positions) == 0:
            return None

        self.last_pos = self.positions[0]
        self.positions = remove_particles(self.positions, self.last_pos, self.stride)[0]
        self.data = remove_particles(self.data, self.last_pos, self.stride)[0]
        patch = extract_particles(self.src_data, self.last_pos, self.cnt, self.radius)[0]
        return patch

    def set_patch(self, patch):
        self.data = np.concatenate((self.data, self.transform_patch(patch)))


def get_data_pair(data_path, config_path, dataset, timestep, var):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    np.random.seed(data_config['seed'])

    path_src = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
    path_ref = "%sreference/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"
    print(path_src)
    print(path_ref)

    return get_data(path_src%(dataset,var,timestep)), get_data(path_ref%(dataset,timestep))

def gen_patches(data_path, config_path, d_stop, t_stop, var, par_var, d_start=0, t_start=0):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
        train_config = json.loads(f.read())

    fac_2d = math.sqrt(pre_config['factor'])
    patch_size = pre_config['patch_size']
    patch_size_ref = pre_config['patch_size_ref']

    l_fac = pre_config['l_fac']
    h_fac = pre_config['h_fac']
    use_tanh = pre_config['use_tanh']

    par_cnt = pre_config['par_cnt']
    par_cnt_ref = pre_config['par_cnt_ref']

    features = train_config['features'][1:]
    use_particles = train_config['explicit'] != 0

    #stride = pre_config['stride']

    # tolerance of surface
    surface = pre_config['surf']

    np.random.seed(data_config['seed'])

    path_src = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
    path_ref = "%sreference/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"
    print(path_src)
    print(path_ref)

    main = np.empty([0,par_cnt, 3]) if use_particles else np.empty([0, patch_size, patch_size, 1])
    main_rot = np.empty([0,par_cnt, 3]) if use_particles else np.empty([0, patch_size, patch_size, 1])
    aux = None
    reference = np.empty([0,par_cnt_ref, 3]) if use_particles else np.empty([0, patch_size_ref, patch_size_ref, 1])
    pos = np.empty([0, 3])

    for d in range(d_start, d_stop):
        for v in range(var):
            for t in range(t_start, t_stop):
                for r in range(par_var):
                    sdf, aux_sdf, par, aux_par, par_rot, positions = load_patches(path_src%(d,v,t), par_cnt, patch_size, surface, grid_aux=features if not use_particles else [], par_aux=features if use_particles else [])
                    main = np.append(main, par if use_particles else np.tanh(sdf*l_fac) if use_tanh else sdf*l_fac, axis=0)
                    main_rot = np.append(main_rot, par_rot if use_particles else None, axis=0)
                    pos = np.append(pos, positions, axis=0)
                    if len(features) > 0:
                        tmp = np.concatenate([(aux_par[f] if use_particles else aux_sdf[f]) for f in features])
                        aux = tmp if aux is None else np.append(aux, tmp, axis=0)

                    sdf, aux_sdf, par, aux_par = load_patches(path_ref%(d,t), par_cnt_ref, patch_size_ref, positions=positions*fac_2d)[:4]
                    reference = np.append(reference, par if use_particles else np.tanh(sdf*h_fac) if use_tanh else sdf*h_fac, axis=0)

    if len(features) > 0:
        main = [main, aux]
    
    return main, reference, main_rot, pos