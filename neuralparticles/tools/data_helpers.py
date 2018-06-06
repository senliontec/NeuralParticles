import sys, os, warnings

import json
from neuralparticles.tools.uniio import *
import numpy as np

import random

import math

import scipy
from scipy import interpolate

def particle_range(arr, pos, r):
	return np.where(np.all(np.abs(np.subtract(arr,pos)) < r, axis=-1))[0]

def particle_radius(arr, pos, radius):
	return np.where(np.linalg.norm(np.subtract(arr,pos), axis=1) < radius)[0]

def insert_patch(data, patch, pos, func):
	patch_size = patch.shape[0]//2
	x0=int(pos[0])-patch_size
	x1=int(pos[0])+patch_size+1
	y0=int(pos[1])-patch_size
	y1=int(pos[1])+patch_size+1

	data[0,y0:y1,x0:x1] = func(data[0,y0:y1,x0:x1], patch)

def remove_particles(data, pos, constraint, aux_data={}):
	par_idx = particle_radius(data, pos, constraint)
	par_aux = {}
	for k, v in aux_data.items():
		par_aux[k] = np.delete(v, par_idx, axis=0)
		
	return np.delete(data, par_idx, axis=0), par_aux

'''def extract_remove_particles(data, pos, cnt, constraint, aux_data={}):
	par_idx = particle_radius(data, pos, constraint)
	np.random.shuffle(par_idx)
	par_idx = par_idx[:min(cnt,len(par_idx))]

	par_pos = np.subtract(data[par_idx],pos)/constraint
	data = np.delete(data, par_idx, axis=0)
	par_aux = {}
	for k, v in aux_data.items():
		par_aux[k] = v[par_idx]
		v = np.delete(v, par_idx, axis=0)

	if len(par_pos) < cnt:
		par_pos = np.concatenate((par_pos,np.zeros((cnt-len(par_pos),par_pos.shape[-1]))))
		for k, v in par_aux.items():
			par_aux[k] = np.concatenate((v,np.zeros((cnt-len(v),v.shape[-1]))))
	
	return data, par_pos, aux_data, par_aux'''

def extract_particles(data, pos, cnt, constraint, aux_data={}):
	par_idx = particle_radius(data, pos, constraint)
	np.random.shuffle(par_idx)
	par_idx = par_idx[:min(cnt,len(par_idx))]

	par_pos = np.subtract(data[par_idx],pos)/constraint
	par_aux = {}
	for k, v in aux_data.items():
		par_aux[k] = v[par_idx]

	if len(par_pos) < cnt:
		par_pos = np.concatenate((par_pos,np.zeros((cnt-len(par_pos),par_pos.shape[-1]))))
		for k, v in par_aux.items():
			par_aux[k] = np.concatenate((v,np.zeros((cnt-len(v),v.shape[-1]))))
			
	return par_pos, par_aux

def extract_patch(data, pos, patch_size):
	patch_size = patch_size//2
	x0 = int(pos[0])-patch_size
	x1 = int(pos[0])+patch_size+1
	y0 = int(pos[1])-patch_size
	y1 = int(pos[1])+patch_size+1
	return data[0,y0:y1,x0:x1]

def sdf_func(sdf):
    x_v = np.arange(0.5, sdf.shape[1]+0.5)
    y_v = np.arange(0.5, sdf.shape[0]+0.5)
    return lambda x: interpolate.interp2d(x_v, y_v, sdf)(x[0],x[1])

def normals(sdf):
    y,x = np.gradient(sdf)
    x = np.expand_dims(x,axis=-1)
    y = np.expand_dims(y,axis=-1)
    g = np.concatenate([x,y],axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    return np.nan_to_num(g/np.linalg.norm(g,axis=-1,keepdims=True))

def nor_func(sdf):
    nor = normals(sdf)
    x_v = np.arange(0.5, sdf.shape[1]+0.5)
    y_v = np.arange(0.5, sdf.shape[0]+0.5)
    return lambda x: np.concatenate([interpolate.interp2d(x_v, y_v, nor[:,:,0])(x[0],x[1]), interpolate.interp2d(x_v, y_v, nor[:,:,1])(x[0],x[1])])

def curvature(sdf):
    dif = np.gradient(normals(sdf))
    return (np.linalg.norm(dif[0],axis=-1)+np.linalg.norm(dif[1],axis=-1))/2

def curv_func(sdf):
    curv = curvature(sdf)
    x_v = np.arange(0.5, curv.shape[1]+0.5)
    y_v = np.arange(0.5, curv.shape[0]+0.5)
    return lambda x: interpolate.interp2d(x_v, y_v, curv)(x[0],x[1])

def in_bound(pos, bnd_min, bnd_max):
    return np.where(np.all([np.all(bnd_min<=pos,axis=-1),np.all(pos<=bnd_max,axis=-1)],axis=0))

def in_surface(sdf, surface):
    return np.where(abs(sdf) < surface)

def get_positions(particle_data, sdf, patch_size, surface=1.0, bnd=0):
    sdf_f = sdf_func(sdf)
    particle_data_bound = particle_data[in_bound(particle_data[:,:2], bnd+patch_size/2,sdf.shape[0]-(bnd+patch_size/2))]
    positions = particle_data_bound[in_surface(np.array([sdf_f(p) for p in particle_data_bound]), surface)[0]]
    return positions

def get_data(prefix, par_aux=[]):
    par_aux_data = {}

    for v in par_aux:
        par_aux_data[v] = readParticles((prefix+"_p%s.uni")%v, "float32")[1]
        
    return readParticles(prefix + "_ps.uni")[1], readUni(prefix + "_sdf.uni")[1], par_aux_data

def load_patches_from_file(data_path, config_path):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
        train_config = json.loads(f.read())

    src_path = data_path + "patches/source/"
    ref_path = data_path + "patches/reference/"

    features = train_config['features'][1:]

    par_cnt = pre_config['par_cnt']
    par_cnt_ref = pre_config['par_cnt_ref']

    src = np.empty((0,par_cnt))
    src_rot = np.empty((0,par_cnt))
    ref = np.empty((0, par_cnt_ref))
    ref_rot = np.empty((0, par_cnt_ref))

    path = "%s%s_%s-%s_p" % (src_path, data_config['prefix'], data_config['id'], pre_config['id'])
    print(path)
    src = [readNumpyRaw(path + 's')]
    for f in features:
        src.append(readNumpyRaw(path + f))

    path = "%s%s_%s-%s_p" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id'])
    print(path)
    ref = [readNumpyRaw(path + 's')]
    for f in features:
        ref.append(readNumpyRaw(path + f))
    
    path = "%s%s_%s-%s_rot_p" % (src_path, data_config['prefix'], data_config['id'], pre_config['id'])
    print(path)
    src_rot = readNumpyRaw(path + 's')
    
    path = "%s%s_%s-%s_rot_p" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id'])
    print(path)
    ref_rot = readNumpyRaw(path + 's')

    return src, ref, src_rot, ref_rot

def load_patches(prefix, par_cnt, patch_size, surface = 1.0, par_aux=[] , bnd=0, positions=None):
    par_patches = np.empty((0, par_cnt, 3))
    par_patches_rot = np.empty((0, par_cnt, 3))
    par_aux_patches = {}
    par_aux_data = {}

    sdf = readUni(prefix + "_sdf.uni")[1]
    particle_data = readParticles(prefix + "_ps.uni")[1]
    for v in par_aux:
        par_aux_data[v] = readParticles((prefix+"_p%s.uni")%v, "float32")[1]
        par_aux_patches[v] = np.empty((0, par_cnt, par_aux_data[v].shape[-1]))

    if positions is None:
        positions = get_positions(particle_data, np.squeeze(sdf), patch_size, surface, bnd)

    nor_f = nor_func(np.squeeze(sdf))

    for pos in positions:
        data, aux = extract_particles(particle_data, pos, par_cnt, patch_size/2, par_aux_data)
        par_patches = np.append(par_patches, [data], axis=0)
        for k, v in aux.items():
            par_aux_patches[k] = np.append(par_aux_patches[k], [v], axis=0)

        nor = nor_f(pos[:2])
        theta = math.atan2(nor[0], nor[1])
        c, s = math.cos(-theta), math.sin(-theta)
        mat = np.matrix([[c,-s,0],[s,c,0],[0,0,1]])
        par_patches_rot = np.append(par_patches_rot, [data*mat], axis=0)

    return par_patches, par_aux_patches, par_patches_rot, positions

def get_data_pair(data_path, config_path, dataset, timestep, var):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
        train_config = json.loads(f.read())

    np.random.seed(data_config['seed'])

    path_src = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
    path_ref = "%sreference/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"
    print(path_src)
    print(path_ref)
    
    features = train_config['features'][1:]

    return get_data(path_src%(dataset,var,timestep), par_aux=features), get_data(path_ref%(dataset,timestep))[:2]

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

    par_cnt = pre_config['par_cnt']
    par_cnt_ref = pre_config['par_cnt_ref']

    features = train_config['features'][1:]

    # tolerance of surface
    surface = pre_config['surf']

    np.random.seed(data_config['seed'])

    path_src = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
    path_ref = "%sreference/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"
    print(path_src)
    print(path_ref)

    main = np.empty([0,par_cnt, 3])
    main_rot = np.empty([0,par_cnt, 3])
    ref_rot = np.empty([0,par_cnt_ref, 3])
    aux = None
    reference = np.empty([0,par_cnt_ref, 3])
    pos = np.empty([0, 3])

    for d in range(d_start, d_stop):
        for v in range(var):
            for t in range(t_start, t_stop):
                for r in range(par_var):
                    par, aux_par, par_rot, positions = load_patches(path_src%(d,v,t), par_cnt, patch_size, surface, par_aux=features)
                    main = np.append(main, par, axis=0)
                    main_rot = np.append(main_rot, par_rot, axis=0)
                    pos = np.append(pos, positions, axis=0)
                    if len(features) > 0:
                        tmp = np.concatenate([(aux_par[f]) for f in features])
                        aux = tmp if aux is None else np.append(aux, tmp, axis=0)

                    par, aux_par, par_rot = load_patches(path_ref%(d,t), par_cnt_ref, patch_size_ref, positions=positions*fac_2d)[:3]
                    reference = np.append(reference, par, axis=0)
                    ref_rot = np.append(ref_rot, par_rot, axis=0)
    
    return [main, aux] if len(features) > 0 else [main], reference, main_rot, ref_rot, pos


class PatchExtractor:
    def __init__(self, src_data, sdf_data, patch_size, cnt, surface=1.0, stride=0, bnd=0, aux_data={}, features=[]):
        self.src_data = src_data
        self.radius = patch_size/2
        self.cnt = cnt
        self.stride = stride if stride > 0 else self.radius
        self.aux_data = aux_data
        self.features = features

        self.pos_backup = get_positions(src_data, np.squeeze(sdf_data), patch_size, surface, bnd)
        np.random.shuffle(self.pos_backup)

        self.reset()
    
    def reset(self):
        self.data = self.src_data.copy()
        self.positions = self.pos_backup
        self.last_pos = None

    def transform_patch(self, patch):
        return np.add(patch * self.radius, self.last_pos)

    def inv_transform_patch(self, patch):
        return np.subtract(patch, self.last_pos) / self.radius

    def get_patch_idx(self, idx):
        if len(self.positions) <= idx:
            return None
        
        patch, aux = extract_particles(self.src_data, self.positions[idx], self.cnt, self.radius, self.aux_data)

        if len(aux) > 0:
            return [np.array([patch]), np.array([np.concatenate([aux[f] for f in self.features])])]
        else:
            return [np.array([patch])]

    def get_patch(self):
        if len(self.positions) == 0:
            return None

        self.last_pos = self.positions[0]
        self.positions = remove_particles(self.positions, self.last_pos, self.stride)[0]
        self.data = remove_particles(self.data, self.last_pos, self.stride)[0]
        patch, aux = extract_particles(self.src_data, self.last_pos, self.cnt, self.radius, self.aux_data)

        if len(aux) > 0:
            return [np.array([patch]), np.array([np.concatenate([aux[f] for f in self.features])])]
        else:
            return [np.array([patch])]

    def set_patch(self, patch):
        self.data = np.concatenate((self.data, self.transform_patch(patch)))
