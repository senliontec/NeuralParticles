import sys, os, warnings

import json
from .uniio import *
from .particle_grid import ParticleIdxGrid
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

def extract_particles(data, pos, cnt, constraint, pad_val=0.0, aux_data={}):
    par_idx = particle_radius(data, pos, constraint)
    np.random.shuffle(par_idx)
    par_idx = par_idx[:min(cnt,len(par_idx))]

    par_pos = np.subtract(data[par_idx],pos)/constraint
    par_aux = {}
    for k, v in aux_data.items():
        par_aux[k] = v[par_idx]

    if len(par_pos) < cnt:
        #idx = np.random.randint(0,len(par_pos),cnt-len(par_pos))
        #pad = par_pos[idx]
        pad = pad_val * np.ones((cnt-len(par_pos), par_pos.shape[-1]))
        par_pos = np.concatenate((par_pos, pad))
        for k, v in par_aux.items():
            #pad = v[idx]
            pad = pad_val * np.ones((cnt-len(v), v.shape[-1]))
            par_aux[k] = np.concatenate((v, pad))
            
    return par_pos, par_aux

def extract_patch(data, pos, patch_size):
    patch_size = patch_size//2
    x0 = int(pos[0])-patch_size
    x1 = int(pos[0])+patch_size+1
    y0 = int(pos[1])-patch_size
    y1 = int(pos[1])+patch_size+1
    return data[0,y0:y1,x0:x1]

def interpol_grid(grid):
    x_v = np.arange(0.5, grid.shape[2]+0.5)
    y_v = np.arange(0.5, grid.shape[1]+0.5)
    if grid.shape[0] == 1:
        z_v = np.array([0.0,1.0])
        return interpolate.RegularGridInterpolator((x_v, y_v, z_v), np.transpose(np.concatenate([grid,grid]),(2,1,0,3)), bounds_error=False)
    else:
        z_v = np.arange(0.5, grid.shape[0]+0.5)
        return interpolate.RegularGridInterpolator((x_v, y_v, z_v), np.transpose(grid,(2,1,0,3)), bounds_error=False)

def normals(sdf):
    g = np.gradient(np.squeeze(sdf))
    if sdf.shape[0] == 1:
        g.insert(0, np.zeros_like(g[0]))
        g = np.expand_dims(g,axis=1)
    g = np.expand_dims(g,axis=-1)
    g = np.concatenate([g[2],g[1],g[0]],axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nan_to_num(g/np.linalg.norm(g,axis=-1,keepdims=True))
    return result

def curvature(sdf):
    dif = np.gradient(normals(sdf))
    return (np.linalg.norm(dif[0],axis=-1)+np.linalg.norm(dif[1],axis=-1)+np.linalg.norm(dif[2],axis=-1))/3

def in_bound(pos, bnd_min, bnd_max):
    return np.where(np.all([np.all(bnd_min<=pos,axis=-1),np.all(pos<=bnd_max,axis=-1)],axis=0))

def get_positions(particle_data, sdf, patch_size, surface=1.0, bnd=0):
    sdf_f = interpol_grid(sdf)
    particle_data_bound = particle_data[in_bound(particle_data[:,:2] if sdf.shape[0] == 1 else particle_data, bnd+patch_size/2,sdf.shape[1]-(bnd+patch_size/2))]
    positions = particle_data_bound[np.where(sdf_f(particle_data_bound) < surface)[0]]#particle_data_bound[in_surface(np.array([sdf_f(p) for p in particle_data_bound]), surface)[0]]
    print(positions.shape)
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

    features = train_config['features'][1:]

    data_cnt = int(data_config['data_count'] * train_config['train_split'])
    t_start = train_config['t_start']
    t_end = train_config['t_end']

    print("load %d dataset from timestep %d to %d" % (data_cnt, t_start, t_end))

    tmp_path = data_path + "tmp/patches_%s_d%03d_%03d-%03d_s%s/" % (os.path.splitext(os.path.basename(config_path))[0], data_cnt, t_start, t_end, ''.join(features))
    if not os.path.exists(tmp_path):
        src_path = data_path + "patches/source/"
        ref_path = data_path + "patches/reference/"

        rot_src_path = "%s%s_%s-%s_rot_ps" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"
        src_path = "%s%s_%s-%s_p" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"
        rot_ref_path = "%s%s_%s-%s_rot_ps" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"
        ref_path = "%s%s_%s-%s_ps" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"

        par_cnt = pre_config['par_cnt']
        par_cnt_ref = pre_config['par_cnt_ref']

        src = [np.empty((0,par_cnt,3))]
        if len(features) > 0:
            src.append(np.empty((0,par_cnt,len(features) + 2 if 'v' in features else 0)))
        src_rot = np.empty((0,par_cnt,3))
        ref = np.empty((0, par_cnt_ref,3))
        ref_rot = np.empty((0, par_cnt_ref,3))
        
        for d in range(data_cnt):
            for t in range(t_start, t_end):
                print("load patch: datasets: %03d timestep: %03d" % (d,t), end="\r", flush=True)
                src[0] = np.append(src[0], readNumpyRaw(src_path % ('s',d,t)), axis=0)
                if len(features) > 0:
                    src[1] = np.append(src[1], np.concatenate([readNumpyRaw(src_path%(f,d,t)) for f in features], axis=-1), axis=0)
                src_rot = np.append(src_rot, readNumpyRaw(rot_src_path%(d,t)), axis=0)
                ref = np.append(ref,readNumpyRaw(ref_path%(d,t)), axis=0)
                ref_rot = np.append(ref_rot,readNumpyRaw(rot_ref_path%(d,t)), axis=0)

        print("\r", flush=True)
        print("cache patch buffer")
        os.makedirs(tmp_path)
        writeNumpyRaw(tmp_path + "src", src[0])
        if len(features) > 0:
            writeNumpyRaw(tmp_path + "aux", src[1])
        writeNumpyRaw(tmp_path + "ref", ref)
        writeNumpyRaw(tmp_path + "src_rot", src_rot)
        writeNumpyRaw(tmp_path + "ref_rot", ref_rot)
    else:
        print("found and loaded cached buffer file")
        src = [readNumpyRaw(tmp_path + "src")]
        if len(features) > 0:
            src.append(readNumpyRaw(tmp_path + "aux"))
        ref = readNumpyRaw(tmp_path + "ref")
        src_rot = readNumpyRaw(tmp_path + "src_rot")
        ref_rot = readNumpyRaw(tmp_path + "ref_rot")

    return src, ref, src_rot, ref_rot

def load_patches(prefix, par_cnt, patch_size, surface = 1.0, par_aux=[] , bnd=0, pad_val=0.0, positions=None):

    par_aux_data = {}
    sdf = readUni(prefix + "_sdf.uni")[1]
    particle_data = readParticles(prefix + "_ps.uni")[1]
    for v in par_aux:
        par_aux_data[v] = readParticles((prefix+"_p%s.uni")%v, "float32")[1]

    if positions is None:
        positions = get_positions(particle_data, sdf, patch_size, surface, bnd)

    par_patches = np.empty((len(positions), par_cnt, 3))
    par_patches_rot = np.empty((len(positions), par_cnt, 3))
    par_aux_patches = {}
        
    for v in par_aux:
        par_aux_patches[v] = np.empty((len(positions), par_cnt, par_aux_data[v].shape[-1]))

    nor_f = interpol_grid(normals(sdf))

    idx_grid = ParticleIdxGrid(particle_data, sdf.shape[:3])

    #import time
    #avg_extract = 0
    #avg_total = 0
    #start = time.time()
    for i in range(len(positions)):
        #start = time.time()
        pos = positions[i]
        print("gen patch: %06d/%d" % (i,len(positions)), end="\r", flush=True)
        idx = idx_grid.get_range(pos, patch_size/2)
        tmp_aux = {}
        for v in par_aux_data:
            tmp_aux[v] = par_aux_data[v][idx]
        data, aux = extract_particles(particle_data[idx], pos, par_cnt, patch_size/2, pad_val, tmp_aux)

        #end = time.time()
        #avg_extract += end - start
        #if i % 1000 == 0:
        #    print("Avg Extract Time: ", avg_extract)
        #    avg_extract = 0

        par_patches[i] = data
        for k, v in aux.items():
            par_aux_patches[k][i] = v

        nor = nor_f(pos)[0]

        theta = math.atan2(nor[0], nor[1])
        c, s = math.cos(-theta), math.sin(-theta)
        mat0 = np.matrix([[c,-s,0],[s,c,0],[0,0,1]])

        theta = math.atan2(nor[2], nor[1])
        c, s = math.cos(theta), math.sin(theta)
        mat1 = np.matrix([[1,0,0],[0,c,-s],[0,s,c]])

        par_patches_rot[i] = data*mat0*mat1

        #end = time.time()
        #avg_total += end - start
        #if i % 1000 == 0:
        #    print("Avg Total Time: ", avg_total)
        #    avg_total = 0
    #end = time.time()
    #print(end-start)
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

def gen_patches(data_path, config_path, d_start=0, d_stop=None, t_start=0, t_stop=None, v_start=0, v_stop=None, pv_start=0, pv_stop=None, features=None):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    fac_d = math.pow(pre_config['factor'], 1/data_config['dim'])
    print(fac_d)
    patch_size = pre_config['patch_size']
    patch_size_ref = pre_config['patch_size_ref']

    par_cnt = pre_config['par_cnt']
    par_cnt_ref = pre_config['par_cnt_ref']

    pad_val = pre_config['pad_val']

    if d_stop is None:
        d_stop = data_config['data_count']
    if t_stop is None:
        t_stop = data_config['frame_count']
    if v_stop is None:
        v_stop = pre_config['var']
    if pv_stop is None:
        pv_stop = pre_config['par_var']
    if features is None:
        features = ['v','d','p']

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
        for v in range(v_start, v_stop):
            for t in range(t_start, t_stop):
                for r in range(pv_start, pv_stop):
                    par, aux_par, par_rot, positions = load_patches(path_src%(d,v,t), par_cnt, patch_size, surface, pad_val=pad_val, par_aux=features)
                    main = np.append(main, par, axis=0)
                    main_rot = np.append(main_rot, par_rot, axis=0)
                    pos = np.append(pos, positions, axis=0)
                    if len(features) > 0:
                        tmp = np.concatenate([(aux_par[f]) for f in features], axis=-1)
                        aux = tmp if aux is None else np.append(aux, tmp, axis=0)

                    par, aux_par, par_rot = load_patches(path_ref%(d,t), par_cnt_ref, patch_size_ref, pad_val=pad_val, positions=positions*fac_d)[:3]
                    reference = np.append(reference, par, axis=0)
                    ref_rot = np.append(ref_rot, par_rot, axis=0)
    
    return [main, aux] if len(features) > 0 else [main], reference, main_rot, ref_rot, pos


class PatchExtractor:
    def __init__(self, src_data, sdf_data, patch_size, cnt, surface=1.0, stride=0, bnd=0, pad_val=0.0, aux_data={}, features=[]):
        self.src_data = src_data
        self.radius = patch_size/2
        self.cnt = cnt
        self.stride = stride if stride > 0 else self.radius
        self.aux_data = aux_data
        self.features = features
        self.pad_val = pad_val

        self.pos_backup = get_positions(src_data, sdf_data, patch_size, surface, bnd)
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
        
        patch, aux = extract_particles(self.src_data, self.positions[idx], self.cnt, self.radius, self.pad_val, self.aux_data)
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
        patch, aux = extract_particles(self.src_data, self.last_pos, self.cnt, self.radius, self.pad_val, self.aux_data)

        if len(aux) > 0:
            return [np.array([patch]), np.array([np.concatenate([aux[f] for f in self.features],axis=-1)])]
        else:
            return [np.array([patch])]

    def set_patch(self, patch):
        self.data = np.concatenate((self.data, self.transform_patch(patch)))
