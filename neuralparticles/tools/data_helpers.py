import sys, os, warnings, shutil

import json
from .uniio import *
from .particle_grid import ParticleIdxGrid, interpol_grid, normals, curvature
import numpy as np

import random

import math, time


def particle_range(arr, pos, r):
    return np.where(np.all(np.abs(np.subtract(arr,pos)) <= r, axis=-1))[0]

def particle_radius(arr, pos, radius):
    return np.where(np.linalg.norm(np.subtract(arr,pos), axis=1) <= radius)[0]

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
    if len(par_idx) > cnt and False:
        print("Warning: using subset of particles (%d/%d)" % (cnt,len(par_idx)))
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

def in_bound(pos, bnd_min, bnd_max):
    return np.where(np.all([np.all(bnd_min<=pos,axis=-1),np.all(pos<=bnd_max,axis=-1)],axis=0))[0]

def in_surface(sdf_v, surface):
    return np.where(abs(sdf_v) < surface)[0]

def get_positions_idx(particle_data, sdf, patch_size, surface=1.0, bnd=0):
    sdf_f = interpol_grid(sdf)
    idx = in_bound(particle_data[:,:2] if sdf.shape[0] == 1 else particle_data, bnd+patch_size/2,sdf.shape[1]-(bnd+patch_size/2))
    return idx[in_surface(sdf_f(particle_data[idx]), surface)]

def get_positions(particle_data, sdf, patch_size, surface=1.0, bnd=0):
    return particle_data[get_positions_idx(particle_data, sdf, patch_size, surface, bnd)]

def get_nearest_idx(data, pos):
    return np.argmin(np.linalg.norm(pos-data, axis=-1), axis=0)

def get_nearest_point(data, pos, aux_data={}):
    idx = np.argmin(np.linalg.norm(pos-data, axis=-1), axis=0)
    aux = {}
    for k, v in aux_data.items():
        aux[k] = v[idx]
    return data[idx], aux

def get_data(prefix, par_aux=[]):
    par_aux_data = {}

    for v in par_aux:
        par_aux_data[v] = readParticles((prefix+"_p%s")%v, "float32")
        
    return readParticles(prefix + "_ps"), readGrid(prefix + "_sdf"), par_aux_data

def load_patches_from_file(data_path, config_path):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
        train_config = json.loads(f.read())

    features = train_config['features']
    features_ref = pre_config['features_ref']

    data_cnt = data_config['data_count']
    t_start = min(train_config['t_start'], data_config['frame_count']-1)
    t_end = min(train_config['t_end'], data_config['frame_count'])

    print("load %d dataset(s) from timestep %d to %d" % (data_cnt, t_start, t_end))

    src_path = data_path + "patches/source/"
    ref_path = data_path + "patches/reference/"
    tmp_path = data_path + "tmp/patches_%s_d%03d_%03d-%03d_s%s/" % (os.path.splitext(os.path.basename(config_path))[0], data_cnt, t_start, t_end, ''.join(features))

    # compares modification time of temporary buffer and first patch, if patches are newer the buffer will be deleted
    patch_path = "%s%s_%s-%s_ps_d%03d_%03d.npy" % (src_path, data_config['prefix'], data_config['id'], pre_config['id'], 0, t_start)
    if os.path.exists(tmp_path) and (not os.path.exists(patch_path) or os.path.getmtime(tmp_path) < os.path.getmtime(patch_path)):
        shutil.rmtree(tmp_path)

    if not os.path.exists(tmp_path):
        src_path = "%s%s_%s-%s_p" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"
        ref_path = "%s%s_%s-%s_p" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"

        par_cnt = pre_config['par_cnt']
        par_cnt_ref = pre_config['par_cnt_ref']

        patch_cnt = 0
        for d in range(data_cnt):
            for t in range(t_start, t_end):
                print("count patches: dataset(s): %03d timestep: %03d" % (d,t), end="\r", flush=True)
                patch_cnt += len(readNumpyRaw(src_path % ('s',d,t)))

        src = [np.empty((patch_cnt,par_cnt,3))]
        if len(features) > 0:
            src.append(np.empty((patch_cnt,par_cnt,len(features) + 2 if 'v' in features or 'n' in features else 0)))
        ref = [np.empty((patch_cnt, par_cnt_ref,3))]
        if len(features_ref) > 0:
            ref.append(np.empty((patch_cnt,par_cnt_ref,len(features_ref) + 2 if 'v' in features_ref or 'n' in features_ref else 0)))
        print("%d patches to load                                                    " % patch_cnt)
        patch_idx = 0
        for d in range(data_cnt):
            for t in range(t_start, t_end):
                print("load patch: dataset(s): %03d timestep: %03d" % (d,t), end="\r", flush=True)
                data = readNumpyRaw(src_path % ('s',d,t))
                src[0][patch_idx:patch_idx+len(data)] = data
                if len(features) > 0:
                    src[1][patch_idx:patch_idx+len(data)] = np.concatenate([readNumpyRaw(src_path%(f,d,t)) for f in features], axis=-1)
                ref[0][patch_idx:patch_idx+len(data)] = readNumpyRaw(ref_path%('s',d,t))
                if len(features_ref) > 0:
                    ref[1][patch_idx:patch_idx+len(data)] = np.concatenate([readNumpyRaw(ref_path%(f,d,t)) for f in features_ref], axis=-1)
                patch_idx += len(data)

        print("\r", flush=True)
        print("cache patch buffer")
        os.makedirs(tmp_path)
        writeNumpy(tmp_path + "src", src[0])
        if len(features) > 0:
            writeNumpy(tmp_path + "aux", src[1])
        writeNumpy(tmp_path + "ref", ref[0])
        if len(features_ref) > 0:
            writeNumpy(tmp_path + "ref_aux", ref[1])
    else:
        print("found and loaded cached buffer file")
        src = [readNumpy(tmp_path + "src")]
        if len(features) > 0:
            src.append(readNumpy(tmp_path + "aux"))
        ref = [readNumpy(tmp_path + "ref")]
        if len(features_ref) > 0:
            ref.append(readNumpy(tmp_path + "ref_aux"))

    return src, ref

def load_patches(prefix, par_cnt, patch_size, surface = 1.0, par_aux=[] , bnd=0, pad_val=0.0, positions=None, stride=0.0):
    particle_data, sdf, par_aux_data = get_data(prefix, par_aux)

    if positions is None:
        p = get_positions(particle_data, sdf, patch_size, surface, bnd)

        if stride > 0.0:
            np.random.shuffle(p)
            positions = np.empty((0, p.shape[-1]))
            while len(p) > 0:
                positions = np.append(positions, [p[0]], axis=0)
                p = remove_particles(p, p[0], stride)[0]
        else:
            positions = p

    par_patches = np.empty((len(positions), par_cnt, 3))
    par_aux_patches = {}
        
    for v in par_aux:
        par_aux_patches[v] = np.empty((len(positions), par_cnt, par_aux_data[v].shape[-1]))

    idx_grid = ParticleIdxGrid(particle_data, sdf.shape[:3])

    for i in range(len(positions)):
        pos = positions[i]
        idx = idx_grid.get_range(pos, patch_size/2)
        tmp_aux = {}
        for v in par_aux_data:
            tmp_aux[v] = par_aux_data[v][idx]
        data, aux = extract_particles(particle_data[idx], pos, par_cnt, patch_size/2, pad_val, tmp_aux)

        par_patches[i] = data
        for k, v in aux.items():
            par_aux_patches[k][i] = v

    return par_patches, par_aux_patches, positions

def get_norm_factor(data_path, config_path):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
        train_config = json.loads(f.read())

    features = train_config['features']
    if len(features) < 1:
        return None

    data_cnt = data_config['data_count']
    t_start = min(train_config['t_start'], data_config['frame_count']-1)
    t_end = min(train_config['t_end'], data_config['frame_count'])

    path_src = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
    tmp_path = data_path + "tmp/cache_%s_d%03d_%03d-%03d_s%s/" % (os.path.splitext(os.path.basename(config_path))[0], data_cnt, t_start, t_end, ''.join(features))
    print(tmp_path)
    _path_src = path_src % (0, 0, t_start) + "_ps.uni"
    if os.path.exists(tmp_path) and (os.path.exists(_path_src) and os.path.getmtime(tmp_path) < os.path.getmtime(_path_src)):
        shutil.rmtree(tmp_path)

    if not os.path.exists(tmp_path):
        norm_factor = np.zeros((len(features) + (2 if 'v' in features else 0),))
        for d in range(data_cnt):
            for v in range(pre_config['var']):
                for t in range(t_start, t_end):
                    data = get_data(path_src%(d,v,t), features)[2]
                    i = 0
                    for f in features:
                        if 'v' == f:
                            norm_factor[i:i+3] = max(norm_factor[i], np.max(np.linalg.norm(data[f], axis=-1)))
                            i+=3
                        else:
                            norm_factor[i] = max(norm_factor[i], np.max(np.abs(data[f])))
                            i+=1
        os.makedirs(tmp_path)
        print("cached norm factor")
        writeNumpy(tmp_path + "norm_factor", norm_factor)
    else:
        print("found and loaded cached norm factor file")
        norm_factor = readNumpy(tmp_path + "norm_factor")

    return norm_factor

def get_data_pair(data_path, config_path, dataset, timestep, var, features=None):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
        train_config = json.loads(f.read())

    path_src = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
    path_ref = "%sreference/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"
    
    if features is None: features = train_config['features']

    return get_data(path_src%(dataset,var,timestep), par_aux=features), get_data(path_ref%(dataset,timestep))[:2]

def gen_patches(data_path, config_path, d_start=0, d_stop=None, t_start=0, t_stop=None, v_start=0, v_stop=None, pv_start=0, pv_stop=None, features=None, features_ref=None):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    fac_d = math.pow(pre_config['factor'], 1/data_config['dim'])
    patch_size = pre_config['patch_size'] * data_config['res'] / fac_d
    patch_size_ref = pre_config['patch_size_ref'] * data_config['res']

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
    if features_ref is None:
        features_ref = pre_config['features_ref']

    # tolerance of surface
    surface = pre_config['surf']
    stride = pre_config['stride']

    np.random.seed(data_config['seed'])

    path_src = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
    path_ref = "%sreference/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"

    main = np.empty([0,par_cnt, 3])
    aux = None
    ref_aux = None
    reference = np.empty([0,par_cnt_ref, 3])
    pos = np.empty([0, 3])

    for d in range(d_start, d_stop):
        for v in range(v_start, v_stop):
            for t in range(t_start, t_stop):
                for r in range(pv_start, pv_stop):
                    #print(path_src%(d,v,t) + " (%d)"%r)
                    
                    par, aux_par, positions = load_patches(path_src%(d,v,t), par_cnt, patch_size, surface, pad_val=pad_val, par_aux=features, bnd=data_config['bnd']/fac_d, stride=stride)
                    main = np.append(main, par, axis=0)
                    pos = np.append(pos, positions, axis=0)
                    if len(features) > 0:
                        tmp = np.concatenate([(aux_par[f]) for f in features], axis=-1)
                        aux = tmp if aux is None else np.append(aux, tmp, axis=0)

                    par, aux_par = load_patches(path_ref%(d,t), par_cnt_ref, patch_size_ref, pad_val=pad_val, positions=positions*fac_d)[:2]
                    reference = np.append(reference, par, axis=0)
                    if len(features_ref) > 0:
                        tmp = np.concatenate([(aux_par[f]) for f in features_ref], axis=-1)
                        ref_aux = tmp if ref_aux is None else np.append(ref_aux, tmp, axis=0)
    
    return [main, aux] if len(features) > 0 else [main], [reference, ref_aux] if len(features_ref) > 0 else [reference], pos


class PatchExtractor:
    def __init__(self, src_data, sdf_data, patch_size, cnt, surface=1.0, stride=-1, bnd=0, pad_val=0.0, aux_data={}, features=[], positions=None, last_pos=None):
        self.src_data = src_data
        self.radius = patch_size/2
        self.cnt = cnt
        self.stride = stride if stride >= 0 else self.radius
        self.aux_data = aux_data
        self.features = features
        self.pad_val = pad_val

        if positions is not None:
            self.positions = positions.copy()
        else:
            idx = get_positions_idx(src_data, sdf_data, patch_size, surface, bnd)
            p = src_data[idx]

            rnd = np.arange(len(idx),dtype=int)
            np.random.shuffle(rnd)
            idx = idx[rnd]
            p = p[rnd]
            self.pos_idx = np.empty((0,),dtype=int)

            i = 0
            while len(p) > 0:
                p_idx = 0 if last_pos is None or i >= len(last_pos) else get_nearest_idx(p, last_pos[i])
                self.pos_idx = np.append(self.pos_idx, [idx[p_idx]])
                r_idx = particle_radius(p, p[p_idx], self.stride)
                p = np.delete(p, r_idx, axis=0)
                idx = np.delete(idx, r_idx, axis=0)

            self.positions = src_data[self.pos_idx]
        '''p = get_positions(src_data, sdf_data, patch_size, surface, bnd)
        np.random.shuffle(p)

        self.positions = np.empty((0, p.shape[-1]))

        while len(p) > 0:
            self.positions = np.append(self.positions, [p[0]], axis=0)
            p = remove_particles(p, p[0], self.stride)[0]'''
        self.pos_backup = self.positions
        self.reset()
    
    def reset(self):
        self.data = self.src_data.copy()
        self.positions = self.pos_backup.copy()
        self.last_idx = -1

    def transform_patch(self, patch, pos):
        return patch * self.radius + pos

    def inv_transform_patch(self, patch, pos):
        return (patch - pos) / self.radius

    def get_patch_pos(self, pos, remove_data=True):
        if remove_data:
            self.data = remove_particles(self.data, pos, self.stride)[0]

        patch, aux = extract_particles(self.src_data, pos, self.cnt, self.radius, self.pad_val, self.aux_data)
        if len(aux) > 0:
            return np.concatenate([patch] + [aux[f] for f in self.features],axis=-1)
        else:
            return patch

    def get_patch(self, idx, remove_data=True):
        return self.get_patch_pos(self.positions[idx], remove_data)
    
    def pop_patch(self, remove_data=True):
        p = self.positions[:1]
        if len(self.positions) == 1:
            self.positions = None
        else:
            self.positions = self.positions[1:]
        return self.get_patch_pos(p, remove_data)
        
    def set_patch(self, patch, idx):
        self.data = np.concatenate((self.data, self.transform_patch(patch, self.positions[idx])))

    def get_patches(self):
        src = np.empty((len(self.positions),self.cnt,3 + len(self.features) + (2 if 'v' in self.features else 0)))

        for i in range(len(self.positions)):
            patch = self.get_patch(i)
            src[i] = patch

        return [src]

    def set_patches(self, patches):
        self.data = np.concatenate((self.data, np.concatenate(self.transform_patch(patches, np.expand_dims(self.positions,axis=1)))))
