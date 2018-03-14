import sys, os, warnings
sys.path.append("manta/scenes/tools")

import json
from helpers import *
from uniio import *
import numpy as np

import random

import math

import scipy
from scipy import interpolate

def sdf_func(sdf):
    x_v = np.arange(0.5, sdf.shape[1]+0.5)
    y_v = np.arange(0.5, sdf.shape[0]+0.5)
    sdf_f = lambda x: interpolate.interp2d(x_v, y_v, sdf)(x[0],x[1])
    return sdf_f

def nor_func(sdf):
    y,x = np.gradient(sdf)
    x = np.expand_dims(x,axis=-1)
    y = np.expand_dims(y,axis=-1)
    g = np.concatenate([x,y],axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    nor = np.nan_to_num(g/np.linalg.norm(g,axis=-1,keepdims=True))
    x_v = np.arange(0.5, sdf.shape[1]+0.5)
    y_v = np.arange(0.5, sdf.shape[0]+0.5)
    return lambda x: np.concatenate([interpolate.interp2d(x_v, y_v, nor[:,:,0])(x[0],x[1]), interpolate.interp2d(x_v, y_v, nor[:,:,1])(x[0],x[1])])

def in_bound(pos, bnd_min, bnd_max):
    return np.where(np.all([np.all(bnd_min<=pos,axis=-1),np.all(pos<=bnd_max,axis=-1)],axis=0))

def in_surface(sdf, surface):
    return np.where(abs(sdf) < surface)

def load_patches(prefix, par_cnt, patch_size, surface = 1.0, stride = 1, grid_aux=[], par_aux=[] , bnd=0, positions=None):
    sdf_patches = np.empty((0, patch_size, patch_size, 1))
    grid_aux_patches = {}
    grid_aux_data = {}

    par_patches = np.empty((0, par_cnt, 3))
    par_patches_rot = np.empty((0, par_cnt, 3))
    par_aux_patches = {}
    par_aux_data = {}

    header, sdf = readUni(prefix + "_sdf.uni")
    for v in grid_aux:
        grid_aux_data[v] = readUni((prefix+"_%s.uni")%v)[1]
        grid_aux_patches[v] = np.empty((0, patch_size, patch_size, grid_aux_data[v].shape[-1]))

    particle_data = readParticles(prefix + "_ps.uni")[1]
    for v in par_aux:
        par_aux_data[v] = readParticles((prefix+"_p%s.uni")%v, "float32")[1]
        par_aux_patches[v] = np.empty((0, par_cnt, par_aux_data[v].shape[-1]))

    if positions is None:
        sdf_f = sdf_func(np.squeeze(sdf))
        particle_data_bound = particle_data[in_bound(particle_data[:,:2], bnd+patch_size/2,header['dimX']-(bnd+patch_size/2))]
        positions = particle_data_bound[in_surface(np.array([sdf_f(p) for p in particle_data_bound]), surface)[0]]

    '''if positions is None:
        positions = get_patches(sdf, patch_size, header['dimX'], header['dimY'], bnd, stride, surface)'''

    nor_f = nor_func(np.squeeze(sdf))

    for pos in positions:
        data = extract_patch(sdf, pos, patch_size)
        sdf_patches = np.append(sdf_patches, [data], axis=0)
        for k, v in grid_aux_data.items():
            data = extract_patch(v, pos, patch_size)
            grid_aux_patches[k] = np.append(grid_aux_patches[k], [data], axis=0)

        data, aux = extract_particles(particle_data, pos, par_cnt, patch_size/2, par_aux_data)
        par_patches = np.append(par_patches, [data], axis=0)
        for k, v in aux.items():
            par_aux_patches[k] = np.append(par_aux_patches[k], [v], axis=0)

        nor = nor_f(pos[:2])
        theta = math.atan2(nor[0], nor[1])
        c, s = math.cos(-theta), math.sin(-theta)
        mat = np.matrix([[c,-s,0],[s,c,0],[0,0,1]])
        par_patches_rot = np.append(par_patches_rot, [data*mat], axis=0)

    return sdf_patches, grid_aux_patches, par_patches, par_aux_patches, par_patches_rot, positions

if __name__ == '__main__':
        
    paramUsed = []

    data_path = getParam("data", "data/", paramUsed)
    manta_path = getParam("manta", "manta/", paramUsed)
    config_path = getParam("config", "config/version_00.txt", paramUsed)
    verbose = int(getParam("verbose", 0, paramUsed)) != 0

    checkUnusedParam(paramUsed)

    src_path_src = data_path + "source/"
    src_path_ref = data_path + "reference/"

    dst_path = data_path + "patches/"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    dst_path_src = dst_path + "source/"
    if not os.path.exists(dst_path_src):
        os.makedirs(dst_path_src)

    dst_path_ref = dst_path + "reference/"
    if not os.path.exists(dst_path_ref):
        os.makedirs(dst_path_ref)

    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())


    frame_count = data_config['frame_count']

    fac_2d = math.sqrt(pre_config['factor'])

    patch_size = pre_config['patch_size']
    patch_size_ref = int(patch_size * fac_2d)

    #border = int(math.ceil(patch_size_ref//2-patch_size//2*fac_2d))

    l_fac = pre_config['l_fac']
    h_fac = pre_config['h_fac']
    use_tanh = pre_config['use_tanh']

    par_cnt = pre_config['par_cnt']
    par_cnt_ref = pre_config['par_cnt_ref']

    np.random.seed(data_config['seed'])

    stride = pre_config['stride']

    # tolerance of surface
    surface = pre_config['surf']

    src_path_src = "%s%s_%s-%s" % (src_path_src, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
    src_path_ref = "%s%s_%s" % (src_path_ref, data_config['prefix'], data_config['id']) + "_d%03d_%03d"
    print(src_path_src)
    print(src_path_ref)

    dst_path_src = "%s%s_%s-%s" % (dst_path_src, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_pvar%02d_%03d_"
    dst_path_ref = "%s%s_%s-%s" % (dst_path_ref, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_pvar%02d_%03d_"
    print(dst_path_src)
    print(dst_path_ref)

    backupSources(dst_path_src)
    backupSources(dst_path_ref)

    grid_aux_data = {"vel", "dens", "pres"}
    par_aux_data = {"v", "d", "p"}

    for d in range(data_config['data_count']):
        for v in range(pre_config['var']):
            for t in range(frame_count):
                for r in range(pre_config['par_var']):
                    #for r in range(repetitions):
                    sdf, aux_sdf, par, aux_par, par_rot, positions = load_patches(src_path_src%(d,v,t), par_cnt, patch_size, surface, stride, grid_aux_data, par_aux_data)
                    path = dst_path_src%(d,v,r,t)
                    for i in range(len(sdf)):
                        writeNumpyBuf(path+"sdf", np.tanh(sdf[i]*l_fac) if use_tanh else sdf[i]*l_fac)
                        writeNumpyBuf(path+"ps", par[i])
                        writeNumpyBuf(path+"ps_rot", par_rot[i])
                        for p in grid_aux_data:
                            writeNumpyBuf(path+p, aux_sdf[p][i])
                        for p in par_aux_data:
                            writeNumpyBuf(path+"p"+p, aux_par[p][i])                

                    sdf, aux_sdf, par, aux_par = load_patches(src_path_ref%(d,t), par_cnt_ref, patch_size_ref, positions=positions*fac_2d)[:4]
                    path = dst_path_ref%(d,v,r,t)
                    for i in range(len(sdf)):
                        writeNumpyBuf(path+"sdf", np.tanh(sdf[i]*h_fac) if use_tanh else sdf[i]*h_fac)
                        writeNumpyBuf(path+"ps", par[i])

    finalizeNumpyBufs()