import sys, os, warnings

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