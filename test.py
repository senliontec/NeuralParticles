import sys, os, warnings
sys.path.append("manta/scenes/tools")
sys.path.append("hungarian/")
sys.path.append("structural_losses/")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
from shell_script import *
from helpers import *
from uniio import *

import scipy
from scipy import interpolate
from sklearn.cluster import KMeans

import keras
from keras import losses
from keras.models import Model, load_model
from keras.layers import Conv2D, Conv2DTranspose, Input, Dense, Dropout
from keras.layers import Reshape, concatenate, add, Flatten, Lambda
from spatial_transformer import *
from split_layer import *

import random
import math

from advection import interpol

from hungarian_loss import hungarian_loss

from particle_grid import ParticleGrid

import scipy.ndimage.filters as fi

import numpy as np

from gen_patches import gen_patches

random.seed(235)
np.random.seed(694)

import keras.backend as K
import tensorflow as tf

from sklearn.decomposition import PCA

'''from tf_nndistance import nn_distance
from tf_approxmatch import approx_match, match_cost

def chamfer_loss(y_true, y_pred):
    cost_p1_p2, _, cost_p2_p1, _ = nn_distance(y_pred, y_true)
    return tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

def emd_loss(y_true, y_pred):
    match = approx_match(y_pred, y_true)
    return tf.reduce_mean(match_cost(y_pred, y_true, match))'''

use_test_data = False
def eigenvector(data):
    pca = PCA()
    pca.fit(data)
    return pca.components_

def density_loss(y_true, y_pred):
    sh = [s for s in y_pred.get_shape()]
    sh[0] = 32

    y_pred.set_shape(sh)
    y_true.set_shape(sh)

    m_pred = K.expand_dims(y_pred,axis=2)

    m_pred_t = K.permute_dimensions(m_pred,(0,2,1,3))

    cost = K.sum(K.square(m_pred - m_pred_t),axis=-1) # K.sqrt(...)
    dens = K.sum(K.exp(-cost), axis=-1)

    return tf.reciprocal(dens)#tf.stack([K.sum(K.map_fn(kernel,K.flatten(c))) for c in tf.unstack(cost)])

def normals(sdf):
    y,x = np.gradient(sdf)
    x = np.expand_dims(x,axis=-1)
    y = np.expand_dims(y,axis=-1)
    g = np.concatenate([x,y],axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nan_to_num(g/np.linalg.norm(g,axis=-1,keepdims=True))

def curvature(nor):
    dif = np.gradient(nor)
    return (np.linalg.norm(dif[0],axis=-1)+np.linalg.norm(dif[1],axis=-1))/2

def sdf_func(sdf):
    x_v = np.arange(0.5, sdf.shape[1]+0.5)
    y_v = np.arange(0.5, sdf.shape[0]+0.5)
    sdf_f = lambda x: interpolate.interp2d(x_v, y_v, sdf)(x[0],x[1])
    nor = normals(sdf)
    nor_f = lambda x: np.concatenate([interpolate.interp2d(x_v, y_v, nor[:,:,0])(x[0],x[1]), interpolate.interp2d(x_v, y_v, nor[:,:,1])(x[0],x[1])])
    return sdf_f, nor_f

def sort_particles(par, nor, fac=1):
    w =  np.sum(np.abs(nor) + np.square(par), axis=-1)
    #w = np.nan_to_num(np.abs(np.reciprocal(np.sum(np.power(par[:,:2],fac) * nor, axis=-1)))) #+ fac1 * fac1 * np.sum(np.square(par), axis=-1)
    return np.argsort(w)

def min_avg(old_avg, new_val):
    m_a = np.repeat(np.expand_dims(old_avg,1),old_avg.shape[1],axis=1)
    m_b = np.repeat(np.expand_dims(new_val,2),new_val.shape[1],axis=2)
    cost = np.linalg.norm(m_b-m_a, axis=-1)
    sort_v = new_val[optimize.linear_sum_assignment(cost)[1]]

    return (old_avg + sort_v) / 2

paramUsed = []

gpu = getParam("gpu", "", paramUsed)

data_path = getParam("data", "data/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0

dataset = int(getParam("dataset", 0, paramUsed))
var = int(getParam("var", 0, paramUsed))

# mode 0: none
# mode 1: translation along normal
# mode 2: translation along x
# mode 3: cosinus waves on surface
trans_mode = int(getParam("trans_mode", 0, paramUsed))

# if smaller then 0 it will replaced by the curvature!
trans_fac = float(getParam("trans_fac", 10.0, paramUsed))
repetitions = int(getParam("rep", 10, paramUsed))
obj_cnt = int(getParam("obj_cnt", 10, paramUsed))
fixed = int(getParam("fixed", 0, paramUsed)) != 0

pre_train_stn = int(getParam("pre_train", 0, paramUsed)) != 0

prefix = getParam("prefix", "test", paramUsed)
source = getParam("l_scr", "source", paramUsed)
reference = getParam("h_scr", "reference", paramUsed)
result = getParam("r_scr", "result", paramUsed)
#sdf_scr = prefix + "/" + getParam("sdf_scr", "sdf_t%03d", paramUsed)
#sdf_t_scr = prefix + "/" + getParam("sdf_t_scr", "test_sdf_t%03d", paramUsed)

source_scr = prefix + "/" + source + "_t%03d"
reference_scr = prefix + "/" + reference + "_t%03d"
test_source_scr = prefix + "/test_" + source + "_t%03d"
test_reference_scr = prefix + "/test_" + reference + "_t%03d"
test_result_scr = prefix + "/test_" + result + "_t%03d"

sdf_reference_scr = prefix + "/sdf_" + reference + "_t%03d"
sdf_test_reference_scr = prefix + "/test_sdf_" + reference + "_t%03d"
sdf_test_result_scr = prefix + "/test_sdf_" + result + "_t%03d"

vec_reference_scr = prefix + "/vec_" + reference + "_t%03d"
vec_test_reference_scr = prefix + "/test_vec_" + reference + "_t%03d"
vec_test_result_scr = prefix + "/test_vec_" + result + "_t%03d"

dim = int(getParam("dim", 50, paramUsed))

patch_sample_cnt = int(getParam("samples", 3, paramUsed))

t_start = int(getParam("t_start", -1, paramUsed))
t_end = int(getParam("t_end", -1, paramUsed))

# mode:
# 0: only particles
# 1: only sdf
# 2: particles + sdf
# 3: particles with sdf loss
# 4: particles with residual network
# 5: only gradient
mode = int(getParam("mode", 0, paramUsed))

par_out  = mode == 0 or mode == 2 or mode == 3 or mode == 4
grid_out = mode == 1 or mode == 2 or mode == 5
gen_grid = mode == 1 or mode == 2 or mode == 3 or mode == 5
use_vec  = mode == 4 or mode == 5

checkUnusedParam(paramUsed)

if not gpu is "":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

if t_start < 0:
    t_start = train_config['t_start']

if t_end < 0:
    t_end = train_config['t_end']

samples = [[random.randint(t_start, t_end-1), random.randint(0, 20)] for i in range(patch_sample_cnt)]
print(samples)

patch_size = pre_config['patch_size']
fac_1d = pre_config['factor']
fac_2d = int(math.sqrt(fac_1d))
ref_patch_size = patch_size * fac_2d
surface = pre_config['surf']
particle_cnt_src = pre_config['par_cnt']
particle_cnt_dst = pre_config['par_cnt_ref']

extrapol = 5

h_dim = dim * fac_2d

def in_bound(pos, bnd_min, bnd_max):
    return np.where(np.all([np.all(bnd_min<=pos,axis=-1),np.all(pos<=bnd_max,axis=-1)],axis=0))

def in_surface(sdf):
    return np.where(abs(sdf) < surface)

def filter2D(kernlen, s, fac):
    dirac = np.zeros((kernlen, kernlen))
    dirac[kernlen//2, kernlen//2] = 1
    return np.clip(fi.gaussian_filter(dirac, s) * fac, a_min=None, a_max=1.0)

def translation(par, sdf, fac):
    res = np.empty((0,3))
    #x_v = np.arange(0.5, nor.shape[1]+0.5)
    #y_v = np.arange(0.5, nor.shape[0]+0.5)
    #nor_f = lambda x: np.concatenate([interpolate.interp2d(x_v, y_v, nor[:,:,0])(x[0],x[1]), interpolate.interp2d(x_v, y_v, nor[:,:,1])(x[0],x[1])])
    sdf_f, nor_f = sdf_func(sdf)
    if fac <= 0:
        x_v = np.arange(0.5, sdf.shape[1]+0.5)
        y_v = np.arange(0.5, sdf.shape[0]+0.5)
        curv = lambda x: interpolate.interp2d(x_v, y_v, curvature(normals(sdf)))(x[0],x[1])
    for p in par:
        n = (nor_f(p[:2]) if trans_mode == 1 else np.array([1,0])) * (fac if fac > 0 else ((-fac) * curv(p[:2]) / (1 - sdf_f(p[:2]))))
        res = np.append(res,np.array([[p[0]+n[0],p[1]+n[1],p[2]]]), axis=0)
    return res

circular_filter = np.reshape(filter2D(ref_patch_size, ref_patch_size*0.2, 500), (ref_patch_size, ref_patch_size))
elem_min = np.vectorize(lambda x,y: min(x,y))
elem_avg = np.vectorize(lambda x,y: y)

class RandomParticles:
    def __init__(self, dimX, dimY, fac_2d, max_size, cnt, cube_prob):
        self.dimX = dimX
        self.dimY = dimY
        self.fac_2d = fac_2d
        self.max_size = max_size
        self.cnt = cnt
        self.cube_prob = cube_prob
    
    def gen_random(self, pos=None, cube=None, a=None):
        self.pos = np.random.random((self.cnt,2))*np.array([self.dimY+1, self.dimX+1]) if pos is None else pos
        self.cube = np.random.random((self.cnt,)) > self.cube_prob if cube is None else cube
        self.a = 1+np.random.random((self.cnt,2))*(self.max_size-1) if a is None else a

    def get_grid(self):
        src_grid = ParticleGrid(self.dimX, self.dimY, 2, extrapol)
        ref_grid = ParticleGrid(self.dimX*self.fac_2d, self.dimY*self.fac_2d, 2, extrapol)
        for i in range(self.cnt):
            if self.cube[i]:
                src_grid.sample_quad(self.pos[i], self.a[i,0], self.a[i,1])
                ref_grid.sample_quad(self.pos[i] * self.fac_2d, self.a[i,0] * self.fac_2d, self.a[i,1] * self.fac_2d)
            else:
                src_grid.sample_sphere(self.pos[i], self.a[i,0])
                if trans_mode == 3:
                    ref_grid.sample_cos_sphere(self.pos[i] * self.fac_2d, self.a[i,0] * self.fac_2d, 6, trans_fac)
                else:
                    ref_grid.sample_sphere(self.pos[i] * self.fac_2d, self.a[i,0] * self.fac_2d)
        src_grid.sample_sdf()
        ref_grid.sample_sdf()
        return src_grid, ref_grid

def load_test(grid, bnd, par_cnt, patch_size, scr, t, positions=None):
    result = np.empty((0,par_cnt,3))

    plot_particles(grid.particles, [0,grid.dimX], [0,grid.dimY], 0.1, (scr+"_not_accum.png")%t)

    sdf_f = sdf_func(np.squeeze(grid.cells))[0]

    if positions is None:
        particle_data_bound = grid.particles[in_bound(grid.particles[:,:2], bnd+patch_size/2, grid.dimX-(bnd+patch_size/2))]
        positions = particle_data_bound[in_surface(np.array([sdf_f(p) for p in particle_data_bound]))[0]]

    plot_particles(positions, [0,grid.dimX], [0,grid.dimY], 0.1, (scr + "_pos.png") % t)

    img = np.empty((0,3))
    i = 0
    for pos in positions:
        par = extract_particles(grid.particles, pos, par_cnt, patch_size/2)[0]
        #sort_idx = sort_particles(par, np.array([sdf_f(p) for p in par]))
        #par = par[sort_idx]

        if [t,i] in samples:
            plot_particles(par, [-1,1], [-1,1], 5, (scr+"_i%03d_patch.png")%(t,i))
        i+=1

        result = np.append(result, [par], axis=0)
        img = np.append(img, np.add(par*patch_size/2, [pos[0], pos[1], 0.]), axis=0)

    plot_particles(img, [0,grid.dimX], [0,grid.dimY], 0.1, (scr + ".png") % t)
    return result, positions

def mean_nor(sdf, positions, t):
    res = np.empty((len(positions),2))
    nor_f = sdf_func(np.squeeze(sdf))[1]
    for i in range(len(positions)):
        res[i] = nor_f(positions[i,:2])
        if [t,i] in samples:
            print(res[i])
    return res

def nor_patches(sdf, positions, patch_size, scr, t):
    res = np.empty((0,patch_size, patch_size, 2))
    ps_half = patch_size/2 - 0.5

    nor_f = sdf_func(np.squeeze(sdf))[1]
    i=0
    img = np.zeros((1,h_dim,h_dim,2))
    for pos in positions:
        tmp = np.array([[[nor_f(pos[:2]-ps_half+np.array([x,y])) for x in range(patch_size)] for y in range(patch_size)]])
        res = np.append(res, tmp, axis=0)

        if [t,i] in samples:
            plot_vec(tmp[0], [0,patch_size], [0,patch_size], (scr+"_i%03d_patch.png")%(t,i))
        i+=1

        if np.all(pos[:2]>ps_half) and np.all(pos[:2]<h_dim-ps_half):
            insert_patch(img, tmp[0], pos.astype(int), elem_avg)

    plot_vec(img[0], [0,h_dim], [0,h_dim], (scr+".png")%t)
    return res

def sdf_patches(sdf, positions, patch_size, scr, t):
    res = np.empty((0,patch_size, patch_size))
    ps_half = patch_size/2 - 0.5

    sdf_f = sdf_func(np.squeeze(sdf))[0]
    i=0
    img = np.ones((1,h_dim,h_dim))*extrapol
    for pos in positions:
        tmp = np.array([[[sdf_f(pos[:2]-ps_half+np.array([x,y]))[0] for x in range(patch_size)] for y in range(patch_size)]])
        res = np.append(res, np.tanh(12.0*tmp), axis=0)

        if [t,i] in samples:
            plot_sdf(tmp[0], [0,patch_size], [0,patch_size], (scr+"_i%03d_patch.png")%(t,i))
        i+=1

        if np.all(pos[:2]>ps_half) and np.all(pos[:2]<h_dim-ps_half):
            insert_patch(img, tmp[0] * circular_filter, pos.astype(int), elem_min)

    plot_sdf(img[0], [0,h_dim], [0,h_dim], (scr+".png")%t)
    return res

def load_src(prefix, bnd, par_cnt, patch_size, scr, t, positions=None):
    result = np.empty((0,par_cnt,3))
    #aux_res = {}
    #aux_data = {}

    particle_data = readParticles(prefix + "_ps.uni")[1]
    '''for k, v in aux.items():
        aux_data[k] = readParticles((prefix+"_p%s.uni")%v, "float32")[1]
        aux_res[k] = np.empty((0,par_cnt, aux_data[k].shape[-1]))
'''
    header, sdf = readUni(prefix + "_sdf.uni")
    sdf_f = sdf_func(np.squeeze(sdf))[0]

    if positions is None:    
        particle_data_bound = particle_data[in_bound(particle_data[:,:2], bnd+patch_size/2,header['dimX']-(bnd+patch_size/2))]
        positions = particle_data_bound[in_surface(np.array([sdf_f(p) for p in particle_data_bound]))[0]]
        
    plot_particles(positions, [0,header['dimX']], [0,header['dimY']], 0.1, (scr + "_pos.png") % t)

    img = np.empty((0,3))
    i = 0
    for pos in positions:
        par = extract_particles(particle_data, pos, par_cnt, patch_size/2)[0]#, aux_data)
        #sort_idx = sort_particles(par, np.array([sdf_f(p) for p in par]))
        #par = par[sort_idx]

        if [t,i] in samples:
            plot_particles(par, [-1,1], [-1,1], 5, (scr+"_i%03d_patch.png")%(t,i))
        i+=1

        result = np.append(result, [par], axis=0)
        '''for k, v in par_aux.items():
            aux_res[k] = np.append(aux_res[k], [v[sort_idx]], axis=0)'''
        img = np.append(img, np.add(par*patch_size/2, [pos[0], pos[1], 0.]), axis=0)

    plot_particles(img, [0,header['dimX']], [0,header['dimY']], 0.1, (scr+".png") % t)
    return result, positions

src_file = "%s_%s-%s"%(data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
dst_file = "%s_%s"%(data_config['prefix'], data_config['id']) + "_d%03d_%03d"

if use_test_data:
    src = np.empty((0,particle_cnt_src,3))
    dst = np.empty((0,particle_cnt_dst,3))

    rotated_src = np.empty((0,particle_cnt_src,3))

    sdf_dst = np.empty((0,ref_patch_size, ref_patch_size, 2) if use_vec else (0,ref_patch_size, ref_patch_size))

    aux_postfix = {
        #"vel":"v"
    }

    aux_src = {}
    for k in aux_postfix:
        aux_src[k] = np.empty((0, particle_cnt_src, 3 if k == "vel" else 1))

    src_gen = RandomParticles(dim,dim,fac_2d,dim/3,obj_cnt,1.0)# if fixed else 0.8)

    data_cnt = var*dataset*(t_end-t_start)*repetitions
    for v in range(var):
        for d in range(dataset):
            for t in range(t_start, t_end):
                src_gen.gen_random(pos=np.array([dim/2+0.5,dim/2+0.5]) if fixed else None)
                for r in range(repetitions):
                    act_d = r+repetitions*((t-t_start)+(t_end-t_start)*(d+v*dataset))
                    print("Generate Data: {}/{}".format(act_d+1,data_cnt), end="\r", flush=True)#,"-"*act_d,"."*(data_cnt-act_d-1)), end="\r", flush=True)   
                    src_data, ref_data = src_gen.get_grid()

                    if trans_mode == 1 or trans_mode == 2:
                        ref_data.particles = translation(ref_data.particles, np.squeeze(ref_data.cells), trans_fac)# if trans_fac > 0 else (-trans_fac / src_gen.a[0,0]))

                    res, positions = load_test(src_data, 0, particle_cnt_src, patch_size, source_scr, t)
                    #res, positions = load_src(data_path + "source/" + src_file%(d,v,t), 4/fac_2d, particle_cnt_src, patch_size, source_scr, t)

                    src = np.append(src, res, axis=0)
                    #for k, val in aux_res.items():
                    #    aux_src[k] = np.append(aux_src[k], val, axis=0)

                    nor = mean_nor(src_data.cells, positions, t)
                    theta = np.arctan2(nor[:,0],nor[:,1])

                    for i in range(len(res)):
                        c, s = np.cos(-theta[i]), np.sin(-theta[i])
                        mat = np.matrix([[c,-s,0],[s,c,0],[0,0,1]])
                        res[i] = res[i] * mat
                        if [t,i] in samples:
                            plot_particles(res[i],[-1,1],[-1,1],5,(source_scr+"_i%03d_rotated_patch.png")%(t,i))
                    rotated_src = np.append(rotated_src, res, axis=0)
                        
                    res = load_test(ref_data, 0, particle_cnt_dst, ref_patch_size, reference_scr, t, positions*fac_2d)[0]
                    #res = load_src(data_path + "reference/" + dst_file%(d,t), 4, particle_cnt_dst, ref_patch_size, reference_scr, t, positions=positions*fac_2d)[0]

                    dst = np.append(dst, res, axis=0)

                    if gen_grid:
                        patch = nor_patches(ref_data.cells, positions*fac_2d, ref_patch_size, vec_reference_scr, t) if use_vec else sdf_patches(ref_data.cells, positions*fac_2d, ref_patch_size, sdf_reference_scr, t)
                        sdf_dst = np.append(sdf_dst, patch, axis=0)
else:
    src, dst, rotated_src = gen_patches(data_path, config_path, dataset, t_end, var, repetitions, t_start=t_start)[:3]

fac = 16
k = 256
dropout = 0.2
batch_size = train_config['batch_size']
epochs = train_config['epochs']

'''src = src[:len(src)//batch_size*batch_size]
dst = dst[:len(src)//batch_size*batch_size]
rotated_src = rotated_src[:len(src)//batch_size*batch_size]

sdf_dst = sdf_dst[:len(src)//batch_size*batch_size]

for k, v in aux_src.items():
    aux_src[k] = v[:len(src)//batch_size*batch_size]'''

inputs = Input((particle_cnt_src,3), name="main")
#aux_input = Input((particle_cnt_src,3))

stn_input = Input((particle_cnt_src,3))
stn = SpatialTransformer(stn_input,particle_cnt_src,dropout=dropout,quat=True,norm=True)
stn_model = Model(inputs=stn_input, outputs=stn)
stn = stn_model(inputs)

trans_input = stn_transform(stn,inputs,quat=True)
inter_model = Model(inputs=inputs, outputs=trans_input)
inter_model.compile(loss=hungarian_loss, optimizer=keras.optimizers.adam(lr=0.001))

if pre_train_stn:
    history = inter_model.fit(x=src,y=rotated_src,epochs=epochs,batch_size=batch_size)
    stn_model.trainable = False

#x = concatenate([intermediate, stn([x,aux_input])],axis=-1)

x = [(Lambda(lambda v: v[:,i:i+1,:])(trans_input)) for i in range(particle_cnt_src)]

x = split_layer(Dropout(dropout),x)
x = split_layer(Dense(fac, activation='tanh'),x)
x = split_layer(Dropout(dropout),x)
x = split_layer(Dense(fac, activation='tanh'),x)

x = concatenate(x, axis=1)
x = stn_transform(SpatialTransformer(x,particle_cnt_src,fac,1),x)

x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_src)]

x = split_layer(Dropout(dropout),x)
x = split_layer(Dense(fac, activation='tanh'),x)
x = split_layer(Dropout(dropout),x)
x = split_layer(Dense(fac*2, activation='tanh'),x)
x = split_layer(Dropout(dropout),x)
x = split_layer(Dense(k, activation='tanh'),x)

features = add(x)

if gen_grid or use_vec:
    x = Flatten()(features)
    x = Dropout(dropout)(x)

    x = Dense(ref_patch_size*ref_patch_size*16)(x)
    x = Reshape((ref_patch_size,ref_patch_size,16))(x)
    
    x = Conv2DTranspose(filters=8, kernel_size=3, strides=1, activation='tanh', padding='same')(x)
    x = Conv2DTranspose(filters=4, kernel_size=3, strides=1, activation='tanh', padding='same')(x)

    if use_vec:
        w = [np.zeros((3,3,2,4)),np.zeros((2,))]
        x = Conv2DTranspose(filters=2, kernel_size=3, strides=1, activation='tanh', padding='same', weights=w)(x)
        inv_grid_out = Reshape((ref_patch_size, ref_patch_size, 2))(x)

        # use grid for residual approach
        if not gen_grid:
            def tmp(v):
                sh = tf.shape(v[1])
                zero = tf.zeros((sh[0],sh[1],1))
                return K.concatenate([interpol(v[0],v[1]),zero],axis=-1)
            x = Lambda(tmp)([inv_grid_out,trans_input])
            inv_par_out = add([trans_input, x])
            out = stn_transform_inv(stn, inv_par_out, quat=True)

        # output vec grid
        def tmp(v):
            sh = tf.shape(v)
            zero = tf.zeros((sh[0], sh[1], sh[2], 1))
            return K.concatenate([v,zero])
        x = Lambda(tmp)(inv_grid_out)
        out_sdf = Lambda(lambda _x: tf.split(stn_grid_transform_inv(stn, _x, quat=True), [2,1], -1)[0])(x)
    else:
        w = [np.zeros((3,3,1,4)),np.zeros((1,))]
        x = Conv2DTranspose(filters=1, kernel_size=3, strides=1, activation='tanh', padding='same', weights=w)(x)
        inv_grid_out = Reshape((ref_patch_size, ref_patch_size))(x)
        out_sdf = stn_grid_transform_inv(stn, inv_grid_out, quat=True)

if par_out and not use_vec:
    x = Flatten()(features)
    x = Dropout(dropout)(x)
    x = Dense(fac*8, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*4, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(3*particle_cnt_dst, activation='tanh')(x)

    inv_par_out = Reshape((particle_cnt_dst,3))(x)
    out = stn_transform_inv(stn,inv_par_out,quat=True)

loss = []
y = []
m_out = []
invm_out = []

if par_out:
    loss.append(hungarian_loss)#emd_loss)
    y.append(dst)
    m_out.append(out)
    invm_out.append(inv_par_out)
if grid_out:
    loss.append('mse')
    y.append(sdf_dst)
    m_out.append(out_sdf)
    invm_out.append(inv_grid_out)
elif use_vec:
    vec_model = Model(inputs=inputs, outputs=out_sdf)
    invm_out.append(inv_grid_out)

model = Model(inputs=inputs, outputs=m_out)
invm = Model(inputs=inputs, outputs=invm_out)

if train_config["adv_fac"] > 0.:
    disc_input = Input((particle_cnt_src,3))
    x = stn_transform(stn_model(disc_input) ,disc_input,quat=True)

    x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_src)]

    x = split_layer(Dropout(dropout),x)
    x = split_layer(Dense(fac, activation='tanh'),x)
    x = split_layer(Dropout(dropout),x)
    x = split_layer(Dense(fac, activation='tanh'),x)

    x = concatenate(x, axis=1)
    x = stn_transform(SpatialTransformer(x,particle_cnt_src,fac,1),x)

    x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_src)]

    x = split_layer(Dropout(dropout),x)
    x = split_layer(Dense(fac, activation='tanh'),x)
    x = split_layer(Dropout(dropout),x)
    x = split_layer(Dense(fac*2, activation='tanh'),x)
    x = split_layer(Dropout(dropout),x)
    x = split_layer(Dense(k, activation='tanh'),x)

    x = add(x)

    x = Flatten()(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*16, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*8, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*4, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid')(x)

    '''disc_input = Input((particle_cnt_src,3), name="main")
    x = Flatten()(disc_input)
    x = Dropout(dropout)(x)
    x = Dense(fac*32, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*16, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*8, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*4, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid')(x)'''

    discriminator = Model(inputs=disc_input, outputs=x)
    discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=train_config["learning_rate"]), metrics=['accuracy'])

if not grid_out and gen_grid:
    sdf_in = Input((ref_patch_size, ref_patch_size))
    sample_out = Lambda(lambda v: K.relu(interpol(v[0],(v[1]+1)*ref_patch_size*0.5)))([sdf_in, out])

    train_model = Model(inputs=[inputs, sdf_in], outputs=[out,sample_out])
    train_model.compile(loss=[hungarian_loss, lambda x,y: K.mean(y,axis=-1)], optimizer=keras.optimizers.adam(lr=0.001), loss_weights=[1., 1.])
    history = train_model.fit(x=[src,sdf_dst],y=[dst,np.empty((dst.shape[0],dst.shape[1]))],epochs=epochs,batch_size=batch_size)
elif train_config["adv_fac"] <= 0.:
    model.compile(loss=loss, optimizer=keras.optimizers.adam(lr=0.001))
    history = model.fit(x=src,y=y,epochs=epochs,batch_size=batch_size)
else:
    # GAN
    z = [Input(shape=(particle_cnt_src,3), name='main')]
    img = model(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The valid takes generated images as input and determines validity
    valid = discriminator(img)

    # The combined model  (stacked generator and discriminator)
    combined = Model(z, [img,valid])
    combined.compile(loss=[loss[0],discriminator.loss_functions[0]], optimizer=keras.optimizers.adam(lr=train_config['learning_rate']),
                    loss_weights=[train_config['mse_fac'], train_config['adv_fac']])

    half_batch = batch_size//2

    train_cnt = int(len(src)*(1-train_config['val_split']))//batch_size*batch_size
    print('train count: %d' % train_cnt)
    eval_cnt = int(len(src)*train_config['val_split'])//batch_size*batch_size
    print('eval count: %d' % eval_cnt)

    cnt_inv = batch_size/train_cnt

    history = {'d_loss':[],'d_acc':[],'g_loss':[],'g_mse':[],'g_adv_loss':[],
            'd_val_loss':[],'d_val_acc':[],'g_val_loss':[],'g_val_mse':[],'g_val_adv_loss':[]}
    idx0 = np.arange(train_cnt+eval_cnt)
    idx1 = np.arange(train_cnt+eval_cnt)

    np.random.shuffle(idx0)
    np.random.shuffle(idx1)
        
    idx0, val_idx0 = np.split(idx0,[train_cnt])
    idx1, val_idx1 = np.split(idx1,[train_cnt])

    for ep in range(epochs):    
        # train
        np.random.shuffle(idx0)
        np.random.shuffle(idx1)
        g_loss = [0.,0.,0.]
        d_loss = [0.,0.]
        
        for i in range(0,train_cnt,batch_size):
            print("Train epoch {}, batch {}/{}".format(ep+1, i+batch_size, train_cnt), end="\r", flush=True)
            x = src[idx0[i:i+half_batch]]
            y = dst[idx0[i+half_batch:i+batch_size]]
            x = model.predict(x)

            d_loss_fake = discriminator.train_on_batch(x, np.zeros((half_batch, 1)))
            d_loss_real = discriminator.train_on_batch(y, np.ones((half_batch, 1)))
            d_loss = np.add(d_loss, cnt_inv * 0.5 * np.add(d_loss_real, d_loss_fake) )
            
            x = src[idx1[i:i+batch_size]]
            y = dst[idx1[i:i+batch_size]]
            g_loss = np.add(g_loss, cnt_inv * np.array(combined.train_on_batch(x, [y, np.ones((batch_size, 1))])))
        
        print("\r", flush=True)
        # eval
        np.random.shuffle(val_idx0)
        np.random.shuffle(val_idx1)
        g_val_loss = [0.,0.,0.]
        d_val_loss = [0.,0.]
    
        x = src[val_idx0]
        y = dst[val_idx0]
        x = model.predict(x)
        
        d_loss_fake = discriminator.evaluate(x, np.zeros((eval_cnt, 1)), batch_size=half_batch, verbose=0)
        d_loss_real = discriminator.evaluate(y, np.ones((eval_cnt, 1)), batch_size=half_batch, verbose=0)
        d_val_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        x = src[val_idx1]
        y = dst[val_idx1]
        g_val_loss = combined.evaluate(x, [y,np.ones((eval_cnt, 1))], batch_size=batch_size, verbose=0)
        
        history['d_loss'].append(d_loss[0])
        history['d_acc'].append(d_loss[1])
        history['d_val_loss'].append(d_val_loss[0])
        history['d_val_acc'].append(d_val_loss[1])
        history['g_loss'].append(g_loss[0])
        history['g_mse'].append(g_loss[1])
        history['g_adv_loss'].append(g_loss[2])
        history['g_val_loss'].append(g_val_loss[0])
        history['g_val_mse'].append(g_val_loss[1])
        history['g_val_adv_loss'].append(g_val_loss[2])
                
        print ("epoch %i" % (ep+1))
        print ("\ttrain: [D loss: %f, acc.: %.2f%%] [G loss: %f, mse: %f, adv: %f]" % (d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1], g_loss[2]))
        print ("\teval.: [D loss: %f, acc.: %.2f%%] [G loss: %f, mse: %f, adv: %f]" % (d_val_loss[0], 100*d_val_loss[1], g_val_loss[0], g_val_loss[1], g_val_loss[2]))

#model.summary()
'''plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

prefix = "test"

plt.savefig("%s/loss.png"%prefix)
plt.clf()'''
#src_gen = RandomParticles(dim,dim,fac_2d,dim/3,10,1.0)
data_cnt = (t_end-t_start)*repetitions
#kmeans = KMeans(n_clusters=10)
for v in range(1):
    for d in range(5,6):
        for t in range(t_start, t_end): 
            if use_test_data:
                src_gen.gen_random(pos=np.array([[dim/2+0.5,dim/2+0.5]]) if fixed else None, a=np.array([[t+1,t+1]]) if fixed else None)
            for r in range(repetitions):
                act_d = r+repetitions*(t-t_start)
                print("Run Test: {}/{}".format(act_d+1,data_cnt), end="\r", flush=True)#,"-"*act_d,"."*(data_cnt-act_d-1)), end="\r", flush=True)   

                if use_test_data:
                    src_data, ref_data = src_gen.get_grid()

                    if trans_mode == 1 or trans_mode == 2:
                        ref_data.particles = translation(ref_data.particles, np.squeeze(ref_data.cells), trans_fac)# if trans_fac > 0 else (-trans_fac / src_gen.a[0,0]))
                
                    src, positions = load_test(src_data, 0, particle_cnt_src, patch_size, test_source_scr, t)
                    ref = load_test(ref_data, 0, particle_cnt_dst, ref_patch_size, test_reference_scr, t, positions*fac_2d)[0]
                else:
                    src, ref, rot_src, positions = gen_patches(data_path, config_path, d+1, t+1, 1, 1, d, t)

                if gen_grid:
                    sdf_ref = nor_patches(ref_data.cells, positions*fac_2d, ref_patch_size, vec_test_reference_scr, t) if use_vec else sdf_patches(ref_data.cells, positions*fac_2d, ref_patch_size, sdf_test_reference_scr, t)
                
                result = model.predict(x=src,batch_size=batch_size)
                grid_result = result

                inv_result = invm.predict(x=src,batch_size=batch_size)
                inv_grid_result = inv_result

                if grid_out and par_out:
                    grid_result = result[1]
                    result = result[0]
                
                if (grid_out or use_vec) and par_out:
                    inv_grid_result = inv_result[1]
                    inv_result = inv_result[0]

                inter_result = inter_model.predict(x=src,batch_size=batch_size)

                for sample in samples:
                    if sample[0] == t and sample[1] < len(src):
                        print(Model(inputs=inputs, outputs=stn).predict(x=src[sample[1]:sample[1]+1],batch_size=1))

                if use_vec:
                    ps_half = ref_patch_size//2
                    res_output = grid_result if grid_out else vec_model.predict(x=src,batch_size=batch_size)
                    img = np.zeros((1,h_dim,h_dim,2))
                    ref_img = np.zeros((1,h_dim,h_dim,2)) if gen_grid else None
                    src_img = np.empty((0,3))

                    for i in range(len(res_output)):
                        tmp = res_output[i]
                        tmp_ref = sdf_ref[i] if gen_grid else None
                        pos = positions[i]*fac_2d
                        if [t,i] in samples:
                            plot_vec(tmp, [0,ref_patch_size], [0,ref_patch_size], (vec_test_result_scr+"_i%03d_patch.png")%(t,i), tmp_ref, (src[i]+1)*ref_patch_size/2, 5)
                            plot_vec(inv_grid_result[i], [0,ref_patch_size], [0,ref_patch_size], (vec_test_result_scr+"_i%03d_inv_patch.png")%(t,i))
                            plot_particles(inter_result[i], [-1,1], [-1,1], 5, (test_result_scr+"_i%03d_inter_patch.png")%(t,i))

                        if np.all(pos[:2]>ps_half) and np.all(pos[:2]<h_dim-ps_half):
                            insert_patch(img, tmp, pos.astype(int), elem_avg)
                            if gen_grid: insert_patch(ref_img, tmp_ref, pos.astype(int), elem_avg)
                        src_img = np.append(src_img, np.add(src[i]*ref_patch_size/2, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.]), axis=0)
                    plot_vec(img[0], [0,h_dim], [0,h_dim], (vec_test_result_scr+"_r%03d.png")%(t,r), ref_img[0] if gen_grid else None, src_img, 0.1)
                elif grid_out:
                    ps_half = ref_patch_size//2
                    img = np.ones((1,h_dim,h_dim))*extrapol
                    ref_img = np.ones((1,h_dim,h_dim))*extrapol
                    src_img = np.empty((0,3))

                    for i in range(len(grid_result)):
                        tmp = np.arctanh(np.clip(grid_result[i],-.999999999999,.999999999999)) / 12.0
                        tmp_ref = np.arctanh(np.clip(sdf_ref[i],-.999999999999,.999999999999)) / 12.0
                        pos = positions[i]*fac_2d
                        if [t,i] in samples:
                            plot_sdf(tmp, [0,ref_patch_size], [0,ref_patch_size], (sdf_test_result_scr+"_i%03d_patch.png")%(t,i), tmp_ref, (src[i]+1)*ref_patch_size/2, 5)
                            plot_sdf(inv_grid_result[i], [0,ref_patch_size], [0,ref_patch_size], (sdf_test_result_scr+"_i%03d_inv_patch.png")%(t,i))
                            plot_particles(inter_result[i], [-1,1], [-1,1], 5, (test_result_scr+"_i%03d_inter_patch.png")%(t,i))

                        if np.all(pos[:2]>ps_half) and np.all(pos[:2]<h_dim-ps_half):
                            insert_patch(img, tmp * circular_filter, pos.astype(int), elem_min)
                            insert_patch(ref_img, tmp_ref, pos.astype(int), elem_min)
                        src_img = np.append(src_img, np.add(src[i]*ref_patch_size/2, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.]), axis=0)
                    plot_sdf(img[0], [0,h_dim], [0,h_dim], (sdf_test_result_scr+"_r%03d.png")%(t,r), ref_img[0], src_img, 0.1)
                
                if par_out:
                    img = np.empty((0,3))
                    ref_img = np.empty((0,3))
                    src_img = np.empty((0,3))

                    for i in range(len(result)):
                        #kmeans.fit(result[i,:,:2])
                        #par = np.append(kmeans.cluster_centers_, np.zeros((10,1)), axis=1)
                        par = result[i]#,:10]
                        if [t,i] in samples:# or True:
                            plot_particles(par, [-1,1], [-1,1], 5, (test_result_scr+"_i%03d_patch.png")%(t,i), ref[i], src[i])
                            plot_particles(inv_result[i], [-1,1], [-1,1], 5, (test_result_scr+"_i%03d_inv_patch.png")%(t,i))
                            plot_particles(inter_result[i], [-1,1], [-1,1], 5, (test_result_scr+"_i%03d_inter_patch.png")%(t,i))
                        
                        img = np.append(img, np.add(par*ref_patch_size/2, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.]), axis=0)
                        ref_img = np.append(ref_img, np.add(ref[i]*ref_patch_size/2, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.]), axis=0)
                        src_img = np.append(src_img, np.add(src[i]*ref_patch_size/2, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.]), axis=0)
                    plot_particles(img, [0,h_dim], [0,h_dim], 0.1, (test_result_scr+"_r%03d.png")%(t,r), ref_img, src_img)

            
