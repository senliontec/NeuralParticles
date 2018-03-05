import sys, os, warnings
sys.path.append("manta/scenes/tools")
sys.path.append("hungarian/")

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

random.seed(235)
np.random.seed(694)

import keras.backend as K
import tensorflow as tf

from sklearn.decomposition import PCA

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
    x,y = np.gradient(sdf)
    x = np.expand_dims(x,axis=-1)
    y = np.expand_dims(y,axis=-1)
    g = np.concatenate([y,x],axis=-1)
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
    nor_f = lambda x: np.concatenate([interpolate.interp2d(x_v, y_v, nor[:,:,1])(x[0],x[1]), interpolate.interp2d(x_v, y_v, nor[:,:,0])(x[0],x[1])])
    '''for i in np.arange(sdf.shape[0],step=2.0):
        for j in np.arange(sdf.shape[1],step=2.0):
            v = sdf_f([i,j])
            if v < 0.0:
                n = nor_f([i,j])
                #plt.plot(i,j,'bo')
                plt.plot([i,i+n[0]],[j,j+n[1]], '.-')
    plt.xlim([0,sdf.shape[0]])
    plt.ylim([0,sdf.shape[1]])
    plt.show()
    exit()'''
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
manta_path = getParam("manta", "manta/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0

dataset = int(getParam("dataset", 0, paramUsed))
var = int(getParam("var", 0, paramUsed))

trans_mode = int(getParam("trans_mode", 0, paramUsed))
# if smaller then 0 use curvature!
trans_fac = float(getParam("trans_fac", 10.0, paramUsed))
repetitions = int(getParam("rep", 10, paramUsed))
obj_cnt = int(getParam("obj_cnt", 10, paramUsed))
fixed = int(getParam("fixed", 0, paramUsed)) != 0

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

patch_size = 9#pre_config['patch_size']
fac_1d = 9
fac_2d = 3
ref_patch_size = patch_size * fac_2d
stride = 1
surface = 1.0
particle_cnt_src = 100 #pre_config['par_cnt']
particle_cnt_dst = 100

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

def translation(par, nor, fac):
    res = np.empty((0,3))
    x_v = np.arange(0.5, nor.shape[1]+0.5)
    y_v = np.arange(0.5, nor.shape[0]+0.5)
    nor_f = lambda x: np.concatenate([interpolate.interp2d(x_v, y_v, nor[:,:,1])(x[0],x[1]), interpolate.interp2d(x_v, y_v, nor[:,:,0])(x[0],x[1])])
    #if fac <= 0:
        #curv = lambda x: interpolate.interp2d(x_v, y_v, curvature(nor))(x[0],x[1])
    for p in par:
        n = (nor_f(p[:2]) if trans_mode == 1 else np.array([1,0])) * fac#(fac if fac > 0 else ((-fac) * curv(p[:2])))
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
                ref_grid.sample_sphere(self.pos[i] * self.fac_2d, self.a[i,0] * self.fac_2d)
                #ref_grid.sample_cos_sphere(self.pos[i] * self.fac_2d, self.a[i,0] * self.fac_2d, 6, 3)
        src_grid.sample_sdf()
        ref_grid.sample_sdf()
        return src_grid, ref_grid

def load_test(grid, bnd, par_cnt, patch_size, scr, t, positions=None):
    result = np.empty((0,par_cnt,3))
    particle_data_nb = grid.particles[in_bound(grid.particles[:,:2], bnd, grid.dimX-bnd)]

    plot_particles(particle_data_nb, [0,grid.dimX], [0,grid.dimY], 0.1, (scr+"_not_accum.png")%t)

    sdf_f, nor_f = sdf_func(np.squeeze(grid.cells))

    if positions is None:
        positions = particle_data_nb[in_surface(np.array([sdf_f(p) for p in particle_data_nb]))[0]]

    plot_particles(positions, [0,grid.dimX], [0,grid.dimY], 0.1, (scr + "_pos.png") % t)

    img = np.empty((0,3))
    i = 0
    for pos in positions:
        par = extract_particles(particle_data_nb, pos, par_cnt, patch_size/2)[0]
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

def load_src(prefix, bnd, par_cnt, patch_size, scr, t, aux={}, positions=None):
    result = np.empty((0,par_cnt,3))
    aux_res = {}
    aux_data = {}

    particle_data = readParticles(prefix + "_ps.uni")[1]
    for k, v in aux.items():
        aux_data[k] = readParticles((prefix+"_p%s.uni")%v, "float32")[1]
        aux_res[k] = np.empty((0,par_cnt, aux_data[k].shape[-1]))

    header, sdf = readUni(prefix + "_sdf.uni")
    sdf_f, nor_f = sdf_func(np.squeeze(sdf))

    bnd_idx = in_bound(particle_data[:,:2], bnd,header['dimX']-bnd)
    particle_data_nb = particle_data[bnd_idx]
    if positions is None:
        positions = particle_data_nb[in_surface(np.array([sdf_f(p) for p in particle_data_nb]))[0]]
    
    img = np.empty((0,3))
    i = 0
    for pos in positions:
        par, par_aux = extract_particles(particle_data_nb, pos, par_cnt, patch_size/2, aux_data)
        sort_idx = sort_particles(par, np.array([sdf_f(p) for p in par]))
        par = par[sort_idx]

        if [t,i] in samples:
            plot_particles(par, [-1,1], [-1,1], 5, (scr+"_i%03d_patch.png")%(t,i))
        i+=1

        result = np.append(result, [par], axis=0)
        for k, v in par_aux.items():
            aux_res[k] = np.append(aux_res[k], [v[sort_idx]], axis=0)
        img = np.append(img, np.add(par*patch_size/2, [pos[0], pos[1], 0.]), axis=0)

    plot_particles(img, [0,header['dimX']], [0,header['dimY']], 0.1, (scr+".png") % t)

    return result, aux_res, positions

src_file = "%s_%s-%s"%(data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
dst_file = "%s_%s"%(data_config['prefix'], data_config['id']) + "_d%03d_%03d"

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

                if trans_mode > 0:
                    ref_data.particles = translation(ref_data.particles, normals(np.squeeze(ref_data.cells)), trans_fac if trans_fac > 0 else (-trans_fac / src_gen.a[0,0]))

                res, positions = load_test(src_data, 4/fac_2d, particle_cnt_src, patch_size, source_scr, t)
                #res, aux_res, positions = load_src(data_path + "source/" + src_file%(d,v,t), 4/fac_2d, particle_cnt_src, patch_size, l_scr+"_patch.png", "test/source_%03d.png", t, aux_postfix)

                src = np.append(src, res, axis=0)
                #for k, val in aux_res.items():
                #    aux_src[k] = np.append(aux_src[k], val, axis=0)

                nor = mean_nor(ref_data.cells, positions*fac_2d, t)
                theta = np.arctan2(nor[:,0],nor[:,1])# / math.pi * 180

                for i in range(len(res)):
                    c, s = np.cos(-theta[i]), np.sin(-theta[i])
                    mat = np.matrix([[c,-s,0],[s,c,0],[0,0,1]])
                    res[i] = res[i] * mat
                    if [t,i] in samples:
                        plot_particles(res[i],[-1,1],[-1,1],5,(source_scr+"_i%03d_rotated_patch.png")%(t,i))
                rotated_src = np.append(rotated_src, res, axis=0)
                    
                res = load_test(ref_data, 4, particle_cnt_dst, ref_patch_size, reference_scr, t, positions*fac_2d)[0]
                #res = load_src(data_path + "reference/" + dst_file%(d,t), 4, particle_cnt_dst, ref_patch_size, h_scr+"_patch.png", "test/reference_%03d.png", t, positions=positions*fac_2d)[0]

                dst = np.append(dst, res, axis=0)

                if gen_grid:
                    patch = nor_patches(ref_data.cells, positions*fac_2d, ref_patch_size, vec_reference_scr, t) if use_vec else sdf_patches(ref_data.cells, positions*fac_2d, ref_patch_size, sdf_reference_scr, t)
                    sdf_dst = np.append(sdf_dst, patch, axis=0)


fac = 16
k = 256
dropout = 0.2
batch_size = train_config['batch_size']
epochs = 3 # train_config['epochs']

src = src[:len(src)//batch_size*batch_size]
dst = dst[:len(src)//batch_size*batch_size]
rotated_src = rotated_src[:len(src)//batch_size*batch_size]

sdf_dst = sdf_dst[:len(src)//batch_size*batch_size]

for k, v in aux_src.items():
    aux_src[k] = v[:len(src)//batch_size*batch_size]

inputs = Input((particle_cnt_src,3), name="main")
#aux_input = Input((particle_cnt_src,3))

stn_input = Input((particle_cnt_src,3))
stn = SpatialTransformer(stn_input,particle_cnt_src,dropout=dropout,quat=True,norm=True)
stn_model = Model(inputs=stn_input, outputs=stn)
stn = stn_model(inputs)

x = stn_transform(stn,inputs,quat=True)
inter_model = Model(inputs=inputs, outputs=x)
inter_model.compile(loss=hungarian_loss, optimizer=keras.optimizers.adam(lr=0.001))
history = inter_model.fit(x=src,y=rotated_src,epochs=epochs,batch_size=batch_size)

stn_model.trainable = False

#x = inputs#Dropout(dropout)(inputs)
#stn = SpatialTransformer(x,particle_cnt_src,dropout=dropout,quat=True,norm=True)
#intermediate = stn_transform(stn,x,quat=True)

#x = concatenate([intermediate, stn([x,aux_input])],axis=-1)

x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_src)]

x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(fac, activation='tanh'))(x)
x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(fac, activation='tanh'))(x)

x = concatenate(x, axis=1)
x = stn_transform(SpatialTransformer(x,particle_cnt_src,fac,1),x)

x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_src)]

x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(fac, activation='tanh'))(x)
x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(fac*2, activation='tanh'))(x)
x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(k, activation='tanh'))(x)

features = add(x)

if gen_grid or use_vec:
    x = Flatten()(features)
    x = Dropout(dropout)(x)

    c = 2 if use_vec else 1

    x = Dense(ref_patch_size*ref_patch_size*16)(x)
    x = Reshape((ref_patch_size,ref_patch_size,16))(x)
    
    x = Conv2DTranspose(filters=8, kernel_size=3, strides=1, activation='tanh', padding='same')(x)
    x = Conv2DTranspose(filters=4, kernel_size=3, strides=1, activation='tanh', padding='same')(x)

    w = [np.zeros((3,3,c,4)),np.zeros((c,))]
    x = Conv2DTranspose(filters=c, kernel_size=3, strides=1, activation='tanh', padding='same', weights=w)(x)

    if use_vec:
        out_sdf = Reshape((ref_patch_size, ref_patch_size, c))(x)
        if not gen_grid:
            def tmp(v):
                sh = tf.shape(v[1])
                zero = tf.zeros((sh[0],sh[1],1))
                return K.concatenate([interpol(v[0],v[1]),zero],axis=-1)
            x = Lambda(tmp)([out_sdf,inputs])
            x = add([inputs, x])
            out = stn_transform_inv(stn, x, quat=True)
    else:
        x = Reshape((ref_patch_size, ref_patch_size))(x)
        inv_trans = x
        out_sdf = stn_grid_transform_inv(stn, x, quat=True)

if par_out and not use_vec:
    x = Flatten()(features)
    x = Dropout(dropout)(x)
    x = Dense(fac*8, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*4, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(3*particle_cnt_dst, activation='tanh')(x)

    x = Reshape((particle_cnt_dst,3))(x)
    inv_trans = x
    out = stn_transform_inv(stn,x,quat=True)

loss = []
y = []
m_out = []

if par_out:
    loss.append(hungarian_loss)
    y.append(dst)
    m_out.append(out)
if grid_out:
    loss.append('mse')
    y.append(sdf_dst)
    m_out.append(out_sdf)
elif use_vec:
    vec_model = Model(inputs=inputs, output=out_sdf)

model = Model(inputs=inputs, outputs=m_out)
#interm = Model(inputs=inputs, outputs=intermediate)

if not grid_out and gen_grid:
    sdf_in = Input((ref_patch_size, ref_patch_size))
    sample_out = Lambda(lambda v: K.relu(interpol(v[0],(v[1]+1)*ref_patch_size*0.5)))([sdf_in, out])

    train_model = Model(inputs=[inputs, sdf_in], output=[out,sample_out])
    train_model.compile(loss=[hungarian_loss, lambda x,y: K.mean(y,axis=-1)], optimizer=keras.optimizers.adam(lr=0.001), loss_weights=[1., 1.])
    history = train_model.fit(x=[src,sdf_dst],y=[dst,np.empty((dst.shape[0],dst.shape[1]))],epochs=epochs,batch_size=batch_size)
else:
    model.compile(loss=loss, optimizer=keras.optimizers.adam(lr=0.001))
    history = model.fit(x=src,y=y,epochs=epochs,batch_size=batch_size)

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
            src_gen.gen_random(pos=np.array([[dim/2+0.5,dim/2+0.5]]) if fixed else None, a=np.array([[t+1,t+1]]) if fixed else None)
            for r in range(repetitions):
                act_d = r+repetitions*(t-t_start)
                print("Run Test: {}/{}".format(act_d+1,data_cnt), end="\r", flush=True)#,"-"*act_d,"."*(data_cnt-act_d-1)), end="\r", flush=True)   

                src_data, ref_data = src_gen.get_grid()

                if trans_mode > 0:
                    ref_data.particles = translation(ref_data.particles, normals(np.squeeze(ref_data.cells)), trans_fac if trans_fac > 0 else (-trans_fac / src_gen.a[0,0]))
            
                src, positions = load_test(src_data, 4/fac_2d, particle_cnt_src, patch_size, test_source_scr, t)

                ref = load_test(ref_data, 4, particle_cnt_dst, ref_patch_size, test_reference_scr, t, positions*fac_2d)[0]

                if gen_grid:
                    sdf_ref = nor_patches(ref_data.cells, positions*fac_2d, ref_patch_size, vec_test_reference_scr, t) if use_vec else sdf_patches(ref_data.cells, positions*fac_2d, ref_patch_size, sdf_test_reference_scr, t)
                
                #src, aux_src, positions = load_src(data_path + "source/" + src_file%(d,v,t), 4/fac_2d, particle_cnt_src, patch_size, t_scr+"_patch.png", "test/test_%03d.png", t, aux_postfix)

                result = model.predict(x=src,batch_size=batch_size)
                grid_result = result
                if grid_out and par_out:
                    grid_result = result[1]
                    result = result[0]
                inter_result = inter_model.predict(x=src,batch_size=batch_size)
                #inv_result = invm.predict(x=src,batch_size=batch_size)

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
                            plot_particles(inter_result[i], [-1,1], [-1,1], 5, (test_result_scr+"_i%03d_inter_patch.png")%(t,i))
                        
                        img = np.append(img, np.add(par*ref_patch_size/2, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.]), axis=0)
                        ref_img = np.append(ref_img, np.add(ref[i]*ref_patch_size/2, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.]), axis=0)
                        src_img = np.append(src_img, np.add(src[i]*ref_patch_size/2, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.]), axis=0)
                    plot_particles(img, [0,h_dim], [0,h_dim], 0.1, (test_result_scr+"_r%03d.png")%(t,r), ref_img, src_img)

            
