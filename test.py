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

from hungarian_loss import HungarianLoss

from particle_grid import ParticleGrid

import scipy.ndimage.filters as fi

random.seed(235)
np.random.seed(694)

import keras.backend as K
import tensorflow as tf

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
    g = np.concatenate([x,y],axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nan_to_num(g/np.linalg.norm(g,axis=-1,keepdims=True))

def curvature(nor):
    dif = np.gradient(nor)
    return (np.linalg.norm(dif[0],axis=-1)+np.linalg.norm(dif[1],axis=-1))/2

def sdf_func(sdf):
    x_v = np.arange(0.5, sdf.shape[0]+0.5)
    y_v = np.arange(0.5, sdf.shape[1]+0.5)
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

data_path = getParam("data", "data/", paramUsed)
manta_path = getParam("manta", "manta/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0

dataset = int(getParam("dataset", 0, paramUsed))
var = int(getParam("var", 0, paramUsed))

use_sdf = int(getParam("use_sdf", 0, paramUsed)) != 0
trans_mode = int(getParam("trans_mode", 0, paramUsed))
# if smaller then 0 use curvature!
trans_fac = float(getParam("trans_fac", 10.0, paramUsed))
repetitions = int(getParam("rep", 10, paramUsed))
obj_cnt = int(getParam("obj_cnt", 10, paramUsed))
fixed = int(getParam("fixed", 0, paramUsed)) != 0

l_scr = getParam("l_scr", "test/source_t%03d", paramUsed)
h_scr = getParam("h_scr", "test/reference_t%03d", paramUsed)
t_scr = getParam("t_scr", "test/test_src_t%03d", paramUsed)
h_t_scr = getParam("h_t_scr", "test/test_ref_t%03d", paramUsed)
r_scr = getParam("r_scr", "test/result_t%03d", paramUsed)
sdf_scr = getParam("sdf_scr", "test/sdf_t%03d", paramUsed)
sdf_t_scr = getParam("sdf_t_scr", "test/test_sdf_t%03d", paramUsed)

dim = int(getParam("dim", 50, paramUsed))

patch_sample_cnt = int(getParam("samples", 3, paramUsed))

t_start = int(getParam("t_start", -1, paramUsed))
t_end = int(getParam("t_end", -1, paramUsed))

nor_out = False
sdf_loss = False

checkUnusedParam(paramUsed)

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

patch_size = 5#pre_config['patch_size']
fac = 9
fac_2d = 3
ref_patch_size = patch_size * fac_2d
stride = 1
surface = 1.0
particle_cnt_src = 100 #pre_config['par_cnt']
particle_cnt_dst = 100

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
    x_v = np.arange(0.5, nor.shape[0]+0.5)
    y_v = np.arange(0.5, nor.shape[1]+0.5)
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
        self.pos = np.random.random((self.cnt,2))*np.array([self.dimX+1, self.dimY+1]) if pos is None else pos
        self.cube = np.random.random((self.cnt,)) > self.cube_prob if cube is None else cube
        self.a = 1+np.random.random((self.cnt,2))*(self.max_size-1) if a is None else a

    def get_grid(self):
        src_grid = ParticleGrid(self.dimX, self.dimY, 2)
        ref_grid = ParticleGrid(self.dimX*self.fac_2d, self.dimY*self.fac_2d, 2)
        for i in range(self.cnt):
            if self.cube[i]:
                src_grid.sample_quad(self.pos[i], self.a[i,0], self.a[i,1])
                ref_grid.sample_quad(self.pos[i] * self.fac_2d, self.a[i,0] * self.fac_2d, self.a[i,1] * self.fac_2d)
            else:
                src_grid.sample_sphere(self.pos[i], self.a[i,0])
                ref_grid.sample_sphere(self.pos[i] * self.fac_2d, self.a[i,0] * self.fac_2d)
                #ref_grid.sample_cos_sphere(self.pos[i] * self.fac_2d, self.a[i,0] * self.fac_2d, 6, 3)
        return src_grid, ref_grid

def load_test(grid, bnd, par_cnt, patch_size, scr, t, positions=None):
    result = np.empty((0,par_cnt,3))
    particle_data_nb = grid.particles[in_bound(grid.particles[:,:2], bnd, grid.dimX-bnd)]

    plt.scatter(particle_data_nb[:,0],particle_data_nb[:,1],s=0.1)
    plt.xlim([0,grid.dimX])
    plt.ylim([0,grid.dimY])
    plt.savefig((scr+"_not_accum.png")%(t))
    plt.clf()

    sdf_f, nor_f = sdf_func(np.squeeze(grid.cells))

    if positions is None:
        positions = particle_data_nb[in_surface(np.array([sdf_f(p) for p in particle_data_nb]))[0]]

    plt.scatter(positions[:,0],positions[:,1],s=0.1)
    plt.xlim([0,grid.dimX])
    plt.ylim([0,grid.dimY])
    plt.savefig((scr + "_pos.png") % t)
    plt.clf()

    img = np.empty((0,3))
    i = 0
    for pos in positions:
        par = extract_particles(particle_data_nb, pos, par_cnt, patch_size)[0]
        #sort_idx = sort_particles(par, np.array([sdf_f(p) for p in par]))
        #par = par[sort_idx]

        if [t,i] in samples:
            plt.scatter(par[:,0],par[:,1],s=5)
            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.savefig((scr+"_i%03d_patch.png")%(t,i))
            plt.clf()
        i+=1

        result = np.append(result, [par], axis=0)
        img = np.append(img, np.add(par*patch_size, [pos[0], pos[1], 0.]), axis=0)

    plt.scatter(img[:,0],img[:,1],s=0.1)
    plt.xlim([0,grid.dimX])
    plt.ylim([0,grid.dimY])
    plt.savefig((scr + ".png") % t)
    plt.clf()

    return result, positions

def sdf_patches(sdf, positions, patch_size, scr, t):
    res = np.empty((0,patch_size, patch_size, 2) if nor_out else (0,patch_size, patch_size))
    ps_half = patch_size // 2

    sdf_f, nor_f = sdf_func(np.squeeze(sdf))
    i=0
    img = np.zeros((1,h_dim,h_dim,2)) if nor_out else np.ones((1,h_dim,h_dim))
    for pos in positions:
        tmp = np.array([[[nor_f(pos[:2]-ps_half+np.array([x,y])) for x in range(patch_size)] for y in range(patch_size)]]) if nor_out else np.array([[[np.tanh(4.0*sdf_f(pos[:2]-ps_half+np.array([x,y]))[0]) for x in range(patch_size)] for y in range(patch_size)]])
        res = np.append(res, tmp, axis=0)

        if [t,i] in samples:
            for x in range(patch_size):
                for y in range(patch_size):
                    v = tmp[0,y,x]
                    if nor_out:
                        plt.plot([x,x+v[0]],[y,y+v[1]], '.-')
                    elif v <= 0.0:
                        plt.plot(x,y,'bo')
            plt.xlim([0,patch_size])
            plt.ylim([0,patch_size])
            plt.savefig((scr+"_i%03d_patch.png")%(t,i))
            plt.clf()
        i+=1

        if np.all(pos[:2]>ps_half) and np.all(pos[:2]<h_dim-ps_half):
            tmp = tmp[0] if nor_out else np.transpose(tmp[0], (1,0)) * circular_filter
            insert_patch(img, tmp, pos.astype(int), elem_avg if nor_out else elem_min)

        '''for x in range(0,h_dim,2):
            for y in range(0,h_dim,2):
                v = img[0,y,x]
                plt.plot([x,x+v[0]],[y,y+v[1]], '-')
                #if v <= 0.0:
                #    plt.plot(x,y,'b.')
        plt.xlim([0,h_dim])
        plt.ylim([0,h_dim])
        plt.savefig((scr+"_i%03d.png")%(t,i-1))
        plt.clf()'''

    for x in range(0,h_dim,2):
        for y in range(0,h_dim,2):
            v = img[0,y,x]
            if nor_out:
                plt.plot([x,x+v[0]],[y,y+v[1]], '.-')
            elif v <= 0.0:
                plt.plot(x,y,'bo')
    plt.xlim([0,h_dim])
    plt.ylim([0,h_dim])
    plt.savefig((scr+".png")%t)
    plt.clf()
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
        par, par_aux = extract_particles(particle_data_nb, pos, par_cnt, patch_size, aux_data)
        sort_idx = sort_particles(par, np.array([sdf_f(p) for p in par]))
        par = par[sort_idx]

        if [t,i] in samples:
            plt.scatter(par[:,0],par[:,1],s=5)
            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.savefig((scr+"_i%03d_patch.png")%(t,i))
            plt.clf()
        i+=1

        result = np.append(result, [par], axis=0)
        for k, v in par_aux.items():
            aux_res[k] = np.append(aux_res[k], [v[sort_idx]], axis=0)
        img = np.append(img, np.add(par*patch_size, [pos[0], pos[1], 0.]), axis=0)

    plt.scatter(img[:,0],img[:,1],s=0.1)
    plt.xlim([0,header['dimX']])
    plt.ylim([0,header['dimY']])
    plt.savefig((scr+".png") % t)
    plt.clf()

    return result, aux_res, positions

src_file = "%s_%s-%s"%(data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
dst_file = "%s_%s"%(data_config['prefix'], data_config['id']) + "_d%03d_%03d"

src = np.empty((0,particle_cnt_src,3))
dst = np.empty((0,particle_cnt_dst,3))

sdf_dst = np.empty((0,ref_patch_size, ref_patch_size, 2) if nor_out else (0,ref_patch_size, ref_patch_size))

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

                res, positions = load_test(src_data, 4/fac_2d, particle_cnt_src, patch_size, l_scr, t)
                #res, aux_res, positions = load_src(data_path + "source/" + src_file%(d,v,t), 4/fac_2d, particle_cnt_src, patch_size, l_scr+"_patch.png", "test/source_%03d.png", t, aux_postfix)

                src = np.append(src, res, axis=0)
                #for k, val in aux_res.items():
                #    aux_src[k] = np.append(aux_src[k], val, axis=0)
                    
                res = load_test(ref_data, 4, particle_cnt_dst, ref_patch_size, h_scr, t, positions*fac_2d)[0]
                #res = load_src(data_path + "reference/" + dst_file%(d,t), 4, particle_cnt_dst, ref_patch_size, h_scr+"_patch.png", "test/reference_%03d.png", t, positions=positions*fac_2d)[0]

                if sdf_loss or use_sdf:
                    sdf_dst = np.append(sdf_dst, sdf_patches(ref_data.cells, positions*fac_2d, ref_patch_size, sdf_scr, t), axis=0)

                dst = np.append(dst, res, axis=0)

fac = 16
k = 256
dropout = 0.5
batch_size = train_config['batch_size']
epochs = 3 # train_config['epochs']

def sdf_particle_loss(y_true, y_pred):
    sh = [s for s in y_pred.get_shape()]
    sh[0] = batch_size
    y_pred.set_shape(sh)

    sh = [s for s in y_true.get_shape()]
    sh[0] = batch_size
    sh[1] = ref_patch_size
    sh[2] = ref_patch_size
    y_true.set_shape(sh)

    def interpol(sdf, pos):
        pos = (pos+1) * 0.5 * ref_patch_size + 0.5
        sdf = tf.pad(sdf, tf.constant([[1,1],[1,1]]), "SYMMETRIC")
        w = sdf.get_shape()[0]
        x = K.cast(pos[:,0], 'int32')
        y = K.cast(pos[:,1], 'int32')
        idx = x + y * w
        facX = pos[:,0]-K.cast(x, 'float32')
        facY = pos[:,1]-K.cast(y, 'float32')
        
        l = x.get_shape()[0]

        sdf = K.flatten(K.relu(sdf))

        v  = K.gather(sdf, idx) * (1-facX) * (1-facY)
        v += K.gather(sdf, idx+1) * facX * (1-facY)
        v += K.gather(sdf, idx+w) * (1-facX) * facY
        v += K.gather(sdf, idx+w+1) * facX * facY

        return v

    y_true = tf.unstack(y_true)
    y_pred = tf.unstack(y_pred)
    return tf.stack([interpol(y_true[i],y_pred[i]) for i in range(batch_size)])

src = src[:len(src)//batch_size*batch_size]
dst = dst[:len(src)//batch_size*batch_size]

sdf_dst = sdf_dst[:len(src)//batch_size*batch_size]

for k, v in aux_src.items():
    aux_src[k] = v[:len(src)//batch_size*batch_size]

inputs = Input((particle_cnt_src,3), name="main")
#aux_input = Input((particle_cnt_src,3))

x = Dropout(dropout)(inputs)
stn = SpatialTransformer(particle_cnt_src,quat=True)
intermediate = stn(x)

x = intermediate
#x = concatenate([intermediate, stn([x,aux_input])],axis=-1)

x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_src)]

x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(fac, activation='tanh'))(x)
x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(fac, activation='tanh'))(x)

x = concatenate(x, axis=1)
x = SpatialTransformer(particle_cnt_src,fac,1)(x)

x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_src)]

x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(fac, activation='tanh'))(x)
x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(fac*2, activation='tanh'))(x)
x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(k, activation='tanh'))(x)

x = add(x)

if use_sdf:
    x = Flatten()(x)
    x = Dropout(dropout)(x)

    c = 2 if nor_out else 1

    b = np.zeros(ref_patch_size*ref_patch_size*c, dtype='float32')
    W = np.zeros((k, ref_patch_size*ref_patch_size*c), dtype='float32')
    x = Dense(ref_patch_size*ref_patch_size*c, activation='tanh', weights=[W,b])(x)

    out = Reshape((ref_patch_size, ref_patch_size, c) if nor_out else (ref_patch_size, ref_patch_size))(x)

else:
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*8, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(fac*4, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(3*particle_cnt_dst, activation='tanh')(x)

    x = Reshape((particle_cnt_dst,3))(x)
    inv_trans = x
    out = InverseTransform(stn)(x)

    '''print(x.get_shape())
    x = Reshape((4,4,64))(x)
    print(x.get_shape())
    x = Conv2DTranspose(filters=16,kernel_size=3,strides=2, activation='tanh', padding='same')(x)
    print(x.get_shape())
    x = Conv2DTranspose(filters=3, kernel_size=3,strides=2, activation='tanh', padding='same')(x)
    print(x.get_shape())
    out = x'''

'''side = int(math.sqrt(particle_cnt_src))
x = Reshape((side, side, fac))(x)

x = Conv2DTranspose(filters=16, kernel_size=3, 
                    strides=1, activation='tanh', padding='same')(x)

x = Conv2DTranspose(filters=9, kernel_size=3, 
                    strides=1, activation='tanh', padding='same')(x)'''

model = Model(inputs=inputs, outputs=out)#[inputs, aux_input], outputs=out)

if use_sdf:
    model.compile(loss='mse', optimizer=keras.optimizers.adam(lr=0.001))
    history=model.fit(x=src,y=sdf_dst,epochs=epochs,batch_size=batch_size)
else:
    if sdf_loss:
        train_model = Model(inputs=inputs, outputs=Flatten()(out))

        g_tr = np.concatenate([np.reshape(dst, (dst.shape[0], particle_cnt_dst*3)), np.reshape(sdf_dst, (sdf_dst.shape[0], ref_patch_size*ref_patch_size))], axis=-1)

        def comb_loss(x,y):
            x.set_shape((batch_size, ref_patch_size * ref_patch_size + particle_cnt_dst * 3))
            y.set_shape((batch_size, particle_cnt_dst * 3))    
            
            pos = K.reshape(x[:,:particle_cnt_dst*3], (batch_size, particle_cnt_dst, 3))
            sdf = K.reshape(x[:,particle_cnt_dst*3:], (batch_size, ref_patch_size, ref_patch_size))
            y = K.reshape(y, (batch_size, particle_cnt_dst, 3))

            return HungarianLoss(batch_size).hungarian_loss(pos,y) + 0.2*sdf_particle_loss(sdf,y)

        train_model.compile(optimizer=keras.optimizers.adam(lr=0.001), loss=comb_loss)
        history=train_model.fit(x=src,y=g_tr,epochs=epochs,batch_size=batch_size)
    else:
        model.compile(optimizer=keras.optimizers.adam(lr=0.001), loss=HungarianLoss(batch_size).hungarian_loss)
        history=model.fit(x=src,y=dst,epochs=epochs,batch_size=batch_size)
        
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

interm = Model(inputs=inputs, outputs=intermediate)
invm = Model(inputs=inputs, outputs=inv_trans)
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
            
                src, positions = load_test(src_data, 4/fac_2d, particle_cnt_src, patch_size, t_scr, t)

                ref = load_test(ref_data, 4, particle_cnt_dst, ref_patch_size, h_t_scr, t, positions*fac_2d)[0]

                sdf_patches(ref_data.cells, positions*fac_2d, ref_patch_size, sdf_t_scr, t)
                
                #src, aux_src, positions = load_src(data_path + "source/" + src_file%(d,v,t), 4/fac_2d, particle_cnt_src, patch_size, t_scr+"_patch.png", "test/test_%03d.png", t, aux_postfix)

                result = model.predict(x=src,batch_size=batch_size)
                inter_result = interm.predict(x=src,batch_size=batch_size)
                inv_result = invm.predict(x=src,batch_size=batch_size)
                print(stn.locnet.predict(x=src,batch_size=batch_size))

                if use_sdf:
                    ps_half = ref_patch_size//2
                    img = np.zeros((1,h_dim,h_dim,2)) if nor_out else np.ones((1,h_dim,h_dim))
                    for i in range(len(result)):
                        tmp = result[i]#np.arctanh(np.clip(result[i],-.999999,.999999))
                        inter_tmp = inter_result[i]
                        pos = positions[i]*fac_2d
                        if [t,i] in samples:
                            for x in range(ref_patch_size):
                                for y in range(ref_patch_size):
                                    v = tmp[y,x]
                                    if nor_out:
                                        plt.plot([x,x+v[0]],[y,y+v[1]], '.-')
                                    elif v <= 0.0:
                                        plt.plot(x,y,'bo')
                            plt.xlim([0,ref_patch_size])
                            plt.ylim([0,ref_patch_size])
                            plt.savefig((r_scr+"_i%03d_patch.png")%(t,i))
                            plt.clf()

                            for x in range(ref_patch_size):
                                for y in range(ref_patch_size):
                                    v = inter_tmp[y,x]
                                    if nor_out:
                                        plt.plot([x,x+v[0]],[y,y+v[1]], '.-')
                                    elif v <= 0.0:
                                        plt.plot(x,y,'bo')
                            plt.xlim([0,ref_patch_size])
                            plt.ylim([0,ref_patch_size])
                            plt.savefig((r_scr+"_i%03d_inter_patch.png")%(t,i))
                            plt.clf()
                        i+=1

                        if np.all(pos[:2]>ps_half) and np.all(pos[:2]<h_dim-ps_half):
                            #tmp = np.transpose(tmp, (1,0)) * circular_filter
                            #insert_patch(img, tmp, pos.astype(int), elem_min)
                            insert_patch(img, tmp[0], pos.astype(int), elem_avg)

                    for x in range(0,h_dim,2):
                        for y in range(0,h_dim,2):
                            v = img[0,y,x]
                            if nor_out:
                                plt.plot([x,x+v[0]],[y,y+v[1]], '.-')
                            elif v <= 0.0:
                                plt.plot(x,y,'bo')
                    plt.xlim([0,h_dim])
                    plt.ylim([0,h_dim])
                    plt.savefig((r_scr+"_r%03d.png")%(t,r))
                    plt.clf()
                else:
                    img = np.empty((0,3))
                    ref_img = np.empty((0,3))

                    for i in range(len(result)):
                        #kmeans.fit(result[i,:,:2])
                        #par = np.append(kmeans.cluster_centers_, np.zeros((10,1)), axis=1)
                        par = result[i]#,:10]

                        if [t,i] in samples:# or True:
                            plt.scatter(result[i,:,0],result[i,:,1],s=5)
                            plt.scatter(ref[i,:,0],ref[i,:,1],c='r',s=5)
                            plt.xlim([-1,1])
                            plt.ylim([-1,1])
                            plt.savefig((r_scr+"_i%03d_patch.png")%(t,i))
                            plt.clf()

                            plt.scatter(inter_result[i,:,0],inter_result[i,:,1],s=5)
                            plt.xlim([-1,1])
                            plt.ylim([-1,1])
                            plt.savefig((r_scr+"_i%03d_inter_patch.png")%(t,i))
                            plt.clf()

                            plt.scatter(inv_result[i,:,0],inv_result[i,:,1],s=5)
                            plt.xlim([-1,1])
                            plt.ylim([-1,1])
                            plt.savefig((r_scr+"_i%03d_inv_patch.png")%(t,i))
                            plt.clf()
                        
                        par = np.add(par*ref_patch_size, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.])
                        img = np.append(img, par, axis=0)
                        ref_img = np.append(ref_img, np.add(ref[i]*ref_patch_size, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.]), axis=0)

                    plt.scatter(img[:,0],img[:,1],s=0.1)
                    plt.scatter(ref_img[:,0],ref_img[:,1],c='r',s=0.01)
                    plt.xlim([0,h_dim])
                    plt.ylim([0,h_dim])
                    plt.savefig((r_scr+"_r%03d.png")%(t,r))
                    plt.clf()

            
