import sys, os
sys.path.append("manta/scenes/tools")
sys.path.append("hungarian/")

import json
from shell_script import *
from helpers import *
from uniio import *

import scipy
from scipy import interpolate

import keras
from keras import losses
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, Dense, Dropout
from keras.layers import Reshape, concatenate, add, Flatten, Lambda
from spatial_transformer import *
from split_layer import *

import random
import math

from hungarian_loss import HungarianLoss

from particle_grid import ParticleGrid

import matplotlib.pyplot as plt

def normals(sdf):
    x,y = np.gradient(sdf)
    x = np.expand_dims(x,axis=-1)
    y = np.expand_dims(y,axis=-1)
    g = np.concatenate([x,y],axis=-1)
    return np.nan_to_num(g/np.linalg.norm(g,axis=-1,keepdims=True))

def curvature(nor):
    dif = np.gradient(nor)[0]
    return np.sum(np.square(dif),axis=1)

def sdf_func(sdf):
    x_v = np.arange(0.5, sdf.shape[0]+0.5)
    y_v = np.arange(0.5, sdf.shape[1]+0.5)
    sdf_f = lambda x: interpolate.interp2d(x_v, y_v, sdf)(x[0],x[1])
    nor = normals(sdf)
    nor_f = lambda x: np.concatenate([interpolate.interp2d(x_v, y_v, nor[:,:,1])(x[0],x[1]), interpolate.interp2d(x_v, y_v, nor[:,:,0])(x[0],x[1])])
    '''for i in np.arange(sdf.shape[0],step=0.5):
        for j in np.arange(sdf.shape[1],step=0.5):
            v = sdf_f(i,j)
            if v > -1.0 and v < 0.1:
                n = nor_f([i,j])
                plt.plot(i,j,'bo')
                plt.plot([i,i+n[0]],[j,j+n[1]], '.-')
    plt.xlim([0,sdf.shape[0]])
    plt.ylim([0,sdf.shape[1]])
    plt.show()'''
    return sdf_f, nor_f    

def sort_particles(par, nor, fac=1):
    w =  np.sum(np.abs(nor) + np.square(par), axis=-1)
    #w = np.nan_to_num(np.abs(np.reciprocal(np.sum(np.power(par[:,:2],fac) * nor, axis=-1)))) #+ fac1 * fac1 * np.sum(np.square(par), axis=-1)
    return np.argsort(w)


paramUsed = []

data_path = getParam("data", "data/", paramUsed)
manta_path = getParam("manta", "manta/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0

dataset = int(getParam("dataset", 0, paramUsed))
var = int(getParam("var", 0, paramUsed))

l_scr = getParam("l_scr", "test/low_patch_t%03d_i%03d.png", paramUsed)
h_scr = getParam("h_scr", "test/high_patch_t%03d_i%03d.png", paramUsed)
t_scr = getParam("t_scr", "test/test_patch_t%03d_i%03d.png", paramUsed)
r_scr = getParam("r_scr", "test/res_patch_t%03d_i%03d.png", paramUsed)

patch_sample_cnt = int(getParam("samples", 3, paramUsed))

t_start = int(getParam("t_start", -1, paramUsed))
t_end = int(getParam("t_end", -1, paramUsed))

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

random.seed(235)
samples = [[random.randint(t_start, t_end-1), random.randint(0, 100)] for i in range(patch_sample_cnt)]
print(samples)

patch_size = 5#pre_config['patch_size']
fac = 9
fac_2d = 3
ref_patch_size = patch_size * fac_2d
stride = 1
surface = 1.0
particle_cnt_src = 50 #pre_config['par_cnt']
particle_cnt_dst = 10

def in_bound(pos, bnd_min, bnd_max):
    return np.where(np.all([np.all(bnd_min<=pos,axis=-1),np.all(pos<=bnd_max,axis=-1)],axis=0))

def in_surface(sdf):
    return np.where(abs(sdf) < surface)

def load_src(prefix, t, bnd, par_cnt, patch_scr, scr, aux={}, positions=None):
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
            size = np.arange(20.0,0.0,-20.0/len(par))
            plt.scatter(par[:,0],par[:,1],s=size)
            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.savefig(patch_scr%(t,i))
            plt.clf()
        i+=1

        result = np.append(result, [par], axis=0)
        for k, v in par_aux.items():
            aux_res[k] = np.append(aux_res[k], [v[sort_idx]], axis=0)
        img = np.append(img, np.add(par*patch_size, [pos[0], pos[1], 0.]), axis=0)

    plt.scatter(img[:,0],img[:,1],s=0.1)
    plt.xlim([0,header['dimX']])
    plt.ylim([0,header['dimY']])
    plt.savefig(scr % t)
    plt.clf()

    return result, aux_res, positions

src_file = "%s_%s-%s"%(data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
dst_file = "%s_%s"%(data_config['prefix'], data_config['id']) + "_d%03d_%03d"

src = np.empty((0,particle_cnt_src,3))
dst = np.empty((0,particle_cnt_dst,3))

aux_postfix = {
    "vel":"v"
}

aux_src = {}
for k in aux_postfix:
    aux_src[k] = np.empty((0, particle_cnt_src, 3 if k == "vel" else 1))

for v in range(var):
    for d in range(dataset):
        for t in range(t_start, t_end):
            res, aux_res, positions = load_src(data_path + "source/" + src_file%(d,v,t), t, 4/fac_2d, particle_cnt_src, l_scr, "test/source_%03d.png", aux_postfix)

            src = np.append(src, res, axis=0)
            for k, val in aux_res.items():
                aux_src[k] = np.append(aux_src[k], val, axis=0)

            res = load_src(data_path + "reference/" + dst_file%(d,t), t, 4, particle_cnt_dst, h_scr, "test/reference_%03d.png", positions=positions*fac_2d)[0]
            dst = np.append(dst, res, axis=0)

fac = 64
k = 1024
dropout = 1.0
batch_size = train_config['batch_size']
epochs = 10 # train_config['epochs']

src = src[:len(src)//batch_size*batch_size]
dst = dst[:len(src)//batch_size*batch_size]

for k, v in aux_src.items():
    aux_src[k] = v[:len(src)//batch_size*batch_size]

inputs = Input((particle_cnt_src,3), name="main")
aux_input = Input((particle_cnt_src,3))

x = Dropout(dropout)(inputs)
stn = SpatialTransformer(particle_cnt_src)
intermediate = stn(x)

x = concatenate([intermediate, stn([x,aux_input])],axis=-1)

'''x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_src)]

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

x = add(x)'''

x = Flatten()(x)
x = Dropout(dropout)(x)
x = Dense(fac*8, activation='tanh')(x)
x = Dropout(dropout)(x)
x = Dense(fac*4, activation='tanh')(x)
x = Dropout(dropout)(x)
x = Dense(3*particle_cnt_dst, activation='tanh')(x)

x = Reshape((particle_cnt_dst,3))(x)
out = InverseTransform(stn)(x)

model = Model(inputs=[inputs, aux_input], outputs=out)
model.compile(loss=HungarianLoss(batch_size).hungarian_loss, optimizer=keras.optimizers.adam(lr=0.001))
        
#model.summary()
history=model.fit(x=[src,aux_src['vel']],y=dst,epochs=epochs,batch_size=batch_size)

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

for v in range(1):
    for d in range(5,6):
        for t in range(t_start, t_end): 
            src, aux_src, positions = load_src(data_path + "source/" + src_file%(d,v,t), t, 4/fac_2d, particle_cnt_src, t_scr, "test/test_%03d.png", aux_postfix)

            result = model.predict(x=[src,aux_src['vel']],batch_size=batch_size)
            inter_result = interm.predict(x=src,batch_size=batch_size)

            img = np.empty((0,3))

            for i in range(len(result)):
                par = np.add(result[i]*ref_patch_size, [positions[i,0]*fac_2d, positions[i,1]*fac_2d, 0.])

                if [t,i] in samples:
                    size = np.arange(20.0,0.0,-20.0/len(par))
                    plt.scatter(result[i,:,0],result[i,:,1],s=size)
                    plt.xlim([-1,1])
                    plt.ylim([-1,1])
                    plt.savefig(r_scr%(t,i))
                    plt.clf()
                img = np.append(img, par, axis=0)

            plt.scatter(img[:,0],img[:,1],s=0.1)
            plt.xlim([0,50*fac_2d])
            plt.ylim([0,50*fac_2d])
            plt.savefig("test/result_%03d.png"%t)
            plt.clf()
