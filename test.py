import sys, os
sys.path.append("manta/scenes/tools")
sys.path.append("hungarian/")

import json
from shell_script import *
from helpers import *
from uniio import *

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

import matplotlib.pyplot as plt
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

patch_size = 5
stride = 1
surface = 0.5
particle_cnt = 20 #pre_config['par_cnt']

src_file = "%s_%s-%s_d%03d_var%02d"%(data_config['prefix'], data_config['id'], pre_config['id'], dataset, var) + "_%03d"
dst_file = "%s_%s_d%03d"%(data_config['prefix'], data_config['id'], dataset) + "_%03d"

src = None
dst = None

for t in range(t_start, t_end):
    particle_data = readParticles(data_path + "source/" + src_file%t + "_ps.uni")[1]

    header, sdf = readUni(data_path + "source/" + src_file%t + "_sdf.uni")
    #patch_pos = get_patches(sdf, pre_config['patch_size'], header['dimX'], header['dimY'], 1, pre_config['surf'])

    positions = None
    img = None
    i = 0
    for pos in particle_data:
        if np.any(pos[:2] < 4/3) or np.any(pos[:2] > header['dimX']-4/3):
            continue
        par = extract_particles(particle_data, pos, particle_cnt, pre_config['patch_size'])

        if [t,i] in samples:
            plt.scatter(par[:,0],par[:,1])
            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.savefig(l_scr%(t,i))
            plt.clf()
        i+=1

        if src is None:
            src = np.array([par])
            positions = np.array([pos])
        else:
            src = np.concatenate([src, np.array([par])])
            positions = np.concatenate([positions, np.array([pos])])

        par = np.add(par, [pos[0], pos[1], 0.])

        if img is None:
            img = par
        else:
            img = np.concatenate([img,par])

    print(src.shape)
    plt.scatter(img[:,0],img[:,1],s=0.3)
    plt.savefig("test/source_%03d.png" % t)
    plt.clf()

    particle_data = readParticles(data_path + "reference/" + dst_file%t + "_ps.uni")[1]

    header, sdf = readUni(data_path + "reference/" + dst_file%t + "_sdf.uni")
    #patch_pos = get_patches(sdf, 15, header['dimX'], header['dimY'], 1, pre_config['surf'])

    img = None
    i = 0

    print(particle_data.shape)

    for pos in positions*3:
        par = extract_particles(particle_data, pos, particle_cnt, 15)

        if [t,i] in samples:
            plt.scatter(par[:,0],par[:,1])
            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.savefig(h_scr%(t,i))
            plt.clf()
        i+=1

        if dst is None:
            dst = np.array([par])
        else:
            dst = np.concatenate([dst, np.array([par])])

        par = np.add(par, [pos[0], pos[1], 0.])

        if img is None:
            img = par
        else:
            img = np.concatenate([img,par])

    print(dst.shape)
    plt.scatter(img[:,0],img[:,1],s=0.1)
    plt.savefig("test/reference_%03d.png" % t)
    plt.clf()

par_cnt = particle_cnt
fac = 64
k = 1024
dropout = 1.0
batch_size = train_config['batch_size']
epochs = 10 # train_config['epochs']

src = src[:len(src)//batch_size*batch_size]
dst = dst[:len(src)//batch_size*batch_size]

src = dst

inputs = Input((par_cnt,3), name="main")

x = Dropout(dropout)(inputs)
x = SpatialTransformer(par_cnt)(x)

intermediate = x

'''x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(par_cnt)]

x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(fac, activation='tanh'))(x)
x = SplitLayer(Dropout(dropout))(x)
x = SplitLayer(Dense(fac, activation='tanh'))(x)

x = concatenate(x, axis=1)
x = SpatialTransformer(par_cnt,fac,1)(x)

x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(par_cnt)]

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
x = Dense(3*par_cnt, activation='tanh')(x)

out = Reshape((par_cnt,3))(x)

model = Model(inputs=inputs, outputs=out)
model.compile(loss=HungarianLoss(batch_size).hungarian_loss, optimizer=keras.optimizers.adam(lr=0.001))
        
#model.summary()

history=model.fit(x=src,y=dst,epochs=epochs,batch_size=batch_size)

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

for t in range(t_start, t_end): 
    img = None
    particle_data = readParticles(data_path + "source/" + src_file%t + "_ps.uni")[1]

    header, sdf = readUni(data_path + "source/" + src_file%t + "_sdf.uni")

    src = None
    positions = None
    i = 0
    for pos in particle_data:
        if np.any(pos[:2] < 4/3) or np.any(pos[:2] > header['dimX']-4/3):
            continue
        par = extract_particles(particle_data, pos, particle_cnt, pre_config['patch_size'])
    
        if [t,i] in samples:
            plt.scatter(par[:,0],par[:,1])
            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.savefig(t_scr%(t,i))
            plt.clf()
        i+=1

        if src is None:
            src = np.array([par])
            positions = np.array([pos])
        else:
            src = np.concatenate([src, np.array([par])])
            positions = np.concatenate([positions, np.array([pos])])

        par = np.add(par, [pos[0], pos[1], 0.])

        if img is None:
            img = par
        else:
            img = np.concatenate([img,par])
    
    plt.scatter(img[:,0],img[:,1],s=0.1)
    plt.savefig("test/test_%03d.png"%t)
    plt.clf()

    result = model.predict(x=src,batch_size=batch_size)
    inter_result = interm.predict(x=src,batch_size=batch_size)

    for i in range(len(result)):
        par = np.add(result[i], [positions[i,0], positions[i,1], 0.])

        if [t,i] in samples:
            plt.scatter(result[i,:,0],result[i,:,1])
            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.savefig(r_scr%(t,i))
            plt.clf()

        if img is None:
            img = par
        else:
            img = np.concatenate([img,par])

    plt.scatter(img[:,0],img[:,1],s=0.1)
    plt.savefig("test/result_%03d.png"%t)
    plt.clf()
