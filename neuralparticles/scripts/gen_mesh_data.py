import numpy as np
from glob import glob
import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw, readNumpyOBJ, writeNumpyOBJ

import math
from collections import OrderedDict
import time

mesh_path = getParam("mesh", "mesh/")
config_path = getParam("config", "config/version_00.txt")

checkUnusedParams()

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

sub_res = data_config['sub_res']
dim = data_config['dim']
vert_area_ratio = sub_res ** 2

factor_d = math.pow(pre_config['factor'], 1/dim)
factor_d = np.array([factor_d, factor_d, 1 if dim == 2 else factor_d])
res = int(data_config['res']/factor_d[0])
bnd = data_config['bnd']


dst_path = "%s%s_%s_" % (mesh_path, data_config['prefix'], data_config['id']) + "%03d"

samples = glob(mesh_path + "*.obj")
samples.sort()

vertices = None
normals = None
faces = None
for i,item in enumerate(samples):
    d = readNumpyOBJ(item)
    if i == 0:
        vertices = np.empty((len(samples), d[0].shape[0], 3))
        normals = np.empty((len(samples), d[1].shape[0], 3))
        faces = np.empty((len(samples), d[2].shape[0], 2, 3),dtype=int)
    vertices[i] = d[0]
    normals[i] = d[1]
    faces[i] = d[2]

vertices -= np.min(vertices,axis=(0,1))
vertices *= (res - 2 * bnd) / np.max(vertices)
vertices += bnd

bary_coord = np.empty((len(faces[0]),),dtype=object)
data_cnt = 0
for i in range(len(samples)):
    if i == 0:
        for fi, f in enumerate(faces[i]):
            v0 = vertices[i,f[0,0]]
            v1 = vertices[i,f[0,1]]
            v2 = vertices[i,f[0,2]]

            area = np.linalg.norm(np.cross(v1-v0, v2-v0))/2
            par_cnt = vert_area_ratio * area
            par_cnt = int(par_cnt) + int(np.random.random() < par_cnt % 1)
            a = np.random.random((par_cnt, 3))
            if par_cnt > 0:
                a /= np.sum(a, axis=-1, keepdims=True)
            bary_coord[fi] = a[:,:2]
            data_cnt += par_cnt

    data = np.empty((data_cnt, 3))
    di = 0
    for fi, f in enumerate(faces[i]):
        v0 = vertices[i,f[0,0]]
        v1 = vertices[i,f[0,1]]
        v2 = vertices[i,f[0,2]]

        for (a1,a2) in bary_coord[fi]:
            data[di] = (v1-v0)*a1 + (v2-v0)*a2 + v0
            di += 1

    hdr = OrderedDict([ ('dim',len(data)),
                        ('dimX',res),
                        ('dimY',res),
                        ('dimZ',1 if dim == 2 else res),
                        ('elementType',0),
                        ('bytesPerElement',16),
                        ('info',b'\0'*256),
                        ('timestamp',(int)(time.time()*1e6))])

    writeParticlesUni(dst_path%i +".uni", hdr, data)
    writeNumpyRaw(dst_path%i, data)
    writeNumpyOBJ(dst_path%i +".obj", data)

    hdr['dim'] = len(vertices[i])
    writeParticlesUni(dst_path%i +"_src.uni", hdr, vertices[i])
    writeNumpyOBJ(dst_path%i +"_src.obj", vertices[i])