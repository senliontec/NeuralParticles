import numpy as np
from glob import glob
import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import particle_radius
from neuralparticles.tools.shell_script import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw, readNumpyOBJ, writeNumpyOBJ, readParticlesUni
from neuralparticles.tools.particle_grid import ParticleIdxGrid
import random
import math
from collections import OrderedDict
import time

def project(n, v):
    return v - np.dot(n,v) * n

def deviation(n, v0, v1):
    t = project(n, v0)
    return t/np.dot(t,v0)

A_l = np.array([
    [+1.,+0.,+0.,+0.],
    [+0.,+0.,+1.,+0.],
    [-3.,+3.,-2.,-1.],
    [+2.,-2.,+1.,+1.]
])
A_r = np.array([
    [+1.,+0.,-3.,+2.],
    [+0.,+0.,+3.,-2.],
    [+0.,+1.,-2.,+1.],
    [+0.,+0.,-1.,+1.]
])

manta_path = getParam("manta", "neuralparticles/")
verbose = int(getParam("verbose", 0)) != 0
mesh_path = getParam("mesh", "mesh/")
config_path = getParam("config", "config/version_00.txt")
gui = int(getParam("gui", 0))
pause = int(getParam("pause", 0))
res = int(getParam("res", -1))

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

if res < 0:
    res = data_config['res']
bnd = data_config['bnd']

min_n = pre_config['min_n']
factor = pre_config['factor']
factor_d = math.pow(factor,1/dim)
lres = int(res/factor_d)
search_r = res/lres * (1/sub_res) * 0.77 if factor > 1 else 0

random.seed(data_config['seed'])
np.random.seed(data_config['seed'])

ref_path = "%sreference/%s_%s_" % (mesh_path, data_config['prefix'], data_config['id']) + "%03d"
src_path = "%ssource/%s_%s_" % (mesh_path, data_config['prefix'], data_config['id']) + "%03d"

if not os.path.exists(mesh_path + "reference/"):
    os.makedirs(mesh_path + "reference/")

if not os.path.exists(mesh_path + "source/"):
    os.makedirs(mesh_path + "source/")

samples = glob(mesh_path + "*.obj")
samples.sort()

vertices = None
normals = None
faces = None
for i,item in enumerate(samples):
    d = readNumpyOBJ(item)
    if i == 0:
        vertices = np.empty((len(samples), d[0].shape[0], 3))
        normals = np.empty((len(samples),), dtype=object)
        faces = np.empty((len(samples), d[2].shape[0], 2, 4),dtype=int)
    vertices[i] = d[0]
    normals[i] = d[1]
    faces[i] = d[2]

vertices -= np.min(vertices,axis=(0,1))
vertices *= (res - 2 * bnd) / np.max(vertices)
vertices += bnd

bary_coord = np.empty((len(faces[0]),),dtype=object)
data_cnt = 0
d_idx = None
for i in range(len(samples)):
    print("Load mesh: %d/%d" % (i+1, len(samples)))
    if i == 0:
        for fi, f in enumerate(faces[i]):
            v = vertices[i,f[0]]

            area = np.linalg.norm(np.cross(v[1]-v[0], v[2]-v[0]))/2
            area += np.linalg.norm(np.cross(v[2]-v[0], v[3]-v[0]))/2

            par_cnt = vert_area_ratio * area
            par_cnt = int(par_cnt) + int(np.random.random() < par_cnt % 1)
            bary_coord[fi] = np.random.random((par_cnt, 2))
            data_cnt += par_cnt

    data = np.empty((data_cnt, 3))
    di = 0
    for fi, f in enumerate(faces[i]):
        v = vertices[i,f[0]]
        n = normals[i][f[1]]

        x01 = (v[1] - v[0])
        x01 /= np.linalg.norm(x01,axis=-1,keepdims=True)
        x32 = (v[2] - v[3])
        x32 /= np.linalg.norm(x32,axis=-1,keepdims=True)
        y12 = (v[2] - v[1])
        y12 /= np.linalg.norm(y12,axis=-1,keepdims=True)
        y03 = (v[3] - v[0])
        y03 /= np.linalg.norm(y03,axis=-1,keepdims=True)

        A_f = np.zeros((4,4,3))

        A_f[0,0] = v[0]
        A_f[0,1] = v[3]
        A_f[1,0] = v[1]
        A_f[1,1] = v[2]

        A_f[0,2] = deviation(n[0], y03, x01)
        A_f[0,3] = deviation(n[3], y03, x32)
        A_f[1,2] = deviation(n[1], y12, x01)
        A_f[1,3] = deviation(n[2], y12, x32)

        A_f[2,0] = deviation(n[0], x01, -y03)
        A_f[2,1] = deviation(n[3], x32, -y03)
        A_f[3,0] = deviation(n[1], x01, -y12)
        A_f[3,1] = deviation(n[2], x32, -y12)

        A_f[2,2] = (A_f[2,1] - A_f[2,0])/np.linalg.norm(y03)
        A_f[2,3] = (A_f[2,1] - A_f[2,0])/np.linalg.norm(y03)
        A_f[3,2] = (A_f[3,1] - A_f[3,0])/np.linalg.norm(y12)
        A_f[3,3] = (A_f[3,1] - A_f[3,0])/np.linalg.norm(y12)

        param = np.zeros((4,4,3))

        for j in range(3):
            param[...,j] = np.matmul(np.matmul(A_l, A_f[...,j]), A_r)

        for (a1,a2) in bary_coord[fi]:
            for j in range(3):
                data[di,j] = np.matmul(np.matmul(np.array([1,a1,a1**2,a1**3]), param[...,j]), np.array([1,a2,a2**2,a2**3]))
            di += 1

    hdr = OrderedDict([ ('dim',len(data)),
                        ('dimX',res),
                        ('dimY',res),
                        ('dimZ',1 if dim == 2 else res),
                        ('elementType',0),
                        ('bytesPerElement',16),
                        ('info',b'\0'*256),
                        ('timestamp',(int)(time.time()*1e6))])

    writeParticlesUni(ref_path%i +"_ps.uni", hdr, data)
    writeNumpyRaw(ref_path%i, data)
    writeNumpyOBJ(ref_path%i +".obj", data)

    if i == 0:
        d_idx = np.arange(data_cnt)
        mask = np.ones(data_cnt, dtype=bool)
        np.random.shuffle(d_idx)

        idx_grid = ParticleIdxGrid(data, (1 if dim == 2 else res, res, res))

        for j in range(data_cnt):
            p_idx = d_idx[j]
            if mask[p_idx]:
                idx = np.array(idx_grid.get_range(data[p_idx], 2*search_r))
                idx = idx[particle_radius(data[idx], data[p_idx], search_r)]
                if len(idx) < min_n:
                    mask[p_idx] = False
                else:
                    mask[idx] = False
                    mask[p_idx] = True
        d_idx = np.where(mask)[0]
        print("particles reduced: %d -> %d (%.1f)" % (data_cnt, len(d_idx), data_cnt/len(d_idx)))
    
    low_res_data = data[d_idx]/factor_d

    hdr['dim'] = len(low_res_data)
    hdr['dimX'] = lres
    hdr['dimY'] = lres
    hdr['dimZ'] = 1 if dim == 2 else lres

    writeParticlesUni(src_path%i +"_ps.uni", hdr, low_res_data)
    writeNumpyRaw(src_path%i, low_res_data)
    writeNumpyOBJ(src_path%i +".obj", low_res_data)