import numpy as np
from glob import glob
import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw, readNumpyOBJ, writeNumpyOBJ

import math
from collections import OrderedDict
import time

def project(n, v):
    return v - np.dot(n,v) * n

def deviation(n, v0, v1):
    t = project(n, v0)
    return t/np.dot(t,v0)

    """n_ = np.cross(v0, v1, axis=-1)
    n_ /= np.linalg.norm(n_,axis=-1,keepdims=True)
    v1_ = np.cross(n_, v0, axis=-1)
    v1_ /= np.linalg.norm(v1_,axis=-1,keepdims=True)

    t0 = np.cross(v0, n)
    t0 /= np.linalg.norm(t0, axis=-1, keepdims=True)
    t1 = np.cross(n, v1_)
    t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    t2 = np.cross(n_, n)
    t2 /= np.linalg.norm(t2, axis=-1, keepdims=True)

    t0 *= np.sign(np.dot(t,t0 * v1_))
    t1 *= np.sign(np.dot(t,t1 * n_))
    t2 *= np.sign(np.dot(t,t2 * v0))
    t = t0 * v1_ + t1 * n_ + t2 * v0

    return t/np.dot(t,v0)"""


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

'''
v = np.array([
    [0.,0.,0.],
    [1.,0.,0.],
    [1.,0.,1.],
    [0.,0.,1.]
]) # 00 10 11 01

n = np.array([
    [-1.,1.,-1.],
    [1.,1.,-1.],
    [1.,1.,1.],
    [-1.,1.,1.]
])
"""
n = np.array([
    [0.,1.,0.],
    [0.,1.,0.],
    [0.,1.,0.],
    [0.,1.,0.]
])
"""
n /= np.linalg.norm(n,axis=-1,keepdims=True)
print(n)


x01 = (v[1] - v[0])
x01 /= np.linalg.norm(x01,axis=-1,keepdims=True)
x32 = (v[2] - v[3])
x32 /= np.linalg.norm(x32,axis=-1,keepdims=True)
y12 = (v[2] - v[1])
y12 /= np.linalg.norm(y12,axis=-1,keepdims=True)
y03 = (v[3] - v[0])
y03 /= np.linalg.norm(y03,axis=-1,keepdims=True)

print(x01)
print(x32)
print(y12)
print(y03)

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

print(A_l)
print(A_f)
print(A_r)

a = np.zeros((4,4,3))

for i in range(3):
    a[...,i] = np.matmul(np.matmul(A_l, A_f[...,i]), A_r)

print(a)

data = np.zeros((1000,3))
for i in range(1000):
    x = np.random.random()
    y = np.random.random()

    for j in range(3):
        data[i,j] = np.matmul(np.matmul(np.array([1,x,x**2,x**3]), a[...,j]), np.array([1,y,y**2,y**3]))

writeNumpyOBJ("quad_test.obj", v)
writeNumpyOBJ("bicubic_test.obj", data)

exit()
'''
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

dst_path = "%soutput/%s_%s_" % (mesh_path, data_config['prefix'], data_config['id']) + "%03d"

if not os.path.exists(mesh_path + "output/"):
    os.makedirs(mesh_path + "output/")

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
        faces = np.empty((len(samples), d[2].shape[0], 2, 4),dtype=int)
    vertices[i] = d[0]
    #normals[i] = d[1]
    faces[i] = d[2]

vertices -= np.min(vertices,axis=(0,1))
vertices *= (res - 2 * bnd) / np.max(vertices)
vertices += bnd

bary_coord = np.empty((len(faces[0]),),dtype=object)
data_cnt = 0
for i in range(len(samples)):
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
        n = normals[i,f[1]]

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

    writeParticlesUni(dst_path%i +".uni", hdr, data)
    writeNumpyRaw(dst_path%i, data)
    writeNumpyOBJ(dst_path%i +".obj", data)
