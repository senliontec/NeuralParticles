import numpy as np
from glob import glob
import os
import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import particle_radius
from neuralparticles.tools.shell_script import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw, readNumpyOBJ, writeNumpyOBJ, readParticlesUni, writeUni
from neuralparticles.tools.particle_grid import ParticleIdxGrid
from neuralparticles.tensorflow.losses.tf_approxmatch import approx_vel, emd_loss
from scipy import optimize
import random
import math
from collections import OrderedDict
import time
import imageio

import keras.backend as K

def _approx_vel(pos, npos, h=0.5, it=1):
    """cost = np.linalg.norm(np.expand_dims(pos, axis=1) - np.expand_dims(npos, axis=0), axis=-1)
    idx = optimize.linear_sum_assignment(cost)

    vel = np.zeros_like(pos)
    vel[idx[0]] = npos[idx[1]] - pos[idx[0]]"""

    vel = K.eval(approx_vel(K.constant(np.expand_dims(pos, 0)), K.constant(np.expand_dims(npos, 0))))[0]

    dist = np.linalg.norm(np.expand_dims(pos, axis=0) - np.expand_dims(pos, axis=1), axis=-1)
    dist = np.exp(-dist/h)
    w = np.clip(np.sum(dist, axis=1, keepdims=True), 1, 10000)
    for i in range(it):
        vel = np.dot(dist, vel)/w

    return vel

def project(n, v):
    return v - np.dot(n,v) * n

def deviation(n, v0, v1):
    t = project(n, v0)
    return t/np.dot(t,v0)

def viewWorldM(rot, pos):
    m = np.zeros((3,4))

    m[0,0] = 1 - 2*rot[2]**2 - 2*rot[3]**2
    m[0,1] = 2*rot[1]*rot[2] - 2*rot[3]*rot[0]
    m[0,2] = 2*rot[1]*rot[3] + 2*rot[2]*rot[0]

    m[1,0] = 2*rot[1]*rot[2] + 2*rot[3]*rot[0]
    m[1,1] = 1 - 2*rot[1]**2 - 2*rot[3]**2
    m[1,2] = 2*rot[2]*rot[3] - 2*rot[1]*rot[0]
    
    m[2,0] = 2*rot[1]*rot[3] - 2*rot[2]*rot[0]
    m[2,1] = 2*rot[2]*rot[3] + 2*rot[1]*rot[0]
    m[2,2] = 1 - 2*rot[1]**2 - 2*rot[2]**2

    m[0:3,3] = -pos

    return m

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

data_path = getParam("data", "data/")
mesh_path = getParam("mesh", "")
config_path = getParam("config", "config/version_00.txt")
debug = int(getParam("debug", 0)) != 0
res = int(getParam("res", -1))
eval = int(getParam("eval", 0)) != 0
test = int(getParam("test", 0)) != 0
t_end = int(getParam("t_end", -1))
gpu = getParam("gpu", "-1")

min_v = np.asarray(getParam("min_v", "-2,-2,2").split(","), dtype="float32")
max_v = np.asarray(getParam("max_v", "2,2,-2").split(","), dtype="float32")
scale = np.abs(max_v - min_v)

checkUnusedParams()

if not gpu is "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

if mesh_path == "":
    mesh_path = data_path

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

off = data_config['data_count'] if eval else 0

discretize = data_config['disc'] 
max_z = data_config['max_z'] 

t_int = data_config['t_int']
culling = data_config["cull"]
scan = data_config["scan"]

random.seed(data_config['seed'])
np.random.seed(data_config['seed'])

if test:
    src_path = "%sreal/%s_%s_" % (data_path, data_config['prefix'], data_config['id']) + "d%03d_%03d"

    if not os.path.exists(data_path + "real/"):
        os.makedirs(data_path + "real/")
    t_int = 1
    obj_cnt = len(glob(mesh_path + "*"))
else:
    ref_path = "%sreference/%s_%s_" % (data_path, data_config['prefix'], data_config['id']) + "d%03d_%03d"
    src_path = "%ssource/%s_%s-%s_" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "d%03d_var00_%03d"

    if not os.path.exists(data_path + "reference/"):
        os.makedirs(data_path + "reference/")

    if not os.path.exists(data_path + "source/"):
        os.makedirs(data_path + "source/")

    obj_cnt = len(glob(mesh_path + "objs/*"))

frame_cnt = data_config['frame_count'] if t_end < 0 else t_end

#if (eval and obj_cnt != data_config['test_count']) or (not eval and obj_cnt != data_config['data_count']):
#    print("Mismatch between obj count and 'data_count'/'test_count' in config file!")
#    exit()

if debug:
    ref_path = data_path + "debug/ref/d%03d_%03d"
    src_path = data_path + "debug/src/d%03d_%03d"

    if not os.path.exists(data_path + "debug/ref/"):
        os.makedirs(data_path + "debug/ref/")

    if not os.path.exists(data_path + "debug/src/"):
        os.makedirs(data_path + "debug/src/")

    obj_cnt = 1
    frame_cnt = 3

for d in range(obj_cnt):
        print("Load dataset %d/%d" % (d+1, obj_cnt))

        cam_cnt = 1
        if scan:
            scan_path = mesh_path + "scans/%04d/" % d 
            with open(scan_path + "cam_data.json") as f:
                cam_data = json.loads(f.read())
                cam_cnt = len(cam_data['transform'])
                near, width, height = cam_data['near'], cam_data['width'], cam_data['height']
            scan_path += "%04d"
        
        if not test:
            obj_path = mesh_path + "objs/%04d/" % d + "%04d.obj"
            vertices = None
            normals = None
            faces = None
            for t in range(frame_cnt*t_int):
                obj = readNumpyOBJ(obj_path%t)
                if t == 0:
                    vertices = np.empty((frame_cnt*t_int, obj[0].shape[0], 3))
                    normals = np.empty((frame_cnt*t_int,), dtype=object)
                    faces = np.empty((frame_cnt*t_int, obj[2].shape[0], 2, 4),dtype=int)
                vertices[t] = obj[0]
                normals[t] = obj[1]
                faces[t] = obj[2]

            min_v = np.min(vertices,axis=(0,1))
            max_v = np.max(vertices,axis=(0,1))
            scale = np.abs(max_v - min_v)

            vertices -= min_v + [0.5,0,0.5] * scale 
            vertices *= (res - 4 * bnd) / np.max(scale)
            vertices += [res/2, bnd*2, res/2]
            print(np.min(vertices,axis=(0,1)))
            print(np.max(vertices,axis=(0,1)))

            bary_coord = np.empty((len(faces[0]),),dtype=object)
            data_cnt = 0

        d_idx = None

        prev_ref = None
        prev_src = None

        prev_idx = None

        hdrsdf = OrderedDict([  ('dimX',lres),
                                ('dimY',lres),
                                ('dimZ',1 if dim == 2 else lres),
                                ('gridType', 16),
                                ('elementType',1),
                                ('bytesPerElement',4),
                                ('info',b'\0'*252),
                                ('dimT',1),
                                ('timestamp',(int)(time.time()*1e6))])
                                        
        hdr = OrderedDict([ ('dim',0),
                            ('dimX',res),
                            ('dimY',res),
                            ('dimZ',1 if dim == 2 else res),
                            ('elementType',0),
                            ('bytesPerElement',16),
                            ('info',b'\0'*256),
                            ('timestamp',(int)(time.time()*1e6))])
        hdrv = hdr.copy()
        hdrv['elementType'] = 1
        hdrv['bytesPerElement'] = 12
        
        for ci in range(cam_cnt):
            print("Load cam: %d/%d" % (ci+1, cam_cnt))
            if scan:
                viewWorld = np.array(cam_data['transform'][ci])   
                if os.path.isfile(scan_path%ci + ".npz"):
                    scan_data = np.load(scan_path%ci + ".npz")['arr_0']
                    if discretize:
                        scan_data = np.floor(np.clip(scan_data / max_z, 0, 1) * 256)/256 * max_z
                else:
                    scan_img_path = scan_path%ci + "/%04d.png"
                    tmp = max_z - imageio.imread(scan_img_path%0)[::-1,:,:1]/256 * max_z
                    
                    scan_data = np.empty((frame_cnt*t_int, tmp.shape[0], tmp.shape[1], 1))
                    scan_data[0] = tmp
                    for t in range(1, frame_cnt*t_int):
                        scan_data[t] = max_z - imageio.imread(scan_img_path%t)[::-1,:,:1]/256 * max_z
                viewV = np.dot(viewWorld[:3,:3], np.array([0,0,-1]))
                viewV = np.dot(np.array([[1,0,0],[0,0,1],[0,-1,0]]), viewV)

            for ti in range(1 if eval else t_int):
                print("Time Intervall: %d/%d" % (ti+1, t_int))
                d_idx = ti + (ci + d*cam_cnt)*(1 if eval else t_int) + off
                print("Dataset: %d" % d_idx)
                for t in range(frame_cnt):
                    t_off = ti+t*t_int
                    print("Load mesh: %d/%d (t_off: %d/%d)" % (t+1, frame_cnt, t_off, frame_cnt*t_int))
                    if not test:
                        if t == 0:
                            data_cnt = 0
                            for fi, f in enumerate(faces[t_off]):
                                v = vertices[t_off,f[0]]

                                area = np.linalg.norm(np.cross(v[1]-v[0], v[2]-v[0]))/2
                                area += np.linalg.norm(np.cross(v[2]-v[0], v[3]-v[0]))/2

                                par_cnt = vert_area_ratio * area
                                par_cnt = int(par_cnt) + int(np.random.random() < par_cnt % 1)
                                bary_coord[fi] = np.random.random((par_cnt, 2))
                                data_cnt += par_cnt

                        data = np.empty((data_cnt, 3))
                        di = 0
                        fltr_idx = np.zeros((data_cnt,), dtype="int32")
                        fltr_i = 0
                        for fi, f in enumerate(faces[t_off]):
                            v = vertices[t_off,f[0]]
                            n = normals[t_off][f[1]]

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
                                if not culling or np.dot(np.mean(n, 0), viewV) <= 0:
                                    fltr_idx[fltr_i] = di
                                    fltr_i += 1
                                di += 1

                        fltr_idx = fltr_idx[:fltr_i]

                        hdr['dim'] = fltr_i
                        hdr['dimX'] = res
                        hdr['dimY'] = res
                        hdr['dimZ'] = 1 if dim == 2 else res
                        
                        data_cull = data[fltr_idx]
                        writeParticlesUni(ref_path%(d_idx,t) +"_ps.uni", hdr, data_cull)
                        #writeNumpyRaw(ref_path%(ci + d*cam_cnt*t_int + off,t), data)
                        if debug:
                            writeNumpyOBJ(ref_path%(d_idx,t) +".obj", data_cull)
                        if t > 0:
                            hdrv['dim'] = len(prev_idx)
                            hdrv['dimX'] = res
                            hdrv['dimY'] = res
                            hdrv['dimZ'] = 1 if dim == 2 else res

                            vel = (data - prev_ref) * data_config['fps']
                            writeParticlesUni(ref_path%(d_idx,t-1) +"_pv.uni", hdrv, vel[prev_idx])
                            if debug:
                                writeNumpyOBJ(ref_path%(d_idx,t) +"_prev.obj", prev_ref[prev_idx])
                                writeNumpyOBJ(ref_path%(d_idx,t) +"_adv.obj", prev_ref[prev_idx]+vel[prev_idx]/data_config['fps'])
                            
                        
                    if not scan:
                        if t == 0:
                            d_idx = np.arange(fltr_i)
                            mask = np.ones(fltr_i, dtype=bool)
                            np.random.shuffle(d_idx)

                            idx_grid = ParticleIdxGrid(data_cull, (1 if dim == 2 else res, res, res))

                            for j in range(fltr_i):
                                p_idx = d_idx[j]
                                if mask[p_idx]:
                                    idx = np.array(idx_grid.get_range(data_cull[p_idx], 2*search_r))
                                    idx = idx[particle_radius(data_cull[idx], data_cull[p_idx], search_r)]
                                    if len(idx) < min_n:
                                        mask[p_idx] = False
                                    else:
                                        mask[idx] = False
                                        mask[p_idx] = True
                            d_idx = np.where(mask)[0]
                            print("particles reduced: %d -> %d (%.1f)" % (fltr_i, len(d_idx), fltr_i/len(d_idx)))
                        low_res_data = data_cull[d_idx]/factor_d

                        hdr['dim'] = len(low_res_data)
                        hdr['dimX'] = lres
                        hdr['dimY'] = lres
                        hdr['dimZ'] = 1 if dim == 2 else lres

                        for ci in range(cam_cnt):
                            writeParticlesUni(src_path%(d_idx,t) +"_ps.uni", hdr, low_res_data)
                            writeUni(src_path%(d_idx,t) +"_sdf.uni", hdrsdf, np.zeros((lres, lres, lres, 1)))
                            #writeNumpyRaw(src_path%(ci + d*cam_cnt*t_int + off,t-1), low_res_data)
                            if debug:
                                writeNumpyOBJ(src_path%(d_idx,t-1) +".obj", low_res_data)
                            if t > 0:
                                hdrv['dim'] = len(prev_src)
                                hdrv['dimX'] = lres
                                hdrv['dimY'] = lres
                                hdrv['dimZ'] = 1 if dim == 2 else lres

                                writeParticlesUni(src_path%(d_idx,t-1) +"_pv.uni", hdrv, (prev_src - low_res_data) * data_config['fps'])
                            
                        prev_src = low_res_data
                        
                    else:
                        a = scan_data[t_off]    
                        o = []
                        for j in range(a.shape[0]):
                            for i in range(a.shape[1]):
                                if a[j,i,0] < max_z:
                                    z = -a[j,i,0]
                                    x = (0.5 - i/a.shape[0]) * width * z / near
                                    y = (0.5 - j/a.shape[1]) * height * z / near
                                    o.append([x,y,z])

                        npo = np.dot(np.concatenate((np.asarray(o), np.ones((len(o),1))), axis=-1), viewWorld.T)[...,:3]
                        npo = np.dot(npo, np.array([[1,0,0],[0,0,1],[0,-1,0]]).T)
                        
                        npo -= min_v + [0.5,0,0.5] * scale 
                        npo *= (res - 4 * bnd) / np.max(scale)
                        npo += [res/2, bnd*2, res/2]
                        npo /= factor_d
                        print(np.min(npo,axis=0))
                        print(np.max(npo,axis=0))

                        hdr['dim'] = len(npo)
                        hdr['dimX'] = lres
                        hdr['dimY'] = lres
                        hdr['dimZ'] = 1 if dim == 2 else lres

                        writeParticlesUni(src_path%(d_idx,t) +"_ps.uni", hdr, npo)
                        writeUni(src_path%(d_idx,t) +"_sdf.uni", hdrsdf, np.zeros((lres, lres, lres, 1)))
                        if debug:
                            writeNumpyOBJ(src_path%(d_idx,t) +".obj", npo)
                        #writeNumpyRaw(src_path%(d_idx,t), npo)
                        
                        if t > 0:            
                            hdrv['dim'] = len(prev_src)
                            hdrv['dimX'] = lres
                            hdrv['dimY'] = lres
                            hdrv['dimZ'] = 1 if dim == 2 else lres
                            
                            if test:
                                vel = _approx_vel(prev_src, npo, 0.03*lres, it=6) * data_config['fps']
                            else:
                                dist = np.linalg.norm(np.expand_dims(prev_ref[prev_idx]/factor_d, axis=0) - np.expand_dims(prev_src, axis=1), axis=-1)
                                dist = np.exp(-dist/(0.01*lres))
                                w = np.clip(np.sum(dist, axis=1, keepdims=True), 1, 10000)
                                vel = np.dot(dist, vel[prev_idx])/w
                                vel /= factor_d

                                vel_test = _approx_vel(prev_src, npo, 0.03*lres, it=6) * data_config['fps']
                                print(np.mean(np.linalg.norm((vel_test-vel)/data_config['fps'], axis=-1)))
                                print("avg vel:" + str(np.mean(vel_test/data_config['fps'], axis=0)))

                            print("dist avg:" + str(np.mean(npo, axis=0) - np.mean(prev_src, axis=0)))
                            print("avg vel:" + str(np.mean(vel/data_config['fps'], axis=0)))

                            writeParticlesUni(src_path%(d_idx,t-1) +"_pv.uni", hdrv, vel)
                            if debug:
                                writeNumpyOBJ(src_path%(d_idx,t) +"_prev.obj", prev_src)
                                writeNumpyOBJ(src_path%(d_idx,t) +"_adv.obj", prev_src+vel/data_config['fps'])

                                print("EMD adv - npo: %f" % K.eval(emd_loss(K.constant(np.expand_dims(prev_src+vel/data_config['fps'], 0)), K.constant(np.expand_dims(npo, 0))))[0])

                        prev_src = npo
                        if not test:
                            print("particles reduced: %d -> %d (%.1f)" % (fltr_i, len(npo), fltr_i/len(npo)))

                    if not test:
                        prev_idx = fltr_idx
                        prev_ref = data