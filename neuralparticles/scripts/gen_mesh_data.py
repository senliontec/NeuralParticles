import numpy as np
from glob import glob
import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import particle_radius
from neuralparticles.tools.shell_script import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw, readNumpyOBJ, writeNumpyOBJ, readParticlesUni, writeUni
from neuralparticles.tools.particle_grid import ParticleIdxGrid
import keras.backend as K
from neuralparticles.tensorflow.losses.tf_approxmatch import approx_vel
import random
import math
from collections import OrderedDict
import time
import imageio

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
# 0: no scan, 1: only src scan, 2: gt scan, src downsampled
scan = int(getParam("scan", 0)) != 0
res = int(getParam("res", -1))
culling = int(getParam("cull", 0)) != 0
eval = int(getParam("eval", 0)) != 0
test = int(getParam("test", 0)) != 0
t_end = int(getParam("t_end", -1))

checkUnusedParams()

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

random.seed(data_config['seed'])
np.random.seed(data_config['seed'])

if test:
    src_path = "%sreal/%s_%s_" % (data_path, data_config['prefix'], data_config['id']) + "d%03d_%03d"

    if not os.path.exists(data_path + "real/"):
        os.makedirs(data_path + "real/")
    
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
        for t in range(frame_cnt):
            obj = readNumpyOBJ(obj_path%t)
            if t == 0:
                vertices = np.empty((frame_cnt, obj[0].shape[0], 3))
                normals = np.empty((frame_cnt,), dtype=object)
                faces = np.empty((frame_cnt, obj[2].shape[0], 2, 4),dtype=int)
            vertices[t] = obj[0]
            normals[t] = obj[1]
            faces[t] = obj[2]

        min_v = np.min(vertices,axis=(0,1))
        max_v = np.max(vertices,axis=(0,1))
        scale = max_v - min_v

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
            else:
                scan_img_path = scan_path%ci + "/%04d.png"
                tmp = 10 - imageio.imread(scan_img_path%0)[::-1,:,:1]/256 * 10
                
                scan_data = np.empty((frame_cnt, tmp.shape[0], tmp.shape[1], 1))
                scan_data[0] = tmp
                for t in range(1, frame_cnt):
                    scan_data[t] = 10 - imageio.imread(scan_img_path%t)[...,:1]/256 * 10
            viewV = np.dot(viewWorld[:3,:3], np.array([0,0,-1]))
            viewV = np.dot(np.array([[1,0,0],[0,0,1],[0,-1,0]]), viewV)

        for t in range(frame_cnt):
            if not test:
                print("Load mesh: %d/%d" % (t+1, frame_cnt))
                if t == 0:
                    data_cnt = 0
                    for fi, f in enumerate(faces[t]):
                        v = vertices[t,f[0]]

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
                for fi, f in enumerate(faces[t]):
                    v = vertices[t,f[0]]
                    n = normals[t][f[1]]

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
                
                writeParticlesUni(ref_path%(ci + d*cam_cnt + off,t) +"_ps.uni", hdr, data[fltr_idx])
                #writeNumpyRaw(ref_path%(ci + d*cam_cnt + off,t), data)
                if debug:
                    writeNumpyOBJ(ref_path%(ci + d*cam_cnt + off,t) +".obj", data[fltr_idx])
                if t > 0:
                    hdrv['dim'] = len(prev_idx)
                    hdrv['dimX'] = res
                    hdrv['dimY'] = res
                    hdrv['dimZ'] = 1 if dim == 2 else res

                    vel = (data - prev_ref) * data_config['fps']
                    writeParticlesUni(ref_path%(ci + d*cam_cnt + off,t-1) +"_pv.uni", hdrv, vel[prev_idx])
                    if debug:
                        writeNumpyOBJ(ref_path%(ci + d*cam_cnt + off,t) +"_prev.obj", prev_ref[prev_idx])
                        writeNumpyOBJ(ref_path%(ci + d*cam_cnt + off,t) +"_adv.obj", prev_ref[prev_idx]+vel[prev_idx]/data_config['fps'])
                    
                prev_idx = fltr_idx
                prev_ref = data
                
                data = data[fltr_idx]

            if not scan:
                if t == 0:
                    d_idx = np.arange(fltr_i)
                    mask = np.ones(fltr_i, dtype=bool)
                    np.random.shuffle(d_idx)

                    idx_grid = ParticleIdxGrid(data, (1 if dim == 2 else res, res, res))

                    for j in range(fltr_i):
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
                    print("particles reduced: %d -> %d (%.1f)" % (fltr_i, len(d_idx), fltr_i/len(d_idx)))
                low_res_data = data[d_idx]/factor_d

                hdr['dim'] = len(low_res_data)
                hdr['dimX'] = lres
                hdr['dimY'] = lres
                hdr['dimZ'] = 1 if dim == 2 else lres

                for ci in range(cam_cnt):
                    writeParticlesUni(src_path%(ci + d*cam_cnt + off,t) +"_ps.uni", hdr, low_res_data)
                    writeUni(src_path%(ci + d*cam_cnt + off,t) +"_sdf.uni", hdrsdf, np.zeros((lres, lres, lres, 1)))
                    #writeNumpyRaw(src_path%(ci + d*cam_cnt + off,t-1), low_res_data)
                    if debug:
                        writeNumpyOBJ(src_path%(ci + d*cam_cnt + off,t-1) +".obj", low_res_data)
                    if t > 0:
                        hdrv['dim'] = len(prev_src)
                        hdrv['dimX'] = lres
                        hdrv['dimY'] = lres
                        hdrv['dimZ'] = 1 if dim == 2 else lres

                        writeParticlesUni(src_path%(ci + d*cam_cnt + off,t-1) +"_pv.uni", hdrv, (prev_src - low_res_data) * data_config['fps'])
                    
                prev_src = low_res_data
                
            else:
                a = scan_data[t]    
                o = []
                for j in range(a.shape[0]):
                    for i in range(a.shape[1]):
                        if a[j,i,0] < 10.0:
                            z = -a[j,i,0]
                            x = (0.5 - i/a.shape[0]) * width * z / near
                            y = (0.5 - j/a.shape[1]) * height * z / near
                            o.append([x,y,z])
                
                if test:
                    min_v = np.array([-2,-2,-2])
                    max_v = np.array([2,2,2])

                    scale = max_v - min_v

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

                writeParticlesUni(src_path%(ci + d*cam_cnt + off,t) +"_ps.uni", hdr, npo)
                writeUni(src_path%(ci + d*cam_cnt + off,t) +"_sdf.uni", hdrsdf, np.zeros((lres, lres, lres, 1)))
                if debug:
                    writeNumpyOBJ(src_path%(ci + d*cam_cnt + off,t) +".obj", npo)
                #writeNumpyRaw(src_path%(ci + d*cam_cnt + off,t), npo)
                if t > 0:
                    hdrv['dim'] = len(prev_src)
                    hdrv['dimX'] = lres
                    hdrv['dimY'] = lres
                    hdrv['dimZ'] = 1 if dim == 2 else lres
                    
                    vel = K.eval(approx_vel(K.constant(np.expand_dims(prev_src, 0)), K.constant(np.expand_dims(npo, 0))))[0] * data_config['fps']
                    writeParticlesUni(src_path%(ci + d*cam_cnt + off,t-1) +"_pv.uni", hdrv, vel)
                    if debug:
                        writeNumpyOBJ(src_path%(ci + d*cam_cnt + off,t) +"_prev.obj", prev_src)
                        writeNumpyOBJ(src_path%(ci + d*cam_cnt + off,t) +"_adv.obj", prev_src+vel/data_config['fps'])

                prev_src = npo

                if not test:
                    print("particles reduced: %d -> %d (%.1f)" % (fltr_i, len(npo), fltr_i/len(npo)))
