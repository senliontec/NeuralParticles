import json
import os
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import PatchExtractor, get_data, get_data_pair, get_norm_factor, get_nearest_idx
import numpy as np
from neuralparticles.tools.uniio import readNumpyRaw
from neuralparticles.tensorflow.models.PUNet import PUNet
import shutil, math

import keras
from keras.models import Model

from neuralparticles.tools.plot_helpers import write_csv, plot_particles

if __name__ == "__main__":
    data_path = getParam("data", "data/")
    config_path = getParam("config", "config/version_00.txt")
    mode = int(getParam("mode", 0))
    cnt = int(getParam("cnt", 100))
    patch_seed = int(getParam("seed", 0))
    dataset = int(getParam("dataset", -1))
    checkUnusedParams()

    if not os.path.exists(data_path + "statistics/"):
        os.makedirs(data_path + "statistics/")

    np.random.seed(34)

    src_path = data_path + "source/"
    ref_path = data_path + "reference/"

    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
        train_config = json.loads(f.read())


    fac_d = math.pow(pre_config['factor'], 1/data_config['dim'])

    par_cnt = pre_config['par_cnt']
    par_cnt_ref = pre_config['par_cnt_ref']

    patch_size = pre_config['patch_size'] * data_config['res'] / fac_d
    patch_size_ref = pre_config['patch_size_ref'] * data_config['res']

    norm_factor = get_norm_factor(data_path, config_path)

    config_dict = {**data_config, **pre_config, **train_config}
    punet = PUNet(**config_dict)
    punet.load_model(data_path + "models/%s_%s_trained.h5" % (data_config['prefix'], config['id']))

    features = Model(inputs=punet.model.inputs, outputs=[punet.model.get_layer("concatenate_1").output])

    if len(punet.model.inputs) > 1:
        print("Too many inputs!")
        exit()

    if mode < 10: # synthetic data
        input_points = np.ones((cnt, par_cnt, punet.model.inputs[0].get_shape()[-1])) * (-2)
        if mode == 0:
            for i in range(cnt):
                input_points[i,0,:3] = i * 0.01
        elif mode == 1:
            for i in range(cnt):
                input_points[i,0,:3] = 0.5
                input_points[i,0,3:6] = [i*norm_factor[0]/cnt,0,0]
        else:
            print("Mode %d not supported!" % mode)

        output = features.predict(input_points)[:,0]
        write_csv(data_path + "statistics/%s_%s-%s-%s_m%02d_features.csv"%(data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], mode), output)

    if mode >= 10: # real data
        d_start = 0
        d_end = data_config['test_count']
        if dataset < 0:
            dataset = d_start

        t_start = 0
        t_end = data_config['frame_count']

        output = None
        if mode == 10 or mode == 11:
            np.random.seed(patch_seed)
            patch_pos = np.random.random((cnt, 3)) * data_config['res'] / fac_d
            input_points = np.ones(((t_end-t_start), cnt, par_cnt, punet.model.inputs[0].get_shape()[-1])) * (-2)
            src_data = None
            p_idx = 0

            patch_path = data_path + "statistics/%s_%s-%s-%s_m%02d_patches/"%(data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], mode)
            if not os.path.exists(patch_path):
                os.makedirs(patch_path)
            patch_path += "patch_%04d.png"
            print(t_end)
            for t in range(t_start, t_end):
                print(t)
                path_src = "%sreal/%s_%s_d%03d_%03d" % (data_path, data_config['prefix'], data_config['id'], dataset, t)
                src_data, sdf_data, par_aux = get_data(path_src, par_aux=train_config['features'])

                patch_extractor = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pre_config['surf'], 0, aux_data=par_aux, features=train_config['features'], pad_val=pre_config['pad_val'], bnd=data_config['bnd']/fac_d, last_pos=patch_pos, stride_hys=1.0, shuffle=True)
                patch_pos = patch_extractor.positions
                patch = patch_extractor.pop_patch()
                print(t)

                ci = np.argmin(np.linalg.norm(patch[...,:3], axis=-1))
                cv = np.copy(patch[0])
                patch[0] = patch[ci]
                patch[ci] = cv

                plot_particles(patch_extractor.positions, [0,int(data_config['res']/fac_d)], [0,int(data_config['res']/fac_d)], 5, patch_path%t, src=patch_pos, z=patch_pos[0][2] if data_config['dim'] == 3 else None)
                patch_pos += par_aux['v'][patch_extractor.pos_idx] / data_config['fps']
                input_points[p_idx] = patch
                p_idx += 1

            if mode == 11:
                np.random.seed(563)
                np.random.shuffle(input_points)
            output = features.predict(np.reshape(input_points, (-1, par_cnt, punet.model.inputs[0].get_shape()[-1])))[:,0,:-8]
            print(output.shape)
            output = np.mean(np.reshape(output, (t_end-t_start, cnt, -1)), axis=(1,2))
            output = np.expand_dims(output, axis=-1)
            print(output.shape)
            output -= np.mean(output, axis=0)
            output /= np.max(np.abs(output))

            write_csv(data_path + "statistics/%s_%s-%s-%s_m%02d_features.csv"%(data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], mode), output)
            hann = np.expand_dims(np.hanning(output.shape[0]), axis=-1)
            output = np.fft.fft(hann*output, axis=0)
            N = len(output)//2+1
            output = np.abs(output[:N])/N
            write_csv(data_path + "statistics/%s_%s-%s-%s_m%02d_features_fft.csv"%(data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], mode), output)
        elif mode == 12: 
            dataset = np.random.randint(d_start, d_end, cnt)
            timestep = np.random.randint(t_start, t_end, cnt)
            input_points = np.ones((cnt, par_cnt, punet.model.inputs[0].get_shape()[-1])) * (-2)

            for p_idx in range(cnt):
                path_src = "%sreal/%s_%s_d%03d_%03d" % (data_path, data_config['prefix'], data_config['id'], dataset[p_idx], timestep[p_idx])
                src_data, sdf_data, par_aux = get_data(path_src, par_aux=train_config['features'])                    
                patch_ex_src = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pad_val=pre_config['pad_val'], aux_data=par_aux, features=train_config['features'], surface = pre_config['surf'], bnd=data_config['bnd']/fac_d, stride=0)
                patch = patch_ex_src.get_patch(0, False)

                ci = np.argmin(np.linalg.norm(patch[...,:3], axis=-1))
                cv = np.copy(patch[0])
                patch[0] = patch[ci]
                patch[ci] = cv

                input_points[p_idx] = patch
            output = features.predict(input_points)[:,0]
            output -= np.mean(output, axis=0)
            output /= np.max(np.abs(output[:,:-9]))

            write_csv(data_path + "statistics/%s_%s-%s-%s_m%02d_features.csv"%(data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], mode), output)
            hann = np.expand_dims(np.hanning(output.shape[0]), axis=-1)
            output = np.fft.fft(hann*output, axis=0)
            N = len(output)//2+1
            output = np.abs(output[:N])/N
            write_csv(data_path + "statistics/%s_%s-%s-%s_m%02d_features_fft.csv"%(data_config['prefix'], data_config['id'], pre_config['id'], train_config['id'], mode), output)
        elif mode == 13: 
            output = np.empty((10000, features.output.get_shape()[-1]))
            idx = 0
            for d in range(d_start, d_end):
                for t in range(t_start, t_end):
                    path_src = "%sreal/%s_%s_d%03d_%03d" % (data_path, data_config['prefix'], data_config['id'], d, t)
                    src_data, sdf_data, par_aux = get_data(path_src, par_aux=train_config['features'])                    
                    patch_ex_src = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pad_val=pre_config['pad_val'], aux_data=par_aux, features=train_config['features'], surface = pre_config['surf'], bnd=data_config['bnd']/fac_d, stride=0)
                    src = patch_ex_src.get_patches()[0]

                    res = features.predict(src)[:,0]
                    print(idx)
                    if idx + len(res) > len(output):
                        ouput = np.concatenate((output, np.empty_like(output)))
                    idx+=len(res)
                    output[idx:idx+len(res)] = res            
            
        else:
            print("Mode %d not supported!" % mode)
    
    
    """out_src = np.ones((par_cnt+1,par_cnt,3))*pre_config['pad_val']
    out_ref = np.ones((par_cnt+1,par_cnt_ref,3))*pre_config['pad_val']
    idx = 0
    for d in range(d_start, d_end):
        for t in range(t_start, t_end):
            print("load patches: dataset(s): %03d timestep: %03d" % (d,t), end="\r", flush=True)
            (src_data, sdf_data, par_aux), (ref_data, ref_sdf_data, _) = get_data_pair(data_path, config_path, d, t, 0, features=train_config['features'])

            patch_ex_src = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pad_val=pre_config['pad_val'], aux_data=par_aux, features=train_config['features'], surface = pre_config['surf'], bnd=data_config['bnd']/fac_d, stride=0)
            patch_ex_ref = PatchExtractor(ref_data, ref_sdf_data, patch_size_ref, par_cnt_ref, pad_val=pre_config['pad_val'], positions=patch_ex_src.positions*fac_d)
            src = patch_ex_src.get_patches()[0]
            ref = patch_ex_ref.get_patches()[0]

            if idx+len(src) > len(pad_cnt_src):
                pad_cnt_src = np.concatenate((pad_cnt_src, np.empty_like(pad_cnt_src)))
                pad_cnt_ref = np.concatenate((pad_cnt_ref, np.empty_like(pad_cnt_ref)))
                pad_cnt_res = np.concatenate((pad_cnt_res, np.empty_like(pad_cnt_res)))

            res = punet.predict(src)
            if type(res) is list:
                cnt = (res[1] * res[0].shape[1]).astype(int)
                res = res[0]
            elif train_config['mask']:
                cnt = np.expand_dims(np.count_nonzero(src[...,0] != pre_config['pad_val'],axis=1), axis=-1) * (res.shape[1]//src.shape[1])
            else:
                cnt = np.zeros((res.shape[0], 1)) * res.shape[1]
            pad_cnt_res[idx:idx+len(cnt)] = cnt

            cnt = np.expand_dims(np.count_nonzero(src[...,0] != pre_config['pad_val'],axis=1), axis=-1)

            pad_cnt_src[idx:idx+len(cnt)] = cnt
            pad_cnt_ref[idx:idx+len(cnt)] = np.expand_dims(np.count_nonzero(ref[...,0] != pre_config['pad_val'],axis=1), axis=-1)

            idx += len(cnt)

            for i in range(len(cnt)):
                if out_src[cnt[i],0,0] == pre_config['pad_val'] or np.random.random() < 0.5:
                    out_src[cnt[i]] = src[i,...,:3]
                    out_ref[cnt[i]] = ref[i]
                    out_res[cnt[i]] = res[i]"""

    """pad_cnt_src = pad_cnt_src[:idx]
    pad_cnt_ref = pad_cnt_ref[:idx]
    pad_cnt_res = pad_cnt_res[:idx]

    for i in range(par_cnt):
        tmp_cnt = np.count_nonzero(out_ref[i,:,0] != pre_config['pad_val'])
        if tmp_cnt > 0:
            plot_particles(out_src[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_src.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None)
            plot_particles(out_ref[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_ref.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None)
            plot_particles(out_res[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_res.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None)

    write_csv(data_path + "statistics/%s_%s-%s_src_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']), pad_cnt_src)
    write_csv(data_path + "statistics/%s_%s-%s_ref_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']), pad_cnt_ref)

    write_csv(data_path + "statistics/%s_%s-%s-%s_res_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id'], train_config['id']), pad_cnt_res)"""
