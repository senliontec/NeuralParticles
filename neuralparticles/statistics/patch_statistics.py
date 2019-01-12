import json
import os
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import PatchExtractor, get_data, get_data_pair
import numpy as np
from neuralparticles.tools.uniio import readNumpyRaw
from neuralparticles.tensorflow.models.PUNet import PUNet
import shutil, math

from neuralparticles.tools.plot_helpers import write_csv, plot_particles

if __name__ == "__main__":
    data_path = getParam("data", "data/")
    config_path = getParam("config", "config/version_00.txt")
    use_network = int(getParam("net", 0)) != 0
    checkUnusedParams()

    if not os.path.exists(data_path + "statistics/"):
        os.makedirs(data_path + "statistics/")


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

    sample_path = data_path + "statistics/%s_%s-%s_sample_patches/" % (data_config['prefix'], data_config['id'], pre_config['id'])
    if os.path.exists(sample_path):
        shutil.rmtree(sample_path)

    os.makedirs(sample_path)

    d_start = data_config['data_count']
    d_end = data_config['data_count'] + data_config['test_count']

    fac_d = math.pow(pre_config['factor'], 1/data_config['dim'])

    par_cnt = pre_config['par_cnt']
    par_cnt_ref = pre_config['par_cnt_ref']

    patch_size = pre_config['patch_size'] * data_config['res'] / fac_d
    patch_size_ref = pre_config['patch_size_ref'] * data_config['res']

    pad_cnt_src = np.empty((0,1))
    pad_cnt_ref = np.empty((0,1))

    t_start = min(train_config['t_start'], data_config['frame_count']-1)
    t_end = min(train_config['t_end'], data_config['frame_count'])

    if use_network:
        config_dict = {**data_config, **pre_config, **train_config}
        punet = PUNet(**config_dict)
        punet.load_model(data_path + "models/%s_%s_trained.h5" % (data_config['prefix'], config['id']))
        features = train_config['features']
        pad_cnt_res = np.empty((0,1))
        out_res = np.ones((par_cnt+1,par_cnt_ref,3))*pre_config['pad_val']

    out_src = np.ones((par_cnt+1,par_cnt,3))*pre_config['pad_val']
    out_ref = np.ones((par_cnt+1,par_cnt_ref,3))*pre_config['pad_val']
    for d in range(d_start, d_end):
        for t in range(t_start, t_end):
            print("load patches: dataset(s): %03d timestep: %03d" % (d,t), end="\r", flush=True)
            (src_data, sdf_data, par_aux), (ref_data, ref_sdf_data, _) = get_data_pair(data_path, config_path, d, t, 0, features=train_config['features'])

            patch_ex_src = PatchExtractor(src_data, sdf_data, patch_size, par_cnt, pad_val=pre_config['pad_val'], positions=src_data, aux_data=par_aux, features=train_config['features'])
            patch_ex_ref = PatchExtractor(ref_data, ref_sdf_data, patch_size_ref, par_cnt_ref, pad_val=pre_config['pad_val'], positions=src_data*fac_d)
            src = patch_ex_src.get_patches()[0]
            ref = patch_ex_ref.get_patches()[0]

            if use_network:
                res = punet.predict(src)
                if type(res) is list:
                    cnt = (res[1] * res[0].shape[1]).astype(int)
                elif train_config['mask']:
                    cnt = np.expand_dims(np.count_nonzero(src[...,0] != pre_config['pad_val'],axis=1), axis=-1) * (res.shape[1]//src.shape[1])
                else:
                    cnt = np.zeros((res.shape[0], 1)) * res.shape[1]
                pad_cnt_res = np.concatenate((pad_cnt_res, (res[1] * res[0].shape[1]).astype(int)))
                res = res[0]

            cnt = np.expand_dims(np.count_nonzero(src[...,0] != pre_config['pad_val'],axis=1), axis=-1)
            pad_cnt_src = np.concatenate((pad_cnt_src, cnt))
            pad_cnt_ref = np.concatenate((pad_cnt_ref, np.expand_dims(np.count_nonzero(ref[...,0] != pre_config['pad_val'],axis=1), axis=-1)))

            for i in range(len(cnt)):
                if out_src[cnt[i],0,0] == pre_config['pad_val'] or np.random.random() < 0.5:
                    out_src[cnt[i]] = src[i,...,:3]
                    out_ref[cnt[i]] = ref[i]
                    if use_network:
                        out_res[cnt[i]] = res[i]

    for i in range(par_cnt):
        tmp_cnt = np.count_nonzero(out_ref[i,:,0] != pre_config['pad_val'])
        if tmp_cnt > 0:
            plot_particles(out_src[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_src.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None)
            plot_particles(out_ref[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_ref.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None)
            if use_network:
                plot_particles(out_res[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_res.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None)
                plot_particles(out_res[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_comp.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None, src=out_src[i], ref=out_ref[i])
            else:
                plot_particles(out_ref[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_comp.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None, src=out_src[i])
 
    write_csv(data_path + "statistics/%s_%s-%s_src_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']), pad_cnt_src)
    write_csv(data_path + "statistics/%s_%s-%s_ref_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']), pad_cnt_ref)

    if use_network:
        write_csv(data_path + "statistics/%s_%s-%s_res_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']), pad_cnt_res)
