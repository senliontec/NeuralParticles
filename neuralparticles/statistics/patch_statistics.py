import json
import os
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import *
import numpy as np
from neuralparticles.tools.uniio import readNumpyRaw
import shutil

from neuralparticles.tools.plot_helpers import write_csv, plot_particles

if __name__ == "__main__":
    data_path = getParam("data", "data/")
    config_path = getParam("config", "config/version_00.txt")
    checkUnusedParams()

    if not os.path.exists(data_path + "statistics/"):
        os.makedirs(data_path + "statistics/")


    src_path = data_path + "patches/source/"
    ref_path = data_path + "patches/reference/"

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

    data_cnt = data_config['data_count']
    frame_cnt = data_config['frame_count']
    features = ['v','d','p']

    src_path = "%s%s_%s-%s_p" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"
    ref_path = "%s%s_%s-%s_ps" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"

    pad_cnt_src = np.empty((0,1))
    pad_cnt_ref = np.empty((0,1))

    t_start = min(train_config['t_start'], data_config['frame_count']-1)
    t_end = min(train_config['t_end'], data_config['frame_count'])

    out_src = np.ones((pre_config['par_cnt']+1,pre_config['par_cnt'],3))*pre_config['pad_val']
    out_ref = np.ones((pre_config['par_cnt']+1,pre_config['par_cnt_ref'],3))*pre_config['pad_val']
    for d in range(data_cnt):
        for t in range(t_start, t_end):
            print("load patch: dataset(s): %03d timestep: %03d" % (d,t), end="\r", flush=True)
            src = readNumpyRaw(src_path % ('s',d,t))
            ref = readNumpyRaw(ref_path%(d,t))

            cnt = np.expand_dims(np.count_nonzero(src[...,0] != pre_config['pad_val'],axis=1), axis=-1)
            pad_cnt_src = np.concatenate((pad_cnt_src, cnt))
            pad_cnt_ref = np.concatenate((pad_cnt_ref, np.expand_dims(np.count_nonzero(ref[...,0] != pre_config['pad_val'],axis=1), axis=-1)))

            for i in range(len(cnt)):
                if out_src[cnt[i],0,0] == pre_config['pad_val'] or np.random.random() < 0.5:
                    out_src[cnt[i]] = src[i]
                    out_ref[cnt[i]] = ref[i]

    for i in range(pre_config['par_cnt']):
        tmp_cnt = np.count_nonzero(out_ref[i,:,0] != pre_config['pad_val'])
        if tmp_cnt > 0:
            plot_particles(out_src[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_src.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None)
            plot_particles(out_ref[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_ref.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None)
            plot_particles(out_ref[i], [-1,1], [-1,1], 5, sample_path + "%06d_%06d_comp.svg"%(i,tmp_cnt), z= 0 if data_config['dim'] == 3 else None, src=out_src[i])
 
    write_csv(data_path + "statistics/%s_%s-%s_src_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']), pad_cnt_src)
    write_csv(data_path + "statistics/%s_%s-%s_ref_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']), pad_cnt_ref)