import json
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import *
import numpy as np
from neuralparticles.tools.uniio import readNumpyRaw

from neuralparticles.tools.plot_helpers import write_csv

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

    data_cnt = data_config['data_count']
    frame_cnt = data_config['frame_count']
    features = ['v','d','p']

    src_path = "%s%s_%s-%s_p" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"
    ref_path = "%s%s_%s-%s_ps" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"

    pad_cnt_src = np.empty((0,1))
    pad_cnt_ref = np.empty((0,1))

    for d in range(data_cnt):
        for t in range(frame_cnt):
            print("load patch: dataset(s): %03d timestep: %03d" % (d,t), end="\r", flush=True)
            src = readNumpyRaw(src_path % ('s',d,t))
            ref = readNumpyRaw(ref_path%(d,t))

            pad_cnt_src = np.concatenate((pad_cnt_src, np.expand_dims(np.count_nonzero(src[...,0] != pre_config['pad_val'],axis=1), axis=-1)))
            pad_cnt_ref = np.concatenate((pad_cnt_ref, np.expand_dims(np.count_nonzero(ref[...,0] != pre_config['pad_val'],axis=1), axis=-1)))

    write_csv(data_path + "statistics/%s_%s-%s_src_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']), pad_cnt_src)
    write_csv(data_path + "statistics/%s_%s-%s_ref_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']), pad_cnt_ref)