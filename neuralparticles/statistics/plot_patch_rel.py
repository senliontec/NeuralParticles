import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from neuralparticles.tools.plot_helpers import read_csv
import numpy as np
import json
from neuralparticles.tools.param_helpers import *

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

    pad_cnt_src = read_csv(data_path + "statistics/%s_%s-%s_src_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']))
    pad_cnt_ref = read_csv(data_path + "statistics/%s_%s-%s_ref_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']))

    grouped_data = np.zeros((pre_config['par_cnt']+1, pre_config['par_cnt_ref']+1))

    for i in range(len(pad_cnt_src)):
        grouped_data[int(min(pad_cnt_src[i,0],pre_config['par_cnt']+1)), int(min(pad_cnt_ref[i,0],pre_config['par_cnt_ref']+1))] += 1

    #grouped_data/=np.max(grouped_data, axis=0, keepdims=True)+1e-10

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')

    X = np.arange(pre_config['par_cnt_ref']+1)
    Y = np.arange(pre_config['par_cnt']+1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X,Y,grouped_data, cmap='hot')
    plt.show()