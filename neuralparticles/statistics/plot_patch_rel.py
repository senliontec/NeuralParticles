import matplotlib.pyplot as plt
import os
from neuralparticles.tools.plot_helpers import read_csv
import numpy as np
import json
from neuralparticles.tools.param_helpers import *

if __name__ == "__main__":
    data_path = getParam("data", "data/")
    config_path = getParam("config", "config/version_00.txt")
    net = int(getParam("net", 0)) != 0
    checkUnusedParams()

    if not os.path.exists(data_path + "statistics/"):
        os.makedirs(data_path + "statistics/")

    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
        train_config = json.loads(f.read())

    pad_cnt_src = read_csv(data_path + "statistics/%s_%s-%s_src_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']))
    if net:
        pad_cnt_ref = read_csv(data_path + "statistics/%s_%s-%s-%s_res_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id'], train_config['id']))
    else:
        pad_cnt_ref = read_csv(data_path + "statistics/%s_%s-%s_ref_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']))

    grouped_data = np.zeros((pre_config['par_cnt_ref']+1, pre_config['par_cnt']+1))

    print(np.mean(pad_cnt_ref/pad_cnt_src))

    for i in range(len(pad_cnt_src)):
        grouped_data[int(min(pad_cnt_ref[i,0],pre_config['par_cnt_ref'])), int(min(pad_cnt_src[i,0],pre_config['par_cnt']))] += 1

    #grouped_data/=np.max(grouped_data, axis=0, keepdims=True)+1e-10

    plt.imshow(grouped_data, cmap='Reds', aspect='auto', origin='lower')

    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')

    X = np.arange(pre_config['par_cnt_ref']+1)
    Y = np.arange(pre_config['par_cnt']+1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X,Y,grouped_data, cmap='hot')"""
    plt.show()