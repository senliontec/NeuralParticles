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

    pad_cnt_src = read_csv(data_path + "statistics/%s_%s-%s_src_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id']))
    pad_cnt_ref = read_csv(data_path + "statistics/%s_%s-%s_%s_patch_cnt.csv"%(data_config['prefix'], data_config['id'], pre_config['id'], "res" if net else "ref"))

    grouped_data = np.zeros((pre_config['par_cnt']+1, pre_config['par_cnt_ref']+1))

    print(np.mean(pad_cnt_ref/pad_cnt_src))

    for i in range(len(pad_cnt_src)):
        grouped_data[int(min(pad_cnt_src[i,0],pre_config['par_cnt'])), int(min(pad_cnt_ref[i,0],pre_config['par_cnt_ref']))] += 1

    #grouped_data/=np.max(grouped_data, axis=0, keepdims=True)+1e-10

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')

    X = np.arange(pre_config['par_cnt_ref']+1)
    Y = np.arange(pre_config['par_cnt']+1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X,Y,grouped_data, cmap='hot')
    plt.show()