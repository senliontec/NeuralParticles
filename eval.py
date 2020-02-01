import os

import time
from collections import OrderedDict

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from neuralparticles.tensorflow.models.PUNet import PUNet

from neuralparticles.tools.data_helpers import PatchExtractor, get_data_pair, extract_particles, in_bound, get_data, get_nearest_idx, get_norm_factor
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.uniio import writeParticlesUni, writeNumpyRaw
from neuralparticles.tensorflow.tools.patch_extract_generator import PatchGenerator


data_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")
gpu = getParam("gpu", "")

real = int(getParam("real", 0)) != 0

checkpoint = int(getParam("checkpoint", -1))

checkUnusedParams()

if not gpu is "":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

if checkpoint > 0:
    model_path = data_path + "models/checkpoints/%s_%s_%02d.h5" % (data_config['prefix'], config['id'], checkpoint)
else:
    model_path = data_path + "models/%s_%s_trained.h5" % (data_config['prefix'], config['id'])

config_dict = {**data_config, **pre_config, **train_config}
config_dict['norm_factor'] = get_norm_factor(data_path, config_path)
tmp_w = train_config["loss_weights"]
if tmp_w[1] <= 0.0:
    tmp_w[1] = 1.0
punet = PUNet(**config_dict)
punet.build_model()
punet.load_model(model_path)

patch_generator = PatchGenerator(data_path, config_path, 100, eval=True)

metrics_names = punet.train_model.metrics_names
metrics_values = punet.eval(patch_generator)

for i in range(len(metrics_names)):
    print("%s: %f" % (metrics_names[i], metrics_values[i]))

