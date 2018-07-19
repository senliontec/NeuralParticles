import os

import json
import math

from neuralparticles.tensorflow.tools.hyper_parameter import HyperParameter, ValueType, SearchType
from neuralparticles.tensorflow.tools.hyper_search import HyperSearch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras

from neuralparticles.tensorflow.models.PUNet import PUNet
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import load_patches_from_file, PatchExtractor, get_data_pair, extract_particles
from neuralparticles.tensorflow.tools.eval_helpers import EvalCallback, EvalCompleteCallback

import numpy as np

data_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0
gpu = getParam("gpu", "")
epochs = int(getParam("epochs", 0))

eval_cnt = int(getParam("eval_cnt", 5))
eval_dataset = getParam("eval_d", []) #'18,18,18,19,19'
eval_t = getParam("eval_t", []) #'5,5,6,6,7'
eval_var = getParam("eval_v", []) #'0,0,0,0,0'
eval_patch_idx = getParam("eval_i", []) #'11,77,16,21,45'

if len(eval_dataset) > 0:
    eval_dataset = list(map(int, eval_dataset.split(',')))
if len(eval_t) > 0:
    eval_t = list(map(int, eval_t.split(',')))
if len(eval_var) > 0:
    eval_var = list(map(int, eval_var.split(',')))
if len(eval_patch_idx) > 0:
    eval_patch_idx = list(map(float, eval_patch_idx.split(',')))

i=0
hyper_teams = []
while(True):
    hyper_par = getParam("hyper%d"%i, None)
    i += 1
    if hyper_par is None:
        break
    else:
        hyper_teams.append(HyperParameter.parse(hyper_par))

checkUnusedParams()

src_path = data_path + "patches/source/"
ref_path = data_path + "patches/reference/"

model_path = data_path + "models/"
if not os.path.exists(model_path):
	os.mkdir(model_path)

tmp_folder = backupSources(data_path)
tmp_model_path = tmp_folder + "models/"
os.mkdir(tmp_model_path)
tmp_eval_path = tmp_folder + "eval/"
os.mkdir(tmp_eval_path)

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

if verbose:
    print("Config Loaded:")
    print(config)
    print(data_config)
    print(pre_config)
    print(train_config)

# copy config files into tmp

np.random.seed(data_config['seed'])
#tf.set_random_seed(data_config['seed'])

if epochs == 0:
    epochs = train_config['epochs']

config_dict = {**data_config, **pre_config, **train_config}
punet = PUNet(**config_dict)

if len(eval_dataset) < eval_cnt:
    eval_dataset.extend(np.random.randint(int(data_config['data_count'] * train_config['train_split']), data_config['data_count'], eval_cnt-len(eval_dataset)))

if len(eval_t) < eval_cnt:
    t_start = min(train_config['t_start'], data_config['frame_count']-1)
    t_end = min(train_config['t_end'], data_config['frame_count'])
    eval_t.extend(np.random.randint(t_start, t_end, eval_cnt-len(eval_t)))

if len(eval_var) < eval_cnt:
    eval_var.extend([0]*(eval_cnt-len(eval_var)))

if len(eval_patch_idx) < eval_cnt:
    eval_patch_idx.extend(np.random.random(eval_cnt-len(eval_patch_idx)))
    
tmp_model_path = '%s%s_%s' % (tmp_model_path, data_config['prefix'], config['id']) 
fig_path = '%s_loss' % tmp_model_path

src_path = "%s%s_%s-%s" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_pvar%02d_%03d"
ref_path = "%s%s_%s-%s" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_pvar%02d_%03d"
print(src_path)
print(ref_path)

print("Load Training Data")

src_data, ref_data = load_patches_from_file(data_path, config_path)

idx = np.arange(src_data[0].shape[0])
np.random.shuffle(idx)
src_data = [s[idx] for s in src_data]
ref_data = ref_data[idx]

print("Load Eval Data")

factor_d = math.pow(pre_config['factor'], 1/data_config['dim'])

eval_patch_extractors = []
eval_ref_datas = []
eval_src_patches = []
eval_ref_patches = []
for i in range(len(eval_dataset)):
    (eval_src_data, eval_sdf_data, eval_par_aux), (eval_ref_data, eval_ref_sdf_data) = get_data_pair(data_path, config_path, eval_dataset[i], eval_t[i], eval_var[i]) 
    eval_ref_datas.append(eval_ref_data)
    np.random.seed(100)
    eval_patch_extractors.append(PatchExtractor(eval_src_data, eval_sdf_data, pre_config['patch_size'], pre_config['par_cnt'], pre_config['surf'], pre_config['stride'], aux_data=eval_par_aux, features=train_config['features'], pad_val=pre_config['pad_val'], bnd=data_config['bnd']/factor_d))
    p_idx = int(eval_patch_idx[i] * len(eval_patch_extractors[i].positions))
    eval_src_patches.append(eval_patch_extractors[i].get_patch(p_idx,False))
    eval_ref_patches.append(extract_particles(eval_ref_data, eval_patch_extractors[i].positions[p_idx] * factor_d, pre_config['par_cnt_ref'], pre_config['patch_size_ref']/2, pre_config['pad_val'])[0])

    print("Eval with dataset %d, timestep %d, var %d, patch idx %d" % (eval_dataset[i], eval_t[i], eval_var[i], p_idx))
    print("Eval trunc src: %d" % (np.count_nonzero(eval_src_patches[i][0][:,:,:1] != pre_config['pad_val'])))
    print("Eval trunc ref: %d" % (np.count_nonzero(eval_ref_patches[i][:,:1] != pre_config['pad_val'])))

config_dict['src'] = src_data
config_dict['ref'] = ref_data
config_dict['callbacks'] = [(EvalCallback(tmp_eval_path + "eval_patch", eval_src_patches, eval_ref_patches,
                                          train_config['features'], multiple_runs=True, z=None if data_config['dim'] == 2 else 0, verbose=1)),
                            (EvalCompleteCallback(tmp_eval_path + "eval", eval_patch_extractors, eval_ref_datas,
                                                  factor_d, data_config['res'], multiple_runs=True, z=None if data_config['dim'] == 2 else data_config['res']//2, verbose=1))]
hs = HyperSearch(punet, hyper_teams, output_folder=tmp_folder)
del config_dict['epochs']
history = hs.search(epochs, **config_dict)

keras.utils.plot_model(punet.model, tmp_model_path + '.pdf') 

print(history)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.savefig(fig_path+".png")
plt.savefig(fig_path+".pdf")

