import os, sys

import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras

from neuralparticles.tensorflow.models.PUNet import PUNet
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import load_patches_from_file, PatchExtractor, get_data_pair, extract_particles, get_nearest_idx, get_norm_factor
from neuralparticles.tensorflow.tools.eval_helpers import EvalCallback, EvalCompleteCallback, NthLogger
#from neuralparticles.tensorflow.tools.patch_generator import PatchGenerator
from neuralparticles.tensorflow.tools.patch_extract_generator import PatchGenerator

import numpy as np

data_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0
gpu = getParam("gpu", "-1")
chunk_size = int(getParam("chunk", 100))
checkpoint = int(getParam("checkpoint", 0))

plot_intervall = int(getParam("plt_i", 100))
eval_cnt = int(getParam("eval_cnt", 5))
eval_dataset = getParam("eval_d", []) #'18,18,18,19,19'
eval_t = getParam("eval_t", []) #'5,5,6,6,7'
eval_var = getParam("eval_v", []) #'0,0,0,0,0'
eval_patch_idx = getParam("eval_i", []) #'11,77,16,21,45'
eval_timesteps = int(getParam("eval_timesteps", 5))

if len(eval_dataset) > 0:
    eval_dataset = list(map(int, eval_dataset.split(',')))
if len(eval_t) > 0:
    eval_t = list(map(int, eval_t.split(',')))
if len(eval_var) > 0:
    eval_var = list(map(int, eval_var.split(',')))
if len(eval_patch_idx) > 0:
    eval_patch_idx = list(map(float, eval_patch_idx.split(',')))

checkUnusedParams()

src_path = data_path + "patches/source/"
ref_path = data_path + "patches/reference/"

model_path = data_path + "models/"
if not os.path.exists(model_path):
	os.mkdir(model_path)

tmp_folder = backupSources(data_path)
tmp_model_path = tmp_folder + "models/"
os.mkdir(tmp_model_path)
tmp_checkpoint_path = tmp_model_path + "checkpoints/"
os.mkdir(tmp_checkpoint_path)
tmp_eval_path = tmp_folder + "eval/"
os.mkdir(tmp_eval_path)

sys.stdout = Logger(tmp_folder + "logfile.log")

if not gpu is "-1":
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

config_dict = {**data_config, **pre_config, **train_config}
config_dict['norm_factor'] = get_norm_factor(data_path, config_path)
punet = PUNet(**config_dict)

pretrain = train_config['pretrain']

if len(eval_dataset) < eval_cnt:
    eval_dataset.extend(np.random.randint(data_config['data_count'], data_config['data_count'] + data_config['test_count'], eval_cnt-len(eval_dataset)))

if len(eval_t) < eval_cnt:
    t_start = min(train_config['t_start'], data_config['frame_count']-1)
    t_end = min(train_config['t_end'], data_config['frame_count'])
    eval_t.extend(np.random.randint(t_start, t_end, eval_cnt-len(eval_t)))

if len(eval_var) < eval_cnt:
    eval_var.extend([0]*(eval_cnt-len(eval_var)))

if len(eval_patch_idx) < eval_cnt:
    eval_patch_idx.extend(np.random.random(eval_cnt-len(eval_patch_idx)))
    
tmp_model_path = '%s%s_%s' % (tmp_model_path, data_config['prefix'], config['id']) 
tmp_checkpoint_path = '%s%s_%s' % (tmp_checkpoint_path, data_config['prefix'], config['id']) + "_{epoch:02d}.h5"
fig_path = '%s_loss' % tmp_model_path

print("Load Training Data")

if chunk_size > 0:
    patch_generator = PatchGenerator(data_path, config_path, chunk_size)
    val_generator = PatchGenerator(data_path, config_path, chunk_size, chunked_idx=patch_generator.get_val_idx())
    if train_config['loss_weights'][2] > 0.0 and pretrain:
        trunc_patch_generator = PatchGenerator(data_path, config_path, chunk_size, trunc=True)
        trunc_val_generator = PatchGenerator(data_path, config_path, chunk_size, chunked_idx=patch_generator.get_val_idx(), trunc=True)
else:
    src_data, ref_data = load_patches_from_file(data_path, config_path)
    src_data = np.concatenate(src_data, axis=-1)

    idx = np.arange(src_data.shape[0])
    np.random.shuffle(idx)
    src_data = src_data[idx]
    ref_data = ref_data[0][idx]


print("Load Eval Data")

np.random.seed(data_config['seed'])

factor_d = math.pow(pre_config['factor'], 1/data_config['dim'])

eval_patch_extractors = [[None for i in range(eval_timesteps)] for j in range(len(eval_dataset))]
eval_ref_datas = [[None for i in range(eval_timesteps)] for j in range(len(eval_dataset))]
eval_src_patches = [[None for i in range(eval_timesteps)] for j in range(len(eval_dataset))]
eval_ref_patches = [[None for i in range(eval_timesteps)] for j in range(len(eval_dataset))]

patch_size = pre_config['patch_size'] * data_config['res'] / factor_d
patch_size_ref = pre_config['patch_size_ref'] * data_config['res']
for i in range(len(eval_dataset)):
    #pos = None
    idx = None
    eval_patch_src, _, eval_patch_aux = get_data_pair(data_path, config_path, eval_dataset[i], eval_t[i], eval_var[i], features=['v'] if len(train_config['features']) == 0 else train_config['features'])[0]
    
    for j in range(eval_timesteps):
        (eval_src_data, eval_sdf_data, eval_par_aux), (eval_ref_data, eval_ref_sdf_data,_) = get_data_pair(data_path, config_path, eval_dataset[i], eval_t[i], eval_var[i]) 
        #eval_par_aux['p'] = np.sign(eval_par_aux['p'])*np.sqrt(np.abs(eval_par_aux['p']))
        eval_ref_datas[i][j] = eval_ref_data

        patch_extractor = PatchExtractor(eval_src_data, eval_sdf_data, patch_size, pre_config['par_cnt'], pre_config['surf'], pre_config['stride'], aux_data=eval_par_aux, features=train_config['features'], pad_val=pre_config['pad_val'], bnd=data_config['bnd']/factor_d, shuffle=False)
        eval_patch_extractors[i][j] = patch_extractor
        
        if idx is None:
            pos = patch_extractor.positions[int(eval_patch_idx[i] * len(patch_extractor.positions))]
            idx = get_nearest_idx(eval_src_data, pos)
            eval_ref_patch = extract_particles(eval_ref_data, pos * factor_d, pre_config['par_cnt_ref'], patch_size_ref/2, pre_config['pad_val'])[0]
        else:
            pos = eval_patch_src[idx]
            eval_ref_patch = np.ones((pre_config['par_cnt_ref'], 3)) * 100
        '''if pos is None:
            pos = patch_extractor.positions[int(eval_patch_idx[i] * len(patch_extractor.positions))]
        else:
            pos = pos + vel/data_config['fps']

        pos, vel = get_nearest_point(eval_src_data, pos, eval_par_aux)
        vel = vel['v']'''
        
        #eval_src_patch = patch_extractor.get_patch_pos(pos,False)
        eval_src_patch = extract_particles(eval_patch_src, pos, pre_config['par_cnt'], patch_size/2, pre_config['pad_val'], eval_patch_aux)
        if len(train_config['features']) > 0:
            eval_src_patch = [np.array([np.concatenate([eval_src_patch[0]] + [eval_src_patch[1][f] for f in train_config['features']],axis=-1)])]
        else:
            eval_src_patch = [np.array([eval_src_patch[0]])]

        eval_src_patches[i][j] = eval_src_patch
        eval_ref_patches[i][j] = eval_ref_patch

        print("Eval with dataset %d, timestep %d, var %d, patch pos (%f, %f, %f)" % (eval_dataset[i], eval_t[i]+j, eval_var[i], pos[0], pos[1], pos[2]))
        print("Eval trunc src: %d" % (np.count_nonzero(eval_src_patch[0][:,:,:1] != pre_config['pad_val'])))
        print("Eval trunc ref: %d" % (np.count_nonzero(eval_ref_patch[:,:1] != pre_config['pad_val'])))

        eval_patch_src = eval_patch_src + 0.01 * eval_patch_aux['v'] / (data_config['fps'] * pre_config['patch_size'])
        #eval_patch_aux['v'] *= 0.9

#src_data[1][:,:,-1] = np.sqrt(np.abs(src_data[1][:,:,-1])) * np.sign(src_data[1][:,:,-1])

punet.build_model()
#keras.utils.plot_model(punet.model, tmp_model_path + '.pdf', show_shapes=False, show_layer_names=True) 
punet.save_model(tmp_model_path+".h5")

if verbose:
    punet.model.summary()
else:
    print("Model parameter count: %d" % punet.model.count_params())

if checkpoint > 0:
    print("Load checkpoint: %scheckpoints/%s_%s_%02d.h5" % (model_path, data_config['prefix'], config['id'], checkpoint))
    punet.load_checkpoint('%scheckpoints/%s_%s_%02d.h5' % (model_path, data_config['prefix'], config['id'], checkpoint))


if chunk_size > 0:
    config_dict['generator'] = patch_generator
    config_dict['val_generator'] = val_generator
    if train_config['loss_weights'][2] > 0.0 and pretrain:
        config_dict['trunc_generator'] = trunc_patch_generator
        config_dict['trunc_val_generator'] = trunc_val_generator
else:
    config_dict['src'] = src_data
    config_dict['ref'] = ref_data
    
config_dict['callbacks'] = [(EvalCallback(tmp_eval_path + "eval_patch", eval_src_patches, eval_ref_patches, punet.model,
                                          train_config['features'], z=None if data_config['dim'] == 2 else 0, truncate=train_config['mask'], verbose=3 if verbose else 2)),
                            keras.callbacks.ModelCheckpoint(tmp_checkpoint_path), NthLogger(plot_intervall)]
config_dict['trunc_callbacks'] = [NthLogger(plot_intervall)]
''',
                            (EvalCompleteCallback(tmp_eval_path + "eval", eval_patch_extractors, eval_ref_datas,punet.model,
                                                  factor_d, data_config['res'], z=None if data_config['dim'] == 2 else data_config['res']//2, verbose=3 if verbose else 1))]'''
history = punet.train(**config_dict, build_model=False)
print(history.history)

m_p = "%s_trained.h5" % tmp_model_path
punet.save_model(m_p)

print("Saved Model: %s" % m_p)

legend = []
for k,v in history.history.items():
    plt.plot(v)
    legend.append(k)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(legend, loc='upper left')

plt.savefig(fig_path+".png")
plt.savefig(fig_path+".svg")

while(True):
    char = input("\nTrained Model only saved temporarily, do you want to save it? [y/n]\n")
    if char == "y" or char == "Y":
        from distutils.dir_util import copy_tree
        copy_tree(os.path.dirname(tmp_model_path), model_path)
        break
    elif char == "n" or char == "N":
        break
