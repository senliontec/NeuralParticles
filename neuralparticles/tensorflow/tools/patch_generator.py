import json
import os
import keras

from neuralparticles.tools.data_helpers import extract_particles, get_positions, get_data
from neuralparticles.tools.particle_grid import ParticleIdxGrid

from neuralparticles.tools.uniio import readNumpyRaw

import numpy as np

class PatchGenerator(keras.utils.Sequence):
    def __init__(self, data_path, config_path,
                 d_start=-1, d_end=-1, t_start=-1, t_end=-1, idx=None):
        np.random.seed(45)
        with open(config_path, 'r') as f:
            config = json.loads(f.read())

        with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
            data_config = json.loads(f.read())

        with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
            pre_config = json.loads(f.read())

        with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
            train_config = json.loads(f.read())

        self.d_start = 0 if d_start < 0 else d_start
        self.d_end = int(data_config['data_count'] * train_config['train_split']) if d_end < 0 else d_end
        self.t_start = train_config['t_start'] if t_start < 0 else t_start
        self.t_end = train_config['t_end'] if t_end < 0 else t_end

        self.trunc = train_config['truncate']
        self.batch_size = train_config['batch_size']

        self.features = train_config['features']

        self.par_cnt = pre_config['par_cnt']
        self.par_cnt_ref = pre_config['par_cnt_ref']
        self.pad_val = pre_config['pad_val']

        self.src_path = "%spatches/source/%s_%s-%s_p" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"
        self.ref_path = "%spatches/reference/%s_%s-%s_ps" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"

        if idx is None:
            patch_cnt = np.empty(((self.d_end-self.d_start), (self.t_end-self.t_start)), dtype=object)
            
            for d in range(self.d_start, self.d_end):
                for t in range(self.t_start, self.t_end):
                    patch_cnt[d-self.d_start,t-self.t_start] = len(readNumpyRaw(self.src_path % ('s',d,t)))
            
            self.idx = np.array([[x,y,z] for x in range(self.d_start, self.d_end) for y in range(self.t_start, self.t_end) for z in range(patch_cnt[x-self.d_start,y-self.t_start])])

            np.random.shuffle(self.idx)

            train_data_cnt = int((1 - train_config['val_split']) * len(self.idx))
            self.idx, self.val_idx = self.idx[:train_data_cnt], self.idx[train_data_cnt:]
        else:
            self.idx = idx
            self.val_idx = None
        
        self.on_epoch_end()


    def get_val_idx(self):
        return self.val_idx


    def __len__(self):
        return int(np.floor(len(self.idx)/self.batch_size))


    def __getitem__(self, index):
        return self._data_generation(index)


    def on_epoch_end(self):
        np.random.shuffle(self.idx)


    def _data_generation(self, index):
        b_idx = self.idx[index*self.batch_size:(index+1)*self.batch_size]

        src = [np.empty((self.batch_size, self.par_cnt, 3))]
        if len(self.features) > 0:
            src.append(np.empty((self.batch_size, self.par_cnt, len(self.features) + 2 if 'v' in self.features else 0)))

        ref = [np.empty((self.batch_size,self.par_cnt_ref,3))]
        if self.trunc:
            ref.append(np.empty((self.batch_size,1)))

        for i in range(self.batch_size):
            d,t,p = b_idx[i]
            src[0][i] = readNumpyRaw(self.src_path%('s',d,t))[p]
            if len(self.features) > 0:
                src[1][i] = np.concatenate([readNumpyRaw(self.src_path%(f,d,t))[p] for f in self.features], axis=-1)
            ref[0][i] = readNumpyRaw(self.ref_path%(d,t))[p]
            if self.trunc:
                ref[1][i] = np.count_nonzero(ref[0][i,:,:1] != self.pad_val, axis=0)/self.par_cnt_ref 
        return src, ref