import json
import os
import keras

from neuralparticles.tools.data_helpers import extract_particles, get_positions, get_data
from neuralparticles.tools.particle_grid import ParticleIdxGrid

import numpy as np

class PatchGenerator(keras.utils.Sequence):
    def __init__(self, data_path, config_path, chunk_size, data_iterations,
                 d_start=-1, d_end=-1, t_start=-1, t_end=-1, v_start=-1, v_end=-1):
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
        self.v_start = 0 if v_start < 0 else v_start
        self.v_end = pre_config['var'] if v_end < 0 else v_end

        self.trunc = train_config['truncate']
        self.batch_size = train_config['batch_size']

        self.features = train_config['features']
        self.patch_size = pre_config['patch_size']
        self.par_cnt = pre_config['par_cnt']
        self.patch_size_ref = pre_config['patch_size_ref']
        self.par_cnt_ref = pre_config['par_cnt_ref']
        self.surface = pre_config['surf']
        self.pad_val = pre_config['pad_val']

        self.path_src = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
        self.path_ref = "%sreference/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"

        self.avg_patch_cnt = self.batch_size / chunk_size
        self.chunk_size = chunk_size
        self.data_iterations = data_iterations

        self.next_idx = 0

        self.on_epoch_end()


    def __len__(self):
        return int(np.floor((self.d_end-self.d_start)*(self.t_end-self.t_start)*(self.v_end-self.v_start)*self.data_iterations/self.chunk_size))


    def __getitem__(self, index):
        return self._data_generation(index)
        '''if index >= self.next_idx:
            self._data_generation()

        i = (index - self.next_idx) * self.batch_size

        X = self.src_patches[i:i+self.batch_size]
        Y = self.ref_patches[i:i+self.batch_size] 

        return X, Y'''


    def on_epoch_end(self):
        self.idx = np.array([[x,y,z] for z in range(self.t_start, self.t_end) for y in range(self.v_start, self.v_end) for x in range(self.d_start, self.d_end)])
        self.idx = np.repeat(self.idx, self.data_iterations, axis=0)
        np.random.shuffle(self.idx)


    def _extract_patches(self, data, positions, shape, par_cnt, patch_size, features=[], aux_data={}):
        idx_grid = ParticleIdxGrid(data, shape)

        patches = [np.empty((len(positions),par_cnt,3))]
        if len(features) > 0:
            patches.append(np.empty((len(positions),par_cnt,len(features) + 2 if 'v' in features else 0)))

        for i in range(len(positions)):
            pos = positions[i]
            #print("gen patch: %06d/%06d" % (i+1,len(positions)), end="\r", flush=True)
            idx = idx_grid.get_range(pos, patch_size/2)
            patches[0][i], aux = extract_particles(data[idx], pos, par_cnt, patch_size/2, self.pad_val, aux_data)
            if len(features) > 0:
                patches[1][i] = np.concatenate([(aux[f]) for f in features], axis=-1)
        
        return patches


    def _data_generation(self, index):
        src_patches = [np.empty((0,self.par_cnt,3))]
        if len(self.features) > 0:
            src_patches.append(np.empty((0,self.par_cnt,len(self.features) + 2 if 'v' in self.features else 0)))

        ref_patches = [np.empty((0,self.par_cnt_ref,3))]
        if self.trunc:
            ref_patches.append(np.empty((0,1)))

        ch_idx = self.idx[index*self.chunk_size:(index+1)*self.chunk_size]
        
        positions = np.empty((self.chunk_size,), dtype=object)
        data = np.empty((self.chunk_size,), dtype=object)
        aux_data = np.empty((self.chunk_size,), dtype=object)

        patch_cnt = 0
        shape = None

        #count positions
        for i in range(self.chunk_size):    
            data[i], sdf, aux_data[i] = get_data(self.path_src%(ch_idx[i][0],ch_idx[i][1],ch_idx[i][2]), self.features)

            positions[i] = get_positions(data[i], sdf, self.patch_size, self.surface)
            np.random.shuffle(positions[i])

            patch_cnt += len(positions[i])
            shape = sdf.shape[:3]
        
        avg = patch_cnt / self.chunk_size

        for i in range(self.chunk_size):
            pos = positions[i][:max(1,int(len(positions[i])/avg * self.avg_patch_cnt))]

            p = self._extract_patches(data[i], pos, shape, self.par_cnt, self.patch_size, self.features, aux_data[i])
            
            src_patches[0] = np.concatenate([src_patches[0], p[0]])
            if len(self.features) > 0:
                src_patches[1] = np.concatenate([src_patches[1], p[1]])

            ref_data, sdf = get_data(self.path_ref%(ch_idx[i][0], ch_idx[i][2]), self.features)[:2]

            ref_patches[0] = np.concatenate([ref_patches[0], self._extract_patches(ref_data, pos, sdf.shape[:3], self.par_cnt_ref, self.patch_size_ref)[0]])
    
        if self.trunc:
            ref_patches[1] = np.count_nonzero(ref_patches[0][:,:,:1] != self.pad_val, axis=1)/self.par_cnt_ref

        p_idx = np.arange(len(src_patches[0]))
        np.random.shuffle(p_idx)
        p_idx = p_idx[:self.batch_size]

        for i in range(len(src_patches)):
            src_patches[i] = src_patches[i][p_idx]
        
        for i in range(len(ref_patches)):
            ref_patches[i] = ref_patches[i][p_idx]

        return src_patches, ref_patches
        #self.next_idx += len(self.src_patches)


    def generator(self):
        index_in_chunk = 0

        while True:
            if index_in_chunk >= self.chunk_size:
                self._data_generation()

            X = self.src_patches[index_in_chunk:(index_in_chunk + self.batch_size)]
            Y = self.ref_patches[index_in_chunk:(index_in_chunk + self.batch_size)] 
            index_in_chunk += self.batch_size

            yield X, Y