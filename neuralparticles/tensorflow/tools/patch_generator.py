import json
import os
import keras

from neuralparticles.tools.data_helpers import extract_particles, get_positions, get_data
from neuralparticles.tools.particle_grid import ParticleIdxGrid

from neuralparticles.tools.uniio import readNumpyRaw

import numpy as np

import random, copy

class Frame:
    dataset = None
    timestep = None
    patch_idx = None

    def __init__(self, dataset, timestep, patch_idx):
        self.dataset = dataset
        self.timestep = timestep
        self.patch_idx = patch_idx


class Chunk:
    frames = None
    patch_idx = None
    size = None

    def __init__(self, size=None, frames=None):
        if not size is None:
            self.frames = np.empty((size, 2))
            self.patch_idx = np.empty((size,), dtype=object)
            self.size = size
        elif not frames is None:
            self.frames = frames[:]
            self.patch_idx = np.empty((len(frames),), dtype=object)
            self.size = len(frames)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return Frame(self.frames[i][0], self.frames[i][1], self.patch_idx[i])


class PatchGenerator(keras.utils.Sequence):
    def __init__(self, data_path, config_path, chunk_size,
                 d_start=-1, d_end=-1, t_start=-1, t_end=-1, chunked_idx=None, fac=1.0):
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
        self.d_end = int(data_config['data_count']) if d_end < 0 else d_end
        self.t_start = min(train_config['t_start'], data_config['frame_count']-1) if t_start < 0 else t_start
        self.t_end = min(train_config['t_end'], data_config['frame_count']) if t_end < 0 else t_end

        self.neg_examples = train_config['neg_examples']

        self.fps = data_config['fps']

        self.batch_cnt = 0

        self.trunc = train_config['truncate']
        self.batch_size = train_config['batch_size']

        self.features = train_config['features']

        self.par_cnt = pre_config['par_cnt']
        self.par_cnt_ref = pre_config['par_cnt_ref']
        self.pad_val = pre_config['pad_val']

        self.chunk_size = chunk_size
        self.chunk_cnt = int(np.ceil((self.d_end - self.d_start) * (self.t_end - self.t_start)/self.chunk_size))
        self.chunked_idx = np.empty((self.chunk_cnt,), dtype=object)
        self.chunked_idx_val = np.empty((self.chunk_cnt,), dtype=object)

        self.src_path = "%spatches/source/%s_%s-%s_p" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"
        self.ref_path = "%spatches/reference/%s_%s-%s_ps" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_%03d"

        if chunked_idx is None:
            idx = np.array([[x,y] for x in range(self.d_start, self.d_end) for y in range(self.t_start, self.t_end)])
            np.random.shuffle(idx)
            idx = idx[:int(len(idx))]

            for i in range(self.chunk_cnt):
                chunk = Chunk(frames=idx[i*self.chunk_size:(i+1)*self.chunk_size])
                chunk_val = Chunk(frames=chunk.frames)

                patch_cnt = 0
                for j in range(chunk.size):                
                    patch_idx = np.arange(len(readNumpyRaw(self.src_path % ('s', chunk.frames[j][0], chunk.frames[j][1]))))

                    if fac < 1.0:
                        patch_idx = patch_idx[random.sample(range(len(patch_idx)), int(fac * len(patch_idx)))]

                    val_choice = random.sample(range(len(patch_idx)), int(np.ceil(train_config['val_split'] * len(patch_idx))))
                    chunk_val.patch_idx[j] = patch_idx[val_choice]
                    chunk.patch_idx[j] = np.delete(patch_idx, val_choice)      

                    patch_cnt += len(chunk.patch_idx[j])
                self.batch_cnt += int(np.ceil(patch_cnt/self.batch_size))
                self.chunked_idx[i] = chunk
                self.chunked_idx_val[i] = chunk_val
        else:
            self.chunked_idx = chunked_idx
            self.chunked_idx_val = None
            for c in self.chunked_idx:
                patch_cnt = 0
                for f in c.patch_idx:
                    patch_cnt += len(f)
                self.batch_cnt += int(np.ceil(patch_cnt/self.batch_size))
        
        self.on_epoch_end()


    def get_val_idx(self):
        return copy.deepcopy(self.chunked_idx_val)


    def __len__(self):
        return self.batch_cnt


    def _load_chunk(self):
        frame_chunk = self.chunked_idx[self.chunk_idx]
        patch_cnt = 0
        for pi in frame_chunk.patch_idx:
            patch_cnt += len(pi)
        self.chunk = np.empty((patch_cnt, 2), dtype=object)

        pi = 0
        for i in range(frame_chunk.size):
            frame = frame_chunk[i]
            pin = pi + len(frame.patch_idx)

            src = readNumpyRaw(self.src_path%('s',frame.dataset,frame.timestep))[frame.patch_idx]
            if len(self.features) > 0:
                src = np.concatenate([src] + [readNumpyRaw(self.src_path%(f,frame.dataset,frame.timestep))[frame.patch_idx] for f in self.features], axis=-1)

            ref = [readNumpyRaw(self.ref_path%(frame.dataset,frame.timestep))[frame.patch_idx]]
            if self.trunc:
                ref.append(np.count_nonzero(ref[0][...,1] != self.pad_val, axis=1)/self.par_cnt_ref)
            
            for j in range(pin-pi):
                self.chunk[pi+j] = [[src[j]], [r[j] for r in ref]]

            pi = pin

        np.random.shuffle(self.chunk)
        self.chunk_idx += 1


    def __getitem__(self, index):
        if(index * self.batch_size < self.idx_offset):
            print("error! %d - %d < 0" % (index * self.batch_size, self.idx_offset))
            self.on_epoch_end()

        index = index * self.batch_size - self.idx_offset
        if index >= len(self.chunk):
            self.idx_offset += index
            index = 0
            self._load_chunk()

        src = [np.array([s[i] for s in self.chunk[index:index+self.batch_size,0]]) for i in range(len(self.chunk[0,0]))]
        ref = [np.array([r[i] for r in self.chunk[index:index+self.batch_size,1]]) for i in range(len(self.chunk[0,1]))]

        if index % 2 == 0 or not self.neg_examples:
            adv_src = src[0][...,:3] + 0.01 * src[0][...,3:6] / self.fps
            src.append(np.concatenate((adv_src, src[0][...,3:]), axis=-1))
            ref.insert(1, np.concatenate((ref[0], ref[0]), axis=1))
        else:
            rnd_idx = np.random.randint(0, len(self.chunk), self.batch_size)
            src.extend([np.array([s[i] for s in self.chunk[rnd_idx,0]]) for i in range(len(self.chunk[0,0]))])
            ref.insert(1, np.concatenate((ref[0], [np.array([r[i] for r in self.chunk[rnd_idx,1]]) for i in range(len(self.chunk[0,1]))][0]), axis=1))
        return src, ref


    def on_epoch_end(self):
        self.chunk_idx = 0
        self.idx_offset = 0
        self._load_chunk()