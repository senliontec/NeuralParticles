import json
import os
import math
import keras

from neuralparticles.tools.data_helpers import extract_particles, get_positions, get_data, get_nearest_idx
from neuralparticles.tools.particle_grid import ParticleIdxGrid

from neuralparticles.tools.uniio import readNumpyRaw

from neuralparticles.tools.data_augmentation import *

import numpy as np

import random, copy

from enum import Flag, auto


class Augmentation(Flag):
    ROTATE = auto()
    SHIFT = auto()
    JITTER = auto()
    JITTER_ROT = auto()
    RND_SAMPLING = auto()


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
                 d_start=-1, d_end=-1, t_start=-1, t_end=-1, chunked_idx=None):
        np.random.seed(45)
        random.seed(45)
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
        
        fac_d = math.pow(pre_config['factor'], 1/data_config['dim'])
        self.patch_size = pre_config['patch_size'] * data_config['res'] / fac_d

        self.batch_cnt = 0

        tmp_w = train_config['loss_weights']
        self.temp_coh = tmp_w[1] > 0.0
        self.trunc = tmp_w[2] > 0.0
        self.batch_size = train_config['batch_size']

        self.features = train_config['features']

        self.par_cnt = pre_config['par_cnt']
        self.par_cnt_ref = pre_config['par_cnt_ref']
        self.pad_val = pre_config['pad_val']

        self.chunk_size = chunk_size
        self.chunk_cnt = int(np.ceil((self.d_end - self.d_start) * (self.t_end - self.t_start)/self.chunk_size))
        self.chunked_idx = np.empty((self.chunk_cnt,), dtype=object)
        self.chunked_idx_val = np.empty((self.chunk_cnt,), dtype=object)

        self.fac = train_config['sub_fac']
        self.gen_vel = train_config['gen_vel']

        self.data_aug = Augmentation(0)
        self.rot_mask = np.array(train_config['rot_mask'])

        self.src_path = "%spatches/source/%s_%s-%s_p" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"
        self.ref_path = "%spatches/reference/%s_%s-%s_p" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "%s_d%03d_%03d"

        if chunked_idx is None:
            if train_config['rot_aug']:
                self.data_aug |= Augmentation.ROTATE
            if train_config['shift_aug']:
                self.data_aug |= Augmentation.SHIFT
            if train_config['jitter_aug']:
                self.data_aug |= Augmentation.JITTER
            if train_config['jitter_rot_aug']:
                self.data_aug |= Augmentation.JITTER_ROT
            if train_config['rnd_sampling']:
                self.data_aug |= Augmentation.RND_SAMPLING
            
            idx = np.array([[x,y] for x in range(self.d_start, self.d_end) for y in range(self.t_start, self.t_end)])
            np.random.shuffle(idx)

            for i in range(self.chunk_cnt):
                chunk = Chunk(frames=idx[i*self.chunk_size:(i+1)*self.chunk_size])
                chunk_val = Chunk(frames=chunk.frames)

                patch_cnt = 0
                for j in range(chunk.size):                
                    patch_idx = np.arange(len(readNumpyRaw(self.src_path % ('s', chunk.frames[j][0], chunk.frames[j][1]))))

                    #if fac < 1.0:
                    #    patch_idx = patch_idx[random.sample(range(len(patch_idx)), int(np.ceil(fac * len(patch_idx))))]

                    val_choice = random.sample(range(len(patch_idx)), int(np.ceil(train_config['val_split'] * len(patch_idx))))
                    chunk_val.patch_idx[j] = patch_idx[val_choice]
                    chunk.patch_idx[j] = np.delete(patch_idx, val_choice)      

                    patch_cnt += int(np.ceil(self.fac * len(chunk.patch_idx[j])))
                self.batch_cnt += int(np.ceil(patch_cnt/self.batch_size))
                self.chunked_idx[i] = chunk
                self.chunked_idx_val[i] = chunk_val
        else:
            self.chunked_idx = chunked_idx
            self.chunked_idx_val = None
            for c in self.chunked_idx:
                patch_cnt = 0
                for f in c.patch_idx:
                    patch_cnt += int(np.ceil(self.fac * len(f)))
                self.batch_cnt += int(np.ceil(patch_cnt/self.batch_size))
        print(self.data_aug)
        self.on_epoch_end()


    def get_val_idx(self):
        return copy.deepcopy(self.chunked_idx_val)


    def __len__(self):
        return self.batch_cnt


    def _load_chunk(self):
        frame_chunk = self.chunked_idx[self.chunk_idx]
        patch_cnt = 0
        for pi in frame_chunk.patch_idx:
            patch_cnt += int(np.ceil(self.fac * len(pi)))
        self.chunk = np.empty((patch_cnt, 2), dtype=object)

        pi = 0
        for i in range(frame_chunk.size):
            frame = frame_chunk[i]
            idx = frame.patch_idx
            if self.fac < 1.0:
                idx = idx[random.sample(range(len(idx)), int(np.ceil(self.fac * len(idx))))]
            pin = pi + len(idx)

            ref = [readNumpyRaw(self.ref_path%('s',frame.dataset,frame.timestep))[idx]]
            if self.trunc:
                ref.append(np.count_nonzero(ref[0][...,1] != self.pad_val, axis=1)/self.par_cnt_ref)

            if self.data_aug & Augmentation.RND_SAMPLING:
                tmp_ref = ref[0]
                src = np.empty((len(tmp_ref), self.par_cnt, 3 + len(self.features) + (2 if 'v' in self.features or 'n' in self.features else 0)))
                if len(self.features) > 0:
                    tmp_ref = np.concatenate([tmp_ref] + [readNumpyRaw(self.ref_path%(f,frame.dataset,frame.timestep))[idx] for f in self.features], axis=-1)
                for j in range(len(tmp_ref)):
                    sample_idx = nonuniform_sampling(self.par_cnt_ref, self.par_cnt)
                    src[j] = tmp_ref[j][sample_idx]
            else:
                src = readNumpyRaw(self.src_path%('s',frame.dataset,frame.timestep))[idx]
                if len(self.features) > 0:
                    src = np.concatenate([src] + [readNumpyRaw(self.src_path%(f,frame.dataset,frame.timestep))[idx] for f in self.features], axis=-1)
                
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

        if self.data_aug & Augmentation.ROTATE:
            src[0], ref[0] = rotate_point_cloud_and_gt(src[0], ref[0], self.rot_mask)
        if self.data_aug & Augmentation.SHIFT:
            src[0], ref[0] = shift_point_cloud_and_gt(src[0], ref[0], shift_range=0.1)
        if (self.data_aug & Augmentation.JITTER) and np.random.rand() > 0.5:
            src[0] = jitter_perturbation_point_cloud(src[0], sigma=0.025,clip=0.05)
        if (self.data_aug & Augmentation.JITTER_ROT) and np.random.rand() > 0.5:
            src[0] = rotate_perturbation_point_cloud(src[0], angle_sigma=0.03, angle_clip=0.09)

        if self.temp_coh:
            if self.gen_vel:
                vel = random_deformation(src[0])
                src[0] = np.concatenate((src[0], vel), axis=-1)
            if index % 2 == 0 or not self.neg_examples:
                if not self.gen_vel:
                    vel = src[0][...,3:6]
                idx = np.argmin(np.linalg.norm(src[0][...,:3], axis=-1), axis=1)
                vel = vel - np.expand_dims(vel[np.arange(len(idx)), idx],axis=1)
                
                adv_src = src[0][...,:3] + vel * 0.1 / self.fps
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