import json
import os
import math
import keras

from neuralparticles.tools.data_helpers import PatchExtractor, get_data, get_data_pair, get_nearest_idx
from neuralparticles.tools.data_augmentation import *
from neuralparticles.tools.plot_helpers import plot_particles

import numpy as np

import random, copy


class Frame:
    dataset = None
    timestep = None
    position_idx = None

    def __init__(self, dataset, timestep, position_idx):
        self.dataset = dataset
        self.timestep = timestep
        self.position_idx = position_idx


class Chunk:
    frames = None
    position_idx = None
    size = None

    def __init__(self, size=None, frames=None):
        if not size is None:
            self.frames = np.empty((size, 2))
            self.position_idx = np.empty((size,), dtype=object)
            self.size = size
        elif not frames is None:
            self.frames = frames[:]
            self.position_idx = np.empty((len(frames),), dtype=object)
            self.size = len(frames)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return Frame(self.frames[i][0], self.frames[i][1], self.position_idx[i])


def extract_series(data_path, config_path, d, t, v, patch_idx=None, cnt=3, shuffle=True, t_int=1, real=False):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
        data_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
        pre_config = json.loads(f.read())

    with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
        train_config = json.loads(f.read())

    factor_d = math.pow(pre_config['factor'], 1/data_config['dim'])

    patch_size = pre_config['patch_size'] * data_config['res'] / factor_d
    patch_size_ref = pre_config['patch_size_ref'] * data_config['res']

    real_path = "%sreal/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"
    if real:
        src_data, src_sdf_data, src_aux = get_data(real_path % (d, t), par_aux=train_config['features'])
    else:
        (src_data, src_sdf_data, src_aux), (ref_data, ref_sdf_data, ref_aux) = get_data_pair(data_path, config_path, d, t, v, ref_features=['v'])     
    
    idx = range(src_data.shape[0]) if patch_idx is None else patch_idx
    pos = src_data[idx]    

    patch_ex_src = []
    if not real: patch_ex_ref = []
    for i in range(cnt):
        if i > 0:
            if not train_config['adv_src']:
                if real:
                    src_data, src_sdf_data, src_aux = get_data(real_path % (d, t+i*t_int), par_aux=train_config['features'])
                else:
                    (src_data, src_sdf_data, src_aux), (ref_data, ref_sdf_data, ref_aux) = get_data_pair(data_path, config_path, d, t+i*t_int, v, ref_features=['v'])
            else:
                src_data = src_data + t_int * src_aux['v'] / data_config['fps']
                if not real: ref_data = ref_data + t_int * ref_aux['v'] / data_config['fps']

        patch_ex_src.append(PatchExtractor(src_data, src_sdf_data, patch_size, pre_config['par_cnt'], pad_val=pre_config['pad_val'], last_pos=pos, aux_data=src_aux, features=train_config['features'], shuffle=shuffle, temp_coh=True))
        if not real: patch_ex_ref.append(PatchExtractor(ref_data, ref_sdf_data, patch_size_ref, pre_config['par_cnt_ref'], pad_val=pre_config['pad_val'], positions=patch_ex_src[i].positions*factor_d, aux_data=ref_aux, features=['v'], shuffle=shuffle))

        pos = patch_ex_src[i].positions + t_int * src_aux['v'][patch_ex_src[i].pos_idx] / data_config['fps']
    if real:
        return patch_ex_src
    return patch_ex_src, patch_ex_ref

class PatchGenerator(keras.utils.Sequence):
    def __init__(self, data_path, config_path, chunk_size,
                 d_start=-1, d_end=-1, t_start=-1, t_end=-1, chunked_idx=None, trunc=False, eval=False):
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

        self.data_path = data_path
        self.config_path = config_path

        self.d_start = (data_config['data_count'] if eval else 0) if d_start < 0 else d_start
        self.d_end = (data_config['data_count'] + (data_config['test_count'] if eval else 0)) if d_end < 0 else d_end
        self.t_start = min(train_config['t_start'], data_config['frame_count']-1) if t_start < 0 else t_start
        self.t_end = min(train_config['t_end'], data_config['frame_count']) if t_end < 0 else t_end
        self.t_int = train_config['t_int']

        self.neg_examples = train_config['neg_examples']

        self.fps = data_config['fps'] / self.t_int
        
        self.fac_d = math.pow(pre_config['factor'], 1/data_config['dim'])
        self.fac_d = np.array([self.fac_d, self.fac_d, 1 if data_config['dim'] == 2 else self.fac_d])
        self.patch_size = pre_config['patch_size'] * data_config['res'] / self.fac_d[0]
        self.patch_size_ref = pre_config['patch_size_ref'] * data_config['res']
        self.par_cnt = pre_config['par_cnt']
        self.par_cnt_ref = pre_config['par_cnt_ref']
        self.bnd = data_config['bnd']/self.fac_d[0]

        self.pad_val = pre_config['pad_val']
        self.surface = pre_config['surf']
        self.stride = pre_config['stride']

        self.use_adv_src = train_config['adv_src']

        self.jitter = train_config['jitter_aug']

        self.batch_cnt = 0

        tmp_w = train_config['loss_weights']
        self.temp_coh = tmp_w[1] > 0.0 or eval
        self.trunc = tmp_w[2] > 0.0 and not train_config['pretrain']
        self.trunc_only = trunc
        self.fac = train_config['sub_fac']
        self.gen_vel = train_config['gen_vel']
        self.batch_size = train_config['batch_size']

        self.features = train_config['features']

        if self.gen_vel:
            print("GEN VEL NOT IMPLEMENTED YET!")
            exit()

        if self.temp_coh:
            self.t_end -= 2 * self.t_int
            if not self.use_adv_src:
                self.t_end -= 2 * self.t_int

        self.chunk_size = chunk_size
        self.chunk_cnt = int(np.ceil((self.d_end - self.d_start) * (self.t_end - self.t_start)/self.chunk_size))
        self.chunked_idx = np.empty((self.chunk_cnt,), dtype=object)
        self.chunked_idx_val = np.empty((self.chunk_cnt,), dtype=object)

        val_split = 0 if eval else train_config['val_split']

        path_src = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"

        if chunked_idx is None:
            idx = np.array([[x,y] for x in range(self.d_start, self.d_end) for y in range(self.t_start, self.t_end)])
            np.random.shuffle(idx)

            for i in range(self.chunk_cnt):
                chunk = Chunk(frames=idx[i*self.chunk_size:(i+1)*self.chunk_size])
                chunk_val = Chunk(frames=chunk.frames)

                patch_cnt = 0
                for j in range(chunk.size):       
                    src_data, sdf_data, _ = get_data(path_src % (chunk.frames[j][0], 0, chunk.frames[j][1]))
                    position_idx = PatchExtractor(src_data, sdf_data, self.patch_size, self.par_cnt, self.surface, self.stride, self.bnd, self.pad_val).pos_idx

                    val_choice = random.sample(range(len(position_idx)), int(np.ceil(val_split * len(position_idx))))
                    chunk_val.position_idx[j] = position_idx[val_choice]
                    chunk.position_idx[j] = np.delete(position_idx, val_choice)      

                    patch_cnt += int(np.ceil(self.fac * len(chunk.position_idx[j])))
                self.batch_cnt += patch_cnt
                self.chunked_idx[i] = chunk
                self.chunked_idx_val[i] = chunk_val
            print("%d data pairs used for training" % self.batch_cnt)
        else:
            self.chunked_idx = chunked_idx
            self.chunked_idx_val = None
            for c in self.chunked_idx:
                patch_cnt = 0
                for f in c.position_idx:
                    patch_cnt += int(np.ceil(self.fac * len(f)))
                self.batch_cnt += patch_cnt
            print("%d data pairs used for validation" % self.batch_cnt)
        self.batch_cnt = int(np.ceil(self.batch_cnt/self.batch_size))
        self.on_epoch_end()


    def get_val_idx(self):
        return copy.deepcopy(self.chunked_idx_val)


    def __len__(self):
        return self.batch_cnt


    def _load_chunk(self):
        frame_chunk = self.chunked_idx[self.chunk_idx]
        self.chunk = np.empty((frame_chunk.size, 8 if (self.temp_coh and not self.use_adv_src) else 2), dtype=object)        
        for i in range(frame_chunk.size):
            frame = frame_chunk[i]
            idx = frame.position_idx
            if self.fac < 1.0:
                idx = idx[random.sample(range(len(idx)), int(np.ceil(self.fac * len(idx))))]

            if self.temp_coh and not self.use_adv_src:
                patch_ex_src, patch_ex_ref = extract_series(self.data_path, self.config_path, frame.dataset, frame.timestep, 0, idx, 5, t_int=self.t_int)
        
                self.chunk[i] = [patch_ex_src[2], patch_ex_ref[2], patch_ex_src[1], patch_ex_src[3], patch_ex_src[0], patch_ex_src[4], patch_ex_ref[1], patch_ex_ref[3]]
            else:
                (src_data, sdf_data, par_aux), (ref_data, ref_sdf_data, ref_aux) = get_data_pair(self.data_path, self.config_path, frame.dataset, frame.timestep, 0, features=self.features, ref_features=['v'])

                patch_ex_src = PatchExtractor(src_data, sdf_data, self.patch_size, self.par_cnt, pad_val=self.pad_val, positions=src_data[idx], aux_data=par_aux, features=self.features)
                patch_ex_ref = PatchExtractor(ref_data, ref_sdf_data, self.patch_size_ref, self.par_cnt_ref, pad_val=self.pad_val, positions=patch_ex_src.positions*self.fac_d, aux_data=ref_aux, features=['v'])
                
                self.chunk[i] = [patch_ex_src, patch_ex_ref] 

        np.random.shuffle(self.chunk)
        self.chunk_idx += 1


    def __getitem__(self, index):
        src = [np.empty((self.batch_size, self.par_cnt, 3 + len(self.features) + (2 if 'v' in self.features or 'n' in self.features else 0)))]

        if self.trunc_only:
            ref = [np.empty((self.batch_size,))]        
        else:
            ref = [np.empty((self.batch_size, self.par_cnt_ref, 3))]
            if self.temp_coh:
                src.append(src[0].copy())
                src.append(src[0].copy())
                if not self.use_adv_src:
                    src.append(src[0].copy())
                    src.append(src[0].copy())
                ref.append(np.empty((self.batch_size, self.par_cnt_ref*3, 3)))

        for i in range(self.batch_size):
            if len(self.chunk) <= 0:
                self.cnt = 0
                if self.chunk_idx == len(self.chunked_idx):
                    src = [s[:i] for s in src]
                    ref = [r[:i] for r in ref]
                    break
                self._load_chunk()

            c_idx = np.random.randint(len(self.chunk))
            src[0][i] = self.chunk[c_idx][0].pop_patch(remove_data=False)

            ref_patch = self.chunk[c_idx][1].pop_patch(remove_data=False)
            if self.trunc_only:
                ref[0][i] = np.count_nonzero(ref_patch[...,1] != self.pad_val, axis=0)/self.par_cnt_ref
            else:
                ref[0][i] = ref_patch[...,:3]

                if self.temp_coh:
                    if np.random.random() < self.neg_examples:
                        rnd_c_idx = np.random.randint(len(self.chunk))
                        rnd_p_idx = np.random.randint(len(self.chunk[rnd_c_idx][0].positions))
                        src[1][i] = self.chunk[rnd_c_idx][0].get_patch(rnd_p_idx, remove_data=False)
                        ref1 = self.chunk[rnd_c_idx][1].get_patch(rnd_p_idx, remove_data=False)

                        rnd_c_idx = np.random.randint(len(self.chunk))
                        rnd_p_idx = np.random.randint(len(self.chunk[rnd_c_idx][0].positions))
                        src[2][i] = self.chunk[rnd_c_idx][0].get_patch(rnd_p_idx, remove_data=False)
                        ref2 = self.chunk[rnd_c_idx][1].get_patch(rnd_p_idx, remove_data=False)

                        ref[1][i] = np.concatenate((ref_patch[...,:3], ref1[...,:3], ref2[...,:3]))
                    else:
                        if self.use_adv_src:                        
                            vel = src[0][i][...,3:6]
                            mask = vel != self.pad_val
                            idx = np.argmin(np.linalg.norm(src[0][i][...,:3], axis=-1), axis=0)
                            vel = vel - np.expand_dims(vel[idx],axis=0)
                            vel = vel * mask
                            
                            adv_src = src[0][i][...,:3] - 2 * vel / (self.fps * self.patch_size)
                            src[1][i] = np.concatenate((adv_src,src[0][i][...,3:]), axis=-1)

                            adv_src = src[0][i][...,:3] + 2 * vel / (self.fps * self.patch_size)
                            src[2][i] = np.concatenate((adv_src,src[0][i][...,3:]), axis=-1)

                            vel = ref_patch[...,3:]
                            mask = vel != self.pad_val
                            idx = np.argmin(np.linalg.norm(ref_patch[...,:3], axis=-1), axis=0)
                            vel = vel - np.expand_dims(vel[idx],axis=0)
                            vel = vel * mask

                            ref[1][i] = np.concatenate((
                                    ref_patch[...,:3],
                                    ref_patch[...,:3] - 2 * vel / (self.fps * self.patch_size_ref),
                                    ref_patch[...,:3] + 2 * vel / (self.fps * self.patch_size_ref))
                            )
                        else:
                            src[1][i] = self.chunk[c_idx][2].pop_patch(remove_data=False)
                            src[2][i] = self.chunk[c_idx][3].pop_patch(remove_data=False)
                            src[3][i] = self.chunk[c_idx][4].pop_patch(remove_data=False)
                            src[4][i] = self.chunk[c_idx][5].pop_patch(remove_data=False)

                            ref[1][i] = np.concatenate((
                                    ref_patch[...,:3],
                                    self.chunk[c_idx][6].pop_patch(remove_data=False)[...,:3],
                                    self.chunk[c_idx][7].pop_patch(remove_data=False)[...,:3])
                            )

            if self.jitter > 0.0:
                for s in src:
                    s += np.random.normal(scale=self.jitter, size=s.shape) * (s[...,:1] != self.pad_val)

            if self.chunk[c_idx][1].stack_empty():
                self.chunk = np.delete(self.chunk, c_idx, 0)

        if self.trunc and not self.trunc_only: 
            ref.append(np.count_nonzero(ref[0][...,1] != self.pad_val, axis=1)/self.par_cnt_ref)

        return src, ref


    def on_epoch_end(self):
        self.chunk_idx = 0
        self._load_chunk()