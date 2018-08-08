import json
import os

import numpy as np

class PatchGenerator(object):
    def __init__(self, data_path, config_path, avg_patch_cnt, ):
        with open(config_path, 'r') as f:
            config = json.loads(f.read())

        with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
            data_config = json.loads(f.read())

        with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
            pre_config = json.loads(f.read())

        with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
            train_config = json.loads(f.read())

        self.data_cnt = data_config['data_count'] * train_config['train_split']
        self.t_start = train_config['t_start']
        self.t_end = train_config['t_end']
    
        self.idx = np.random.shuffle(np.transpose(np.indices((self.data_cnt,self.t_end - self.t_start)), (2,1,0)))
        self.idx[:,:,1] += self.t_start

        self.path_src = "%ssource/%s_%s-%s" % (data_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
        self.path_ref = "%sreference/%s_%s" % (data_path, data_config['prefix'], data_config['id']) + "_d%03d_%03d"

    def next_chunk(self):
        


    def generator(self, batch_size):
        index_in_chunk = 0

        while True:
            if index_in_chunk >= self._active_block_length:
                self.next_chunk()
            if inputs == None:
                index_in_chunk = 0
                inputs = self._active_blocks
            if outputs == None:
                outputs = self._active_blocks

            X = [self.__getattribute__(block)[index_in_chunk:(index_in_chunk + used_data_size)] for block in inputs]
            Y = [self.__getattribute__(block)[index_in_chunk:(index_in_chunk + used_data_size)] for block in outputs]
        
            index_in_chunk += batch_size

            yield X, Y