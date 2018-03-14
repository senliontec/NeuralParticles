from uniio import *
import numpy as np

class Dataset:
    def __init__(self, prefix, start, end, t_start, t_end, src_features, var_cnt=1, par_var_cnt=1, ref_prefix="", ref_features=[]):
        self.src_features=src_features
        self.ref_features=ref_features
        self.data = {}
        self.ref_data = {}
        def read_dataset(path):
            tmp = None
            for d in range(start, end):
                for var in range(var_cnt):
                    for t in range(t_start, t_end):
                        for r in range(par_var_cnt):
                            buf = NPZBuffer(path%(d,var,r,t))
                            while True:
                                v = buf.next()
                                if v is None:
                                    break
                                tmp = [v] if tmp is None else np.append(tmp, [v], axis=0)
            return tmp
        
        for f in src_features:
            self.data[f] = read_dataset(prefix+'_'+f)

        if ref_prefix != "":
            for f in ref_features:
                self.ref_data[f] = read_dataset(ref_prefix+'_'+f)

    def get_data_splitted(self, idx=None):
        def split(data, features, idx=None):
            if idx is None:
                x = [np.array(data[features[0]])]
                if len(features) > 1:
                    x = [x[0], np.array(np.concatenate([data[f] for f in features[1:]],axis=-1))]
            else:
                x = [np.array(data[features[0]][idx])]
                if len(features) > 1:
                    x = [x[0], np.array(np.concatenate([data[f][idx] for f in features[1:]],axis=-1))]
            return x

        if len(self.ref_features) > 0:
            return split(self.data, self.src_features, idx), split(self.ref_data, self.ref_features, idx)
        else:
            return split(self.data, self.src_features, idx)
        
            