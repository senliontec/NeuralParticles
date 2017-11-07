from uniio import *
import numpy as np

class Dataset:
    def __init__(self, prefix, ref_prefix, offset, count, src_features, ref_features):
        self.data = None
        self.ref_data = None
        def read_dataset(path):
            tmp = None
            for d in range(offset, count):
                buf = NPZBuffer(path%d)
                while True:
                    v = buf.next()
                    if v is None:
                        break
                    if tmp is None:
                        tmp = [v]
                    else:
                        tmp = np.append(tmp, [v], axis=0)
            return tmp
        
        for f in src_features:
            if self.data is None:
                self.data = read_dataset(prefix+'_'+f)
            else:
                self.data = np.append(self.data, read_dataset(prefix+'_'+f), axis=3)

        for f in ref_features:
            if self.ref_data is None:
                self.ref_data = read_dataset(ref_prefix+'_'+f)
            else:
                self.ref_data = np.append(self.ref_data, read_dataset(ref_prefix+'_'+f), axis=3)


    def get_batch(self, size):
        rnd_idx = np.random.randint(0, len(self.data),size)
        return (self.data[rnd_idx],
                self.ref_data[rnd_idx])
