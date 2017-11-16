from uniio import *
import numpy as np

class Dataset:
    def __init__(self, prefix, start, end, t_start, t_end, src_features, ref_prefix="", ref_features=[]):
        self.data = None
        self.ref_data = None
        def read_dataset(path,lim):
            tmp = None
            for d in range(start, end):
                for t in range(t_start, t_end):
                    buf = NPZBuffer(path%(d,t))
                    while True:
                        v = buf.next()
                        if v is None:
                            break
                        if lim > 0:
                            v = v[:,:2]
                            np.random.shuffle(v)
                            v = np.resize(v,lim)
                        if tmp is None:
                            tmp = [v]
                        else:
                            tmp = np.append(tmp, [v], axis=0)
            return tmp
        
        for f in src_features:
            lim = 0
            if f == "ps":
                lim = 100
            if self.data is None:
                self.data = read_dataset(prefix+'_'+f,lim)
            else:
                self.data = np.append(self.data, read_dataset(prefix+'_'+f,lim), axis=3)

        if ref_prefix != "":
            for f in ref_features:
                lim = 0
                if f == "ps":
                    lim = 1000
                if self.ref_data is None:
                    self.ref_data = read_dataset(ref_prefix+'_'+f,lim)
                else:
                    self.ref_data = np.append(self.ref_data, read_dataset(ref_prefix+'_'+f,lim), axis=3)


    def get_batch(self, size):
        rnd_idx = np.random.randint(0, len(self.data),size)
        return (self.data[rnd_idx],
                self.ref_data[rnd_idx])
