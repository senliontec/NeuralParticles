from uniio import *
import numpy as np

class Dataset:
    def __init__(self, prefix, ref_prefix, offset, count):
        self.data = None
        self.ref_data = None

        for d in range(offset, count):
            src_buf = NPZBuffer(prefix%d)
            ref_buf = NPZBuffer(ref_prefix%d)
            while True:
                s_v = src_buf.next()
                r_v = ref_buf.next()
                if s_v is None:
                    break
                if self.data is None:
                    self.data = [s_v]
                    self.ref_data = [r_v]
                else:
                    self.data = np.append(self.data, [s_v], axis=0)
                    self.ref_data = np.append(self.ref_data, [r_v], axis=0)


    def get_batch(self, size):
        rnd_idx = np.random.randint(0, len(self.data),size)
        return (self.data[rnd_idx],
                self.ref_data[rnd_idx])
