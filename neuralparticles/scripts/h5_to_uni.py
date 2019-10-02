import time
import numpy as np
from collections import OrderedDict

from neuralparticles.tools.uniio import writeParticlesUni, readNumpyH5
from neuralparticles.tools.param_helpers import getParam, checkUnusedParams

in_path = getParam("in", "data/%04d.h5")
out_path = getParam("out", "data/%04d.uni")

bnd = int(getParam("bnd", 1))

key = getParam("key", "positions")

data_cnt = int(getParam("cnt", 10))
res = int(getParam("res", 150))

min_v = np.asarray(getParam("min", "0,0,0").split(","), dtype="float32")
max_v = np.asarray(getParam("max", "1,1,1").split(","), dtype="float32")

checkUnusedParams()


hdr = OrderedDict([ ('dim',0),
                    ('dimX',res),
                    ('dimY',res),
                    ('dimZ',res),
                    ('elementType',0),
                    ('bytesPerElement',16),
                    ('info',b'\0'*256),
                    ('timestamp',(int)(time.time()*1e6))])

for d in range(data_cnt):
    print("Dataset: %d/%d" % (d+1, data_cnt))
    data = readNumpyH5(in_path%d, key)
    data = (data - min_v) / (max_v - min_v)
    data = data * (res - 2*bnd) + bnd 
    print(np.min(data, axis=0))
    print(np.max(data, axis=0))
    print(data.shape)
    hdr['dim'] = len(data)
    writeParticlesUni(out_path%d, hdr, data)
