import json
from neuralparticles.tools.param_helpers import getParam, checkUnusedParams
from neuralparticles.tools.uniio import writeNumpyXYZ, readParticlesUni
from glob import glob

import numpy as np

import os

def uni_to_xyz(data_path, output_path, trunc_data=False):
    if trunc_data:
        in_data = [readParticlesUni(f,"float32")[1] for f in glob(data_path)]
        min_size = min([v.shape[0] for v in in_data])
        print("truncate to size: ", min_size)
        i = 0
        for f in glob(data_path):
            v = in_data[i].copy()
            np.random.shuffle(v)
            writeNumpyXYZ(output_path + os.path.splitext(os.path.basename(f))[0], v[:min_size])
            i += 1
    else:
        for f in glob(data_path):
            writeNumpyXYZ(output_path + os.path.splitext(os.path.basename(f))[0],readParticles(f,"float32")[1])

if __name__ == "__main__":
    data_path = getParam("src", "data/source/*")
    output_path = getParam("dst", "data/tmp/")
    trunc = int(getParam("trunc", 0)) != 0
    checkUnusedParams()

    uni_to_xyz(data_path, output_path, trunc)