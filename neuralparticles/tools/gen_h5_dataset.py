import json
from neuralparticles.tools.param_helpers import getParam, checkUnusedParams
from neuralparticles.tools.uniio import writeNumpyH5, readNumpyRaw
from glob import glob

import h5py

import numpy as np

import os

def npy_to_h5(src_path, ref_path, output_path):
    data = None
    for f in glob(src_path):
        print(f)
        tmp = np.array(readNumpyRaw(f[:-4]))
        data = tmp if data is None else np.append(data, tmp, axis=0)
    writeNumpyH5(output_path, data, 'source')

    data = None
    for f in glob(ref_path):
        print(f)
        tmp = np.array(readNumpyRaw(f[:-4]))
        data = tmp if data is None else np.append(data, tmp, axis=0)
    writeNumpyH5(output_path, data, 'reference')

if __name__ == "__main__":
    src_path = getParam("src", "data/source/*")
    ref_path = getParam("ref", "data/reference/*")
    output_path = getParam("dst", "data/tmp/test")
    checkUnusedParams()

    npy_to_h5(src_path, ref_path, output_path)