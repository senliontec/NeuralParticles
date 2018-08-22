import sys, os
from .param_helpers import getParam, checkUnusedParams
from .plot_helpers import plot_particles, read_csv, write_csv
from neuralparticles.tools.data_helpers import get_nearest_idx

import numpy as np
import matplotlib.pyplot as plt


path = getParam("src", "")
res_path = getParam("res", "")
dtIdx = int(getParam("idx", 0))
out = getParam("out", "")

checkUnusedParams()


src = read_csv(path)
res = read_csv(res_path)

print(len(res))
for i in range(len(src)):
    j = get_nearest_idx(res[:,:2], src[i, :2])
    print(j)

plot_particles(src[dtIdx:dtIdx+1], [-1,1], [-1,1], ref=src, path=out + "i%04d_src.svg"%dtIdx)
plot_particles(res[dtIdx*9:(dtIdx+1)*9], [-1,1], [-1,1], ref=res, path=out + "i%04d_res.svg"%dtIdx)
