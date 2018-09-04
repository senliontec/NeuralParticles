import sys, os
from .param_helpers import getParam, checkUnusedParams
from .plot_helpers import plot_particles, read_csv, write_csv
from neuralparticles.tools.data_helpers import get_nearest_idx

import numpy as np
import matplotlib.pyplot as plt


path = getParam("src", "")
res_path = getParam("res", "")
fac = getParam("fac", 9)
dtIdx = int(getParam("idx", 0))
out = getParam("out", "")

checkUnusedParams()

src = read_csv(path)
res = read_csv(res_path)

plot_particles(src[dtIdx:dtIdx+1], [-1,1], [-1,1], ref=src, path=out%(dtIdx,"in"), s=10)
plot_particles(res[dtIdx:dtIdx+len(src)*fac:len(src)], [-1,1], [-1,1], ref=res, path=out%(dtIdx,"out"), s=10)
