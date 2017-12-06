import numpy as np
import sys
sys.path.append("manta/scenes/tools/")
from helpers import *

np.random.seed(3)
par = np.array([[4,2,5],[2,6,1],[2,1,7],[2,1,1]])
pos = np.array([2,2,2])
print(extract_particles(par,pos,3,3))
