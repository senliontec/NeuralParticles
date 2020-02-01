
import pickle
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.uniio import readParticlesUni
import numpy as np
from neuralparticles.tools.particle_grid import ParticleIdxGrid

import matplotlib.pyplot as plt

ref_path = getParam("ref", "data/reference/reference_%03d.uni")
pred_path = getParam("pred", "data/result/result_%03d.uni")
out_path = getParam("out", "out_%03d.pkl")
err_path = getParam("err", "err.pkl")
res = int(getParam("res", 200))

load = int(getParam("load", 0)) != 0

t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", -1))

checkUnusedParams()

if not load:
    for t in range(t_start, t_end):
        tmp_map = {}
        print("timestep: %d/%d" % (t+1, t_end))
        ref_data = readParticlesUni(ref_path%t, data_type="float32")[1]
        pred_data = readParticlesUni(pred_path%t, data_type="float32")[1]
        idx_grid = ParticleIdxGrid(ref_data, (res, res, res))
        
        for p in pred_data:
            nn_idx = idx_grid.get_nn(p)[0]
            if nn_idx in tmp_map.keys():
                tmp_map[nn_idx].append(p)
            else:
                tmp_map[nn_idx] = [p]

        with open(out_path%t, 'wb') as f:
            pickle.dump(tmp_map, f, protocol=3)

def _compute_stats(x,ref):
    if x.shape[0] == 0:
        err = 0
        pos = ref
    else:
        err = np.linalg.norm(x-ref, axis=-1)
        pos = np.mean(x, axis=0)
    return {
            'pos': pos,
            'mean': np.mean(err),
            'mse': np.mean(err**2),
            'var': np.var(err),
            'min': np.min(err),
            'max': np.max(err),
            'median': np.median(err),
            'num_particles': x.shape[0],
            }
    
errors = {}
tmp = readParticlesUni(ref_path%0, data_type="float32")[1]
ref_data = np.empty((t_end-t_start, len(tmp), 3))

for t in range(t_start, t_end):
    with open(out_path%t, 'rb') as f:
        m_d = pickle.load(f)
    print("timestep: %d/%d" % (t+1, t_end))
    ref_data[t-t_start] = readParticlesUni(ref_path%t, data_type="float32")[1]
    for i in range(len(ref_data[t-t_start])):
        errors[(i, t)] = _compute_stats(np.array(m_d[i] if i in m_d else []), ref_data[t-t_start,i])

with open(err_path, 'wb') as f:
    pickle.dump(errors, f, protocol=3)

