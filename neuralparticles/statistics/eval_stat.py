import pickle
import numpy as np
from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.uniio import readParticlesUni
import matplotlib.pyplot as plt


err0_path = getParam("err0", "err0.pkl")
err1_path = getParam("err1", "")
ref_path = getParam("ref", "data/reference/reference_%03d.uni")
t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", -1))
out_path = getParam("out", "out_%s.pdf")

checkUnusedParams()

with open(err0_path, 'rb') as f:
    errors0 = pickle.load(f)

if err1_path != "":
    with open(err1_path, 'rb') as f:
        errors1 = pickle.load(f)

tmp = readParticlesUni(ref_path%0, data_type="float32")[1]
ref_data = np.empty((t_end-t_start, len(tmp), 3))
for t in range(t_start, t_end):
    ref_data[t] = readParticlesUni(ref_path%t, data_type="float32")[1]

gt_d1 = ref_data[1:] - ref_data[:-1]
gt_d2 = gt_d1[1:] - gt_d1[:-1]

mean_err0 = np.array([[errors0[(i, t)]['mean'] for i in range(len(ref_data[t]))] for t in range(t_start, t_end)])
mean_pos0 = np.array([[errors0[(i, t)]['pos'] for i in range(len(ref_data[t]))] for t in range(t_start, t_end)])
mean_cnt0 = np.array([[errors0[(i, t)]['num_particles'] for i in range(len(ref_data[t]))] for t in range(t_start, t_end)], dtype="float32")

mean_err0_d1 = mean_err0[1:] - mean_err0[:-1]
mean_pos0_d1 = mean_pos0[1:] - mean_pos0[:-1]
mean_cnt0_d1 = mean_cnt0[1:] - mean_cnt0[:-1]

mean_err0_d2 = np.mean(mean_err0_d1[1:] - mean_err0_d1[:-1], axis=-1)
mean_pos0_d2 = np.mean(np.linalg.norm(mean_pos0_d1[1:] - mean_pos0_d1[:-1] - gt_d2, axis=-1), axis=-1)
mean_cnt0_d2 = np.mean(mean_cnt0_d1[1:] - mean_cnt0_d1[:-1], axis=-1)

mean_err0 = np.mean(mean_err0, axis=-1)
mean_pos0 = np.mean(mean_pos0, axis=-1)
mean_cnt0 = np.mean(mean_cnt0, axis=-1)

mean_err0_d1 = np.mean(mean_err0_d1, axis=-1)
mean_pos0_d1 = np.mean(np.linalg.norm(mean_pos0_d1 - gt_d1, axis=-1), axis=-1)
mean_cnt0_d1 = np.mean(mean_cnt0_d1, axis=-1)

if err1_path != "":
    mean_err1 = np.array([[errors1[(i, t)]['mean'] for i in range(len(ref_data[t]))] for t in range(t_start, t_end)])
    mean_pos1 = np.array([[errors1[(i, t)]['pos'] for i in range(len(ref_data[t]))] for t in range(t_start, t_end)])
    mean_cnt1 = np.array([[errors1[(i, t)]['num_particles'] for i in range(len(ref_data[t]))] for t in range(t_start, t_end)], dtype="float32")

    mean_err1_d1 = mean_err1[1:] - mean_err1[:-1]
    mean_pos1_d1 = mean_pos1[1:] - mean_pos1[:-1]
    mean_cnt1_d1 = mean_cnt1[1:] - mean_cnt1[:-1]

    mean_err1_d2 = np.mean(mean_err1_d1[1:] - mean_err1_d1[:-1], axis=-1)
    mean_pos1_d2 = np.mean(np.linalg.norm(mean_pos1_d1[1:] - mean_pos1_d1[:-1] - gt_d2, axis=-1), axis=-1)
    mean_cnt1_d2 = np.mean(mean_cnt1_d1[1:] - mean_cnt1_d1[:-1], axis=-1)

    mean_err1 = np.mean(mean_err1, axis=-1)
    mean_pos1 = np.mean(mean_pos1, axis=-1)
    mean_cnt1 = np.mean(mean_cnt1, axis=-1)

    mean_err1_d1 = np.mean(mean_err1_d1, axis=-1)
    mean_pos1_d1 = np.mean(np.linalg.norm(mean_pos1_d1 - gt_d1, axis=-1), axis=-1)
    mean_cnt1_d1 = np.mean(mean_cnt1_d1, axis=-1)

print("Mean Error 0: %f" % np.mean(mean_err0))
print("Var Error 0: %f" % np.var(mean_err0))
plt.plot(mean_err0, color="#ff430f")
if err1_path != "":
    print("Mean Error 1: %f" % np.mean(mean_err1))
    print("Var Error 1: %f" % np.var(mean_err1))
    plt.plot(mean_err1, color="#014586")
plt.savefig(out_path%"err")
plt.clf()

print("Mean Error 1st Der 0: %f" % np.mean(mean_err0_d1))
print("Var Error 1st Der 0: %f" % np.var(mean_err0_d1))
plt.plot(mean_err0_d1, color="#ff430f")
if err1_path != "":
    print("Mean Error 1st Der 1: %f" % np.mean(mean_err1_d1))
    print("var Error 1st Der 1: %f" % np.var(mean_err1_d1))
    plt.plot(mean_err1_d1, color="#014586")
plt.savefig(out_path%"err_d1")
plt.clf()

print("Mean Error 0 2nd Der: %f" % np.mean(mean_err0_d2))
print("Var Error 0 2nd Der: %f" % np.var(mean_err0_d2))
plt.plot(mean_err0_d2, color="#ff430f")
if err1_path != "":
    print("Mean Error 1 2nd Der: %f" % np.mean(mean_err1_d2))
    print("Var Error 1 2nd Der: %f" % np.var(mean_err1_d2))
    plt.plot(mean_err1_d2, color="#014586")
plt.savefig(out_path%"err_d2")
plt.clf()

print("Mean Pos 1st Der 0: %f" % np.mean(mean_pos0_d1))
print("Var Pos 1st Der 0: %f" % np.var(mean_pos0_d1))
plt.plot(mean_pos0_d1, color="#ff430f")
if err1_path != "":
    print("Mean Pos 1st Der 1: %f" % np.mean(mean_pos1_d1)) 
    print("Var Pos 1st Der 1: %f" % np.var(mean_pos1_d1)) 
    plt.plot(mean_pos1_d1, color="#014586")
plt.savefig(out_path%"pos_d1")
plt.clf()

print("Mean Pos 2nd Der 0: %f" % np.mean(mean_pos0_d2))
print("Var Pos 2nd Der 0: %f" % np.var(mean_pos0_d2))
plt.plot(mean_pos0_d2, color="#ff430f")
if err1_path != "":
    print("Mean Pos 2nd Der 1: %f" % np.mean(mean_pos1_d2))
    print("Var Pos 2nd Der 1: %f" % np.var(mean_pos1_d2))
    plt.plot(mean_pos1_d2, color="#014586")
plt.savefig(out_path%"pos_d2")
plt.clf()

print("Mean Count 0: %f" % np.mean(mean_cnt0))
print("Var Count 0: %f" % np.var(mean_cnt0))
plt.plot(mean_cnt0, color="#ff430f")
if err1_path != "":
    print("Mean Count 1: %f" % np.mean(mean_cnt1))
    print("Var Count 1: %f" % np.var(mean_cnt1))
    plt.plot(mean_cnt1, color="#014586")
plt.savefig(out_path%"cnt")
plt.clf()

print("Mean Count 1st Der 0: %f" % np.mean(mean_cnt0_d1))
print("Var Count 1st Der 0: %f" % np.var(mean_cnt0_d1))
plt.plot(mean_cnt0_d1, color="#ff430f")
if err1_path != "":
    print("Mean Count 1st Der 1: %f" % np.mean(mean_cnt1_d1))
    print("Var Count 1st Der 1: %f" % np.var(mean_cnt1_d1))
    plt.plot(mean_cnt1_d1, color="#014586")
plt.savefig(out_path%"cnt_d1")
plt.clf()

print("Mean Count 2nd Der 0: %f" % np.mean(mean_cnt0_d2))
print("Var Count 2nd Der 0: %f" % np.var(mean_cnt0_d2))
plt.plot(mean_cnt0_d2, color="#ff430f")
if err1_path != "":
    print("Mean Count 2nd Der 1: %f" % np.mean(mean_cnt1_d2))
    print("Var Count 2nd Der 1: %f" % np.var(mean_cnt1_d2))
    plt.plot(mean_cnt1_d2, color="#014586")
plt.savefig(out_path%"cnt_d2")
plt.clf()
