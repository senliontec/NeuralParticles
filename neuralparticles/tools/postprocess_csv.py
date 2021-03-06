import sys, os
sys.path.append("manta/scenes/tools")

from helpers import read_csv, write_csv, getParam, checkUnusedParam, plot_particles

import random
import math
import numpy as np

paramUsed = []

csv_in = getParam("csv", "", paramUsed)
csv_out = getParam("csv_out", "", paramUsed)

# reduction of data

# for sigmoid-based reduction
rdc_fac = float(getParam("reduce", 0.0, paramUsed))
steepness = float(getParam("steepness", 4.0, paramUsed))
off = float(getParam("offset", 0.0, paramUsed))

# simple clipping
clip = int(getParam("clip", 0, paramUsed))

# modification of data

# multiplicative factor
factor = float(getParam("factor", 1.0, paramUsed))
x = float(getParam("add_x", 0.0, paramUsed))
y = float(getParam("add_y", 0.0, paramUsed))

# extract detail
c_x = float(getParam("cx", 0.0, paramUsed))
c_y = float(getParam("cy", 0.0, paramUsed))
s_x = float(getParam("sx", 0.0, paramUsed))
s_y = float(getParam("sy", 0.0, paramUsed))

show = int(getParam("show", 0, paramUsed)) != 0

if csv_out is "":
    csv_out = csv_in[:-4] + "_mod.csv"

checkUnusedParam(paramUsed)

def sigmoid(x):
  return (1 - 1 / (1 + np.exp(-steepness*(x+off)))) / rdc_fac


data = read_csv(csv_in)*factor
data[:,:2] += np.array([[x,y]])
print("data shape: {}".format(data.shape))

if s_x > 0.0 and s_y > 0.0:
  c = np.array([[c_x, c_y]])
  s = np.array([[s_x, s_y]])/2
  data = data[np.where(np.all(np.concatenate(((data[:,:2] > c - s), (data[:,:2] < c + s)), axis=-1),axis=-1))[0]]
  data[:,:2] = (data[:,:2] - c) / s

if rdc_fac > 0:
  sig = sigmoid(2*np.arange(len(data))/len(data)-1)
  data = data[np.where(np.random.random((len(data),)) < sig)]

if clip > 0:
  data = data[:min(clip, len(data))]

print("new shape: {}".format(data.shape))
write_csv(csv_out, data)

if show:
  plot_particles(data)
