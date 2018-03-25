import sys, os
sys.path.append("manta/scenes/tools")

from helpers import read_csv, write_csv, getParam, checkUnusedParam

import random
import math
import numpy as np

paramUsed = []

csv_in = getParam("csv", "", paramUsed)
csv_out = getParam("csv_out", "", paramUsed)
factor = float(getParam("factor", 1.0, paramUsed))
rdc_fac = float(getParam("reduce", 1.0, paramUsed))
steepness = float(getParam("steepness", 4.0, paramUsed))

if csv_out is "":
    csv_out = csv_in[:-4] + "_mod.csv"

checkUnusedParam(paramUsed)

def sigmoid(x):
  return (1 - 1 / (1 + np.exp(-steepness*x))) / rdc_fac


data = read_csv(csv_in)
print("data shape: {}".format(data.shape))

sig = sigmoid(np.arange(-1,1,2/len(data)))
data = data[np.where(np.random.random((len(data),)) < sig)]

#data = data[:min(len(data),sub_cnt)]
print("new shape: {}".format(data.shape))
write_csv(csv_out, data*factor)
