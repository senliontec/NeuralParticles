from manta import *
import math
from tools.uniio import *
from tools.helpers import *
import numpy

paramUsed = []

h_in_path = getParam("h_in", "", paramUsed)
l_in_path = getParam("l_in", "", paramUsed)

h_out_path = getParam("h_out", "", paramUsed)
l_out_path = getParam("l_out", "", paramUsed)

t = int(getParam("t", 50, paramUsed))
t_start = int(getParam("t_start", 0, paramUsed))
t_end = int(getParam("t_end", t, paramUsed))

t = t_end - t_start

use_tanh = int(getParam("tanh", 0, paramUsed)) != 0
l_fac = float(getParam("l_fac", 1.0, paramUsed))
h_fac = float(getParam("h_fac", 1.0, paramUsed))

surface = float(getParam("surface", 0.5, paramUsed))

patch_size = int(getParam("psize", 5, paramUsed))
high_patch_size = int(getParam("hpsize", patch_size, paramUsed))

stride = int(getParam("stride", 1, paramUsed))

particle_cnt = int(getParam("par_cnt", 0, paramUsed))

fac = float(high_patch_size)/patch_size

border = int(math.ceil(high_patch_size//2-patch_size//2*fac))

print("fac: %f, patch size: %d, high patch size: %d" % (fac, patch_size, high_patch_size))

checkUnusedParam(paramUsed)

backupSources(h_out_path)
backupSources(l_out_path)

props = ["vel", "dens", "pres"]

for i in range(t_start, t_end):
	path = (l_in_path % i) + "_"
	header, l_data = readUni(path + "sdf.uni")
	l_prop_data = {}
	for p in props:
		l_prop_data[p] = readUni(path + p + ".uni")[1]
	l_particle_data = readParticles(path+"ps.uni")[1]

	if h_in_path != "":
		path = (h_in_path % i) + "_"
		h_data = readUni(path + "sdf.uni")[1]
		h_data=np.pad(h_data,((0,0),(border,border),(border,border),(0,0)),mode="edge")
		h_prop_data = {}
		for p in props:
			h_prop_data[p] = readUni(path + p + ".uni")[1]
			h_prop_data[p]=np.pad(h_prop_data[p],((0,0),(border,border),(border,border),(0,0)),mode="edge")
		h_particle_data = readParticles(path+"ps.uni")[1]

	patch_pos = get_patches(l_data, patch_size, header['dimX'], header['dimY'], stride, surface)

	for pos in patch_pos:
		path = (l_out_path%i) + "_"

		if particle_cnt > 0:
			par = extract_particles(l_particle_data, pos, particle_cnt)
			if par is None:
				continue

			if h_in_path != "":
				h_par = extract_particles(h_particle_data, pos, particle_cnt)
				if h_par is None:
					continue
		
		data = l_fac * extract_patch(l_data, pos, patch_size)
		writeNumpyBuf(path + "sdf", numpy.tanh(data) if use_tanh else data)
		for p in props:
			writeNumpyBuf(path + p, extract_patch(l_prop_data[p], pos, patch_size))
		if particle_cnt > 0:
			writeNumpyBuf(path + "ps", par)
		
		if h_in_path != "":
			pos = (fac*pos).astype(int)

			path = (h_out_path%i) + "_"
			data = h_fac * extract_patch(h_data, pos+border, high_patch_size)
			writeNumpyBuf(path + "sdf", numpy.tanh(data) if use_tanh else data)
			for p in props:
				writeNumpyBuf(path + p, extract_patch(h_prop_data[p], pos+border, high_patch_size))
			if particle_cnt > 0:
				writeNumpyBuf(path + "ps", h_par)

finalizeNumpyBufs()