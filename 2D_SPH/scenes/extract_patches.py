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

patch_size = int(patch_size/2)
high_patch_size = int(high_patch_size/2)

border = int(math.ceil(high_patch_size-patch_size*fac))

print("fac: %f, patch size: %d, high patch size: %d" % (fac, patch_size, high_patch_size))

checkUnusedParam(paramUsed)

backupSources(h_out_path)
backupSources(l_out_path)

props = ["vel", "dens", "pres"]

def particle_range(arr, start, end):
	for i in range(len(start)):
		arr = arr[np.where((arr[:,i]>=start[i])&(arr[:,i]<=end[i]))]
	return arr

for i in range(t_start, t_end):
	path = (l_in_path % i) + "_"
	header, l_data = readUni(path + "sdf.uni")
	l_prop_data = {}
	for p in props:
		_, l_prop_data[p] = readUni(path + p + ".uni")
	_,l_particle_data = readParticles(path+"ps.uni")

	if h_in_path != "":
		path = (h_in_path % i) + "_"
		_, h_data = readUni(path + "sdf.uni")
		h_data=np.pad(h_data,((0,0),(border,border),(border,border),(0,0)),mode="edge")
		h_prop_data = {}
		for p in props:
			_, h_prop_data[p] = readUni(path + p + ".uni")
			h_prop_data[p]=np.pad(h_prop_data[p],((0,0),(border,border),(border,border),(0,0)),mode="edge")
		_,h_particle_data = readParticles(path+"ps.uni")

	#TODO: handle also 3D
	#for z in range(1, header['dimZ']-1):
	for x in range(patch_size,header['dimX']-patch_size, stride):
		for y in range(patch_size,header['dimY']-patch_size, stride):
			if(abs(l_data[0,y,x,0]) < surface):
				path = (l_out_path%i) + "_"
				hx0 = x-patch_size
				hx1 = x+patch_size+1
				hy0 = y-patch_size
				hy1 = y+patch_size+1

                if particle_cnt > 0:
                    par = numpy.subtract(particle_range(l_particle_data, [hx0, hy0], [hx1, hy1]), [(hx0+hx1)/2, (hy0+hy1)/2, 0.])
                    if par.shape[1] < particle_cnt:
                        continue

                    idx = np.argsort(np.linalg.norm(par,axis=1))
                    writeNumpyBuf(path + "ps", par[idx[:particle_cnt]])

				data = l_fac * l_data[0,hy0:hy1,hx0:hx1]
				writeNumpyBuf(path + "sdf", numpy.tanh(data) if use_tanh else data)
				for p in props:
					writeNumpyBuf(path + p, l_prop_data[p][0,hy0:hy1,hx0:hx1])

                #writeNumpyBuf(path + "ps", numpy.subtract(particle_range(l_particle_data, [hx0, hy0], [hx1, hy1]), [hx0, hy0, 0]))

				if h_in_path != "":
					path = (h_out_path%i) + "_"
					hx0 = int(fac*x-high_patch_size)+border
					hx1 = int(fac*x+high_patch_size)+border+1
					hy0 = int(fac*y-high_patch_size)+border
					hy1 = int(fac*y+high_patch_size)+border+1

                    if particle_cnt > 0:
                        par = numpy.subtract(particle_range(h_particle_data, [hx0-border,hy0-border], [hx1-border-1,hy1-border-1]), [(hx0+hx1)/2-border,(hy0+hy1)/2-border,0])
                        idx = np.argsort(np.linalg.norm(par,axis=1))
                        writeNumpyBuf(path + "ps", par[idx[:10]])

					data = h_fac * h_data[0,hy0:hy1,hx0:hx1]
					writeNumpyBuf(path + "sdf", numpy.tanh(data) if use_tanh else data)
					for p in props:
						writeNumpyBuf(path + p, h_prop_data[p][0,hy0:hy1,hx0:hx1])
                    
					#writeNumpyBuf(path + "ps", numpy.subtract(particle_range(h_particle_data, [hx0-border,hy0-border], [hx1-border-1,hy1-border-1]), [hx0-border,hy0-border,0] ))


finalizeNumpyBufs()