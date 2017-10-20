from manta import *
import math
from tools.uniio import *
from tools.helpers import *

paramUsed = []

h_in_path = getParam("h_in", "", paramUsed)
l_in_path = getParam("l_in", "", paramUsed)

h_out_path = getParam("h_out", "", paramUsed)
l_out_path = getParam("l_out", "", paramUsed)

t = int(getParam("t", 50, paramUsed))

surface = float(getParam("surface", 0.5, paramUsed))

patch_size = int(int(getParam("psize", 5, paramUsed)) / 2)
stride = int(getParam("stride", 1, paramUsed))

checkUnusedParam(paramUsed)

backupSources(h_out_path)
backupSources(l_out_path)

props = ["vel", "dens", "pres"]

for i in range(t):
	path = (l_in_path % i) + "_"
	header, l_data = readUni(path + "sdf.uni")
	l_prop_data = {}
	for p in props:
		_, l_prop_data[p] = readUni(path + p + ".uni")

	path = (h_in_path % i) + "_"
	_, h_data = readUni(path + "sdf.uni")
	h_prop_data = {}
	for p in props:
		_, h_prop_data[p] = readUni(path + p + ".uni")

	#TODO: handle also 3D
	#for z in range(1, header['dimZ']-1):
	for x in range(patch_size,header['dimX']-patch_size, stride):
		for y in range(patch_size,header['dimY']-patch_size, stride):
			if(abs(l_data[0,y,x]) < surface):
				path = (l_out_path%i) + "_"
				writeNumpyBuf(path + "sdf", l_data[0,y-patch_size:y+patch_size+1,x-patch_size:x+patch_size+1])
				for p in props:
					writeNumpyBuf(path + p, l_prop_data[p][0,y-patch_size:y+patch_size+1,x-patch_size:x+patch_size+1])

				path = (h_out_path%i) + "_"
				writeNumpyBuf(path + "sdf", h_data[0,y-patch_size:y+patch_size+1,x-patch_size:x+patch_size+1])
				for p in props:
					writeNumpyBuf(path + p, h_prop_data[p][0,y-patch_size:y+patch_size+1,x-patch_size:x+patch_size+1])

finalizeNumpyBufs()

'''
s = Solver(name="main", gridSize=gs, dim=dim)

levelset = s.create(LevelsetGrid)
dens = s.create(RealGrid)
vel = s.create(Vec3Grid)
pres = s.create(RealGrid)
gFlags   = s.create(FlagGrid)

gFlags.initDomain(FlagFluid)

high_levelset = s.create(LevelsetGrid)
high_dens = s.create(RealGrid)
high_vel = s.create(Vec3Grid)
high_pres = s.create(RealGrid)

high_gFlags   = s.create(FlagGrid)

high_gFlags.initDomain(FlagFluid)

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

for i in range(t):
	path = low_res_path % i
	levelset.load(path + "_sdf.uni")
	dens.load(path + "_dens.uni")
	vel.load(path + "_vel.uni")
	pres.load(path + "_pres.uni")

	path = high_res_path % i
	high_levelset.load(path + "_sdf.uni")
	high_dens.load(path + "_dens.uni")
	high_vel.load(path + "_vel.uni")
	high_pres.load(path + "_pres.uni")

	s.step()
'''
