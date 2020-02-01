#import sys
#sys.path.append("tools/")

from manta import *
import math
import os
import tools.global_tools
from param_helpers import *

guion = int(getParam("gui", 1)) != 0
pause = int(getParam("pause", 0)) != 0

in_path = getParam("in", "")
out_path = getParam("out", "")

factor = int(getParam("factor", 10))

dim = int(getParam("dim", 2))
high_res = int(getParam("res", 150))
sres = int(getParam("sres", 2))

minN = int(getParam("min_n", 3))

seed = int(getParam("seed", 29837913847))

t = int(getParam("t", 50))

factor_d = math.pow(factor,1/dim)

checkUnusedParams()

res = int(high_res/factor_d)

print("grid down-scale: %d -> %d" %(high_res, res))

search_r = high_res/res * (1/sres) * 0.77 if factor > 1 else 0#0.73

print("search radius: %f" % search_r)

high_gs = vec3(high_res, high_res, high_res if dim == 3 else 1)
gs = vec3(res, res, res if dim == 3 else 1)

s = Solver(name="low", gridSize=gs, dim=dim)
high_s = Solver(name='IISPH', gridSize=high_gs, dim=dim)

pp = s.create(BasicParticleSystem)

gFlags   = s.create(FlagGrid)

gFlags.initDomain(max(1, 4//factor))

high_pp = high_s.create(BasicParticleSystem)

high_gIdxSys  = high_s.create(ParticleIndexSystem)
high_gIdx     = high_s.create(IntGrid)
high_gCnt     = high_s.create(IntGrid)
high_neighbor = high_s.create(ParticleNeighbors)
high_gFlags   = high_s.create(FlagGrid)
high_gFlags.initDomain(FlagFluid)

out = {}
if out_path != "":
	out['frame'] = 0

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

for i in range(t):
	path = in_path % i

	pp.load(path + "_ps.uni")
	high_pp.load(path + "_ps.uni")	

	gridParticleIndex(parts=high_pp, indexSys=high_gIdxSys, flags=high_gFlags, index=high_gIdx, counter=high_gCnt)
	high_neighbor.update(pts=high_pp, indexSys=high_gIdxSys, index=high_gIdx, radius=search_r, notiming=True)

	hcnt = pp.pySize()

	reduceParticlesNeighbors(pp, high_neighbor,minN,seed)
	lcnt = pp.pySize()

	print("particles reduced: %d -> %d (%.1f)" % (hcnt, lcnt, hcnt/lcnt))

	if out_path != "":
		path = out_path % i
		pp.save(path + "_ps.uni")

	s.step()
