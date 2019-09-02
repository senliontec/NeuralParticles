#import sys
#sys.path.append("tools/")

from manta import *
import math
import tools.global_tools
from param_helpers import *
import numpy as np

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

blur_sigma = float(getParam("blur", 1.0)) * float(factor_d) / 3.544908 # 3.544908 = 2 * sqrt( PI )
sdf_off = float(getParam("sdf_off", 0.0))

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

pT = pp.create(PdataInt)        # particle type
pV = pp.create(PdataVec3)       # velocity
pD = pp.create(PdataReal)       # density
pP = pp.create(PdataReal)       # pressure

gFlags   = s.create(FlagGrid)

gFlags.initDomain(max(1, 4//factor))

high_pp = high_s.create(BasicParticleSystem)

high_gIdxSys  = high_s.create(ParticleIndexSystem)
high_gIdx     = high_s.create(IntGrid)
high_gCnt     = high_s.create(IntGrid)
high_neighbor = high_s.create(ParticleNeighbors)
high_gFlags   = high_s.create(FlagGrid)
high_levelset = high_s.create(LevelsetGrid)
tmp_levelset = high_s.create(LevelsetGrid)
high_gFlags.initDomain(FlagFluid)

if dim==3:
	h_mesh	 = high_s.create(Mesh)
	mesh     = s.create(Mesh)

gIdxSys  = s.create(ParticleIndexSystem)
gIdx     = s.create(IntGrid)
gCnt     = s.create(IntGrid)
levelset = s.create(LevelsetGrid)

out = {}
if out_path != "":
	out['frame'] = 0
	out['dens'] = s.create(RealGrid)
	out['vel'] = s.create(Vec3Grid)
	out['pres'] = s.create(RealGrid)

	sm_arR = np.zeros((res if dim==3 else 1,res,res,1))
	sm_arV = np.zeros((res if dim==3 else 1,res,res,3))

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

for i in range(t):
	path = in_path % i

	pp.load(path + "_ps.uni")
	pT.load(path + "_pt.uni")
	pV.load(path + "_pv.uni")
	pD.load(path + "_pd.uni")
	pP.load(path + "_pp.uni")

	high_pp.load(path + "_ps.uni")
	high_levelset.load(path + "_sdf.uni")

	gridParticleIndex(parts=high_pp, indexSys=high_gIdxSys, flags=high_gFlags, index=high_gIdx, counter=high_gCnt)
	high_neighbor.update(pts=high_pp, indexSys=high_gIdxSys, index=high_gIdx, radius=search_r, notiming=True)

	hcnt = cntPts(t=pT, itype=FlagFluid)

	reduceParticlesNeighborsDens(pp, high_neighbor, pD, search_r, 1.0, minN, seed)
	print(pD.getMin())
	print(pD.getMax())
	lcnt = cntPts(t=pT, itype=FlagFluid)

	print("particles reduced: %d -> %d (%.1f)" % (hcnt, lcnt, hcnt/lcnt))

	
	extrapolateLsSimple(phi=high_levelset, distance=4, inside=True)
	extrapolateLsSimple(phi=high_levelset, distance=4)

	blurRealGrid(high_levelset, tmp_levelset, blur_sigma)

	high_levelset.addConst(sdf_off)
	interpolateGrid(levelset, tmp_levelset)
	levelset.multConst(res/high_res)

	extrapolateLsSimple(phi=levelset, distance=8, inside=True)
	extrapolateLsSimple(phi=levelset, distance=8)

	hcnt = cntPts(t=pT, itype=FlagFluid)
	maskParticles(pp, levelset)
	lcnt = cntPts(t=pT, itype=FlagFluid)
	print("particles reduced: %d -> %d (%.1f)" % (hcnt, lcnt, hcnt/lcnt))

	pV.multConst(vec3(1/factor_d))
	pP.multConst(1/factor)

	if out_path != "":
		path = out_path % i
		pp.save(path + "_ps.uni")
		pV.save(path + "_pv.uni")
		pD.save(path + "_pd.uni")
		pP.save(path + "_pp.uni")

		mapPartsToGridVec3(flags=gFlags, target=out['vel'], parts=pp, source=pV)
		mapPartsToGrid(flags=gFlags, target=out['dens'], parts=pp, source=pD)
		mapPartsToGrid(flags=gFlags, target=out['pres'], parts=pp, source=pP)

		#out['levelset'].multConst(high_res/res)
		levelset.save(path + "_sdf.uni")
		copyGridToArrayLevelset(target=sm_arR, source=levelset)
		np.savez_compressed(path + "_sdf.npz", sm_arR)

		extrapolateVec3Simple(out['vel'], levelset, 8)
		out['vel'].save(path + "_vel.uni")
		copyGridToArrayVec3(target=sm_arV, source=out['vel'])
		np.savez_compressed(path + "_vel.npz", sm_arV)

		out['dens'].save(path + "_dens.uni")
		copyGridToArrayReal(target=sm_arR, source=out['dens'])
		np.savez_compressed(path + "_dens.npz", sm_arR)

		out['pres'].save(path + "_pres.uni")
		copyGridToArrayReal(target=sm_arR, source=out['pres'])
		np.savez_compressed(path + "_pres.npz", sm_arR)
		
	if dim==3 and guion:
		high_levelset.createMesh(h_mesh)
		levelset.createMesh(mesh)

	s.step()
