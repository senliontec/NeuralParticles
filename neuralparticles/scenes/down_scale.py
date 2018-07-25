#import sys
#sys.path.append("tools/")

from manta import *
import math
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

blur_sigma = float(getParam("blur", 1.0))
sdf_off = float(getParam("sdf_off", 0.0))

checkUnusedParams()

res = int(high_res/math.pow(factor,1/dim))

print("grid down-scale: %d -> %d" %(high_res, res))

search_r = high_res/res * 0.40 if factor > 1 else 0#.5*high_res/(res*sres) * math.sqrt(dim)

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
high_gFlags.initDomain(FlagFluid)

if dim==3:
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

	#if upres:
	#	out['h_levelset'] = high_s.create(LevelsetGrid)
	#	out['h_dens'] = high_s.create(RealGrid)
	#	out['h_vel'] = high_s.create(Vec3Grid)
	#	out['h_pres'] = high_s.create(RealGrid)

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

	#reduceParticlesRandom(pp, factor, seed)
	#reduceParticlesNeighbors(pp, high_neighbor,minN,seed)
	#reduceParticlesPoisson(pp, high_neighbor,factor,seed)
	#reduceParticlesDens(pp, pD, factor, seed)
	reduceParticlesNeighborsDens(pp, high_neighbor, pD, search_r, 1.0, minN, seed)
	print(pD.getMinValue())
	print(pD.getMaxValue())
	lcnt = cntPts(t=pT, itype=FlagFluid)

	print("particles reduced: %d -> %d (%.1f)" % (hcnt, lcnt, hcnt/lcnt))

	if blur_sigma > 0:
		blurRealGrid(high_levelset, high_levelset, blur_sigma)

	high_levelset.addConst(sdf_off)
	interpolateGrid(levelset, high_levelset)
	levelset.multConst(res/high_res)
	
	extrapolateLsSimple(phi=levelset, distance=4, inside=True)
	extrapolateLsSimple(phi=levelset, distance=4)

	#blurRealGrid(levelset, levelset, blur_sigma)

	hcnt = cntPts(t=pT, itype=FlagFluid)
	maskParticles(pp, levelset)
	lcnt = cntPts(t=pT, itype=FlagFluid)
	print("particles reduced: %d -> %d (%.1f)" % (hcnt, lcnt, hcnt/lcnt))

	if out_path != "":
		path = out_path % i
		pp.save(path + "_ps.uni")
		pV.save(path + "_pv.uni")
		pD.save(path + "_pd.uni")
		pP.save(path + "_pp.uni")


		mapPartsToGridVec3(flags=gFlags, target=out['vel'], parts=pp, source=pV)
		mapPartsToGrid(flags=gFlags, target=out['dens'], parts=pp, source=pD)
		mapPartsToGrid(flags=gFlags, target=out['pres'], parts=pp, source=pP)

		if False:#upres:
			interpolateGrid(out['h_levelset'], levelset );
			extrapolateLsSimple(phi=out['h_levelset'], distance=4, inside=True)
			out['h_levelset'].multConst(high_res/res)
			out['h_levelset'].save(path + "_sdf.uni")

			# TODO: multiplicate by multConst(high_res/res)?
			interpolateGridVec3(out['h_vel'], out['vel'] );
			out['h_vel'].save(path + "_vel.uni")

			interpolateGrid(out['h_dens'], out['dens'] );
			out['h_dens'].save(path + "_dens.uni")

			interpolateGrid(out['h_pres'], out['pres'] );
			out['h_pres'].save(path + "_pres.uni")
		else:
			#out['levelset'].multConst(high_res/res)
			levelset.save(path + "_sdf.uni")
			out['vel'].save(path + "_vel.uni")
			out['dens'].save(path + "_dens.uni")
			out['pres'].save(path + "_pres.uni")
		
	if dim==3 and guion:
		extrapolateLsSimple(phi=levelset, distance=4, inside=True)
		levelset.createMesh(mesh)

	s.step()
