#import sys
#sys.path.append("tools/")

from manta import *
import math
import tools.global_tools
from param_helpers import *
from tools.IISPH import IISPH

guion = int(getParam("gui", 1)) != 0
pause = int(getParam("pause", 0)) != 0

in_path = getParam("in", "")
out_path = getParam("out", "")

factor = int(getParam("factor", 9))

dim = int(getParam("dim", 2))
res = int(getParam("res", 150))
sres = int(getParam("sres", 2))

minN = int(getParam("min_n", 3))

seed = int(getParam("seed", 29837913847))

t = int(getParam("t", 50))

checkUnusedParams()

factor_d = math.pow(factor,1/dim)

low_res = int(res/factor_d)

print("grid down-scale: %d -> %d" %(res, low_res))

iisph = IISPH(low_res, dim, sres, bnd=min(1,int(4/factor_d))) # pass correct parameters!

gs = vec3(res, res, res if dim == 3 else 1)

s = Solver(name="IISPH", gridSize=gs, dim=dim)

pp = s.create(BasicParticleSystem)
pT = pp.create(PdataInt)
pV = pp.create(PdataVec3)
pP = pp.create(PdataReal)

gIdxSys  = s.create(ParticleIndexSystem)
gIdx     = s.create(IntGrid)
gCnt     = s.create(IntGrid)
neighbor = s.create(ParticleNeighbors)
gFlags   = s.create(FlagGrid)

gFlags.initDomain(FlagFluid)

levelset = s.create(LevelsetGrid)
vel = s.create(Vec3Grid)
pres = s.create(RealGrid)
low_levelset = iisph.s.create(LevelsetGrid)
low_vel = iisph.s.create(Vec3Grid)
low_pres = iisph.s.create(RealGrid)

use_sdf = True
radius = 1.0 if use_sdf else 1.2#.5*res/(low_res*sres) * math.sqrt(dim)
print(radius)

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

path = in_path % 0

pp.load(path + "_ps.uni")
pT.load(path + "_pt.uni")
pV.load(path + "_pv.uni")
pP.load(path + "_pp.uni")

for i in range(1,t):
	gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt, notiming=True)

	unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=levelset, radiusFactor=1.0, ptype=pT, exclude=FlagObstacle)
	#extrapolateLsSimple(phi=levelset, distance=4, inside=True)
	mapPartsToGridVec3(flags=gFlags, target=vel, parts=pp, source=pV)
	mapPartsToGrid(flags=gFlags, target=pres, parts=pp, source=pP)

	interpolateGrid(low_levelset, levelset)
	interpolateGridVec3(low_vel, vel)
	interpolateGrid(low_pres, pres)
	low_levelset.multConst(1.0/factor_d)
	low_vel.multConst(vec3(1.0/factor_d))
	low_pres.multConst(1.0/math.pow(factor_d, 2))

	if use_sdf:
		iisph.init_fluid(low_levelset)
	else:
		neighbor.update(pts=pp, indexSys=gIdxSys, index=gIdx, radius=radius, notiming=True)
		iisph.init_pp(pp, pT, pV, pP, neighbor)

	iisph.apply_vel(low_vel)
	#iisph.apply_pres(low_pres)

	hcnt = cntPts(t=pT, itype=FlagFluid)
	lcnt = cntPts(t=iisph.pT, itype=FlagFluid)

	print("particles reduced: %d -> %d (%.1d)" % (hcnt, lcnt, hcnt/lcnt))

	path = in_path % i

	pp.load(path + "_ps.uni")
	pT.load(path + "_pt.uni")
	pV.load(path + "_pv.uni")
	pP.load(path + "_pp.uni")

	while iisph.s.timeTotal*30 < i:
		#iisph.s.step()
		print(iisph.s.timeTotal*30)
		iisph.update()