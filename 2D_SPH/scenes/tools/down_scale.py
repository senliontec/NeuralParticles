from manta import *

from helpers import *
paramUsed = []

guion = int(getParam("gui", 1, paramUsed)) != 0
pause = int(getParam("pause", 0, paramUsed)) != 0

in_path = getParam("in", "", paramUsed)
out_path = getParam("out", "", paramUsed)

factor = int(getParam("factor", 10, paramUsed))

dim = int(getParam("dim", 2, paramUsed))
res  = int(getParam("res", 150, paramUsed))
bnd = int(getParam("bnd", 4, paramUsed))

t = int(getParam("t", 50, paramUsed))

checkUnusedParam(paramUsed)

# some random seed for the downscale
seed = 29837913847

gs = vec3(res, res, 1)

s = Solver(name='IISPH', gridSize=gs, dim=dim)
pp = s.create(BasicParticleSystem)

pT = pp.create(PdataInt)        # particle type
pV = pp.create(PdataVec3)       # velocity
pD = pp.create(PdataReal)       # density
pP = pp.create(PdataReal)       # pressure

gFlags   = s.create(FlagGrid)

gFlags.initDomain(FlagFluid)

out = {}
if out_path != "":
	out['gIdxSys'] = s.create(ParticleIndexSystem)
	out['gIdx'] = s.create(IntGrid)
	out['gCnt'] = s.create(IntGrid)

	out['frame'] = 0
	out['levelset'] = s.create(LevelsetGrid)
	out['dens'] = s.create(RealGrid)
	out['vel'] = s.create(Vec3Grid)
	#out['velOld'] = s.create(Vec3Grid)
	out['pres'] = s.create(RealGrid)

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

	reduceParticlesRandom(pp, factor, seed)

	if out_path != "":
		path = out_path % i
		pp.save(path + "_ps.uni")

		gridParticleIndex(parts=pp, indexSys=out['gIdxSys'], flags=gFlags, index=out['gIdx'], counter=out['gCnt'])

		unionParticleLevelset(parts=pp, indexSys=out['gIdxSys'], flags=gFlags, index=out['gIdx'], phi=out['levelset'], radiusFactor=1.0, ptype=pT, exclude=FlagObstacle)
		extrapolateLsSimple(phi=out['levelset'], distance=4, inside=True)
		out['levelset'].save(path + "_sdf.uni")

		mapPartsToGridVec3(flags=gFlags, target=out['vel'], parts=pp, source=pV)
		out['vel'].save(path + "_vel.uni")

		mapPartsToGrid(flags=gFlags, target=out['dens'], parts=pp, source=pD)
		out['dens'].save(path + "_dens.uni")

		mapPartsToGrid(flags=gFlags, target=out['pres'], parts=pp, source=pP)
		out['pres'].save(path + "_pres.uni")

	s.step()
