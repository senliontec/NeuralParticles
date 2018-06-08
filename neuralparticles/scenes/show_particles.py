from manta import *
import tools.global_tools
from param_helpers import *

guion = True

in_path = getParam("in", "")
sdf_path = getParam("sdf", "")

res  = int(getParam("res", 150))
sres = int(getParam("sres", 2))

t = int(getParam("t", 50))
t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", t))

t = t_end - t_start

screenshot = getParam("scr", "")

checkUnusedParams()

pause = screenshot == ""

gs = vec3(res, res, 1)

gs_show = vec3(res, res, 3)

s = Solver(name='IISPH', gridSize=gs, dim=2)
s_show = Solver(name="show", gridSize=gs_show, dim=3)

pp = s.create(BasicParticleSystem)

#if sdf_path != "":
sdf = s.create(LevelsetGrid)
sdf_show = s_show.create(LevelsetGrid)
sdf_show.setBound(value=0., boundaryWidth=1)
mesh = s_show.create(Mesh)

flags_show = s.create(FlagGrid)
flags_show.initDomain()
flags_show.fillGrid(TypeEmpty)

gFlags   = s.create(FlagGrid)

gFlags.initDomain(FlagFluid)

gIdxSys  = s.create(ParticleIndexSystem)
gIdx     = s.create(IntGrid)
gCnt     = s.create(IntGrid)

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

for i in range(t_start,t_end):
	if sdf_path != "":
		sdf.load(sdf_path % i)
		sdf.reinitMarching(flags=gFlags)
	else:
		gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt)
		unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=sdf, radiusFactor=1.0)

	placeGrid2d(sdf,sdf_show,dstz=1) 
	sdf_show.createMesh(mesh)
	
	if in_path != "":
		pp.load(in_path % i)
	elif sdf_path != "":
		pp.clear()
		sampleLevelsetWithParticles(phi=sdf, flags=gFlags, parts=pp, discretization=sres, randomness=0)
	s.step()
	if screenshot != "":
		gui.screenshot(screenshot % i if t > 1 else screenshot)
