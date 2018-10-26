from manta import *
import tools.global_tools
from param_helpers import *

guion = True

in_path = getParam("in", "")
src_path = getParam("src", "")
ref_path = getParam("ref", "")

sdf_path = getParam("sdf", "")
src_sdf_path = getParam("src_sdf", "")
ref_sdf_path = getParam("ref_sdf", "")

res  = int(getParam("res", 150))
sres = int(getParam("sres", 2))

fac = float(getParam("fac", 1.))

t = int(getParam("t", 50))
t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", t))

t = t_end - t_start

screenshot = getParam("scr", "")

dim = int(getParam("dim", 2))

checkUnusedParams()

pause = True

gs = vec3(res, res, 1 if dim == 2 else res)
gs_show = vec3(res, res, 3 if dim == 2 else res)

s = Solver(name='high', gridSize=gs, dim=dim)
s_show = Solver(name="show", gridSize=gs_show, dim=3)

pp = s.create(BasicParticleSystem)

sdf_show = s_show.create(LevelsetGrid)
sdf_show.setBound(value=0., boundaryWidth=1)
mesh = s_show.create(Mesh)

flags_show = s.create(FlagGrid)
flags_show.initDomain()
flags_show.fillGrid(TypeEmpty)

sdf = s.create(LevelsetGrid)

gFlags   = s.create(FlagGrid)
gFlags.initDomain(4)

gIdxSys  = s.create(ParticleIndexSystem)
gIdx     = s.create(IntGrid)
gCnt     = s.create(IntGrid)

if ref_path != "":
	ref_pp = s.create(BasicParticleSystem)
	ref_sdf = s.create(LevelsetGrid)
	ref_mesh = s_show.create(Mesh)

if src_path != "":
	gs_small = vec3(res//fac, res//fac, 1 if dim == 2 else res//fac)
	gs_show_small = vec3(res//fac, res//fac, 3 if dim == 2 else res//fac)
	s_small = Solver(name='small', gridSize=gs_small, dim=dim)
	s_show_small = Solver(name='show_small', gridSize=gs_show_small, dim=3)
	src_pp = s_small.create(BasicParticleSystem)
	src_gIdxSys  = s_small.create(ParticleIndexSystem)
	src_gIdx     = s_small.create(IntGrid)
	src_gCnt     = s_small.create(IntGrid)
	src_sdf = s_small.create(LevelsetGrid)
	src_sdf_high = s.create(LevelsetGrid)
	sdf_show_small = s_show_small.create(LevelsetGrid)
	sdf_show_small.setBound(value=0., boundaryWidth=1)
	src_mesh_high = s_show.create(Mesh)
	src_mesh = s_show_small.create(Mesh)
	
if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

for i in range(t_start,t_end):
	'''if sdf_path != "":
		sdf.load(sdf_path % i)
		sdf.reinitMarching(flags=gFlags)
		pp.clear()
		sampleLevelsetWithParticles(phi=sdf, flags=gFlags, parts=pp, discretization=sres, randomness=0)
	else:'''
	pp.load(in_path % i)
	
	if sdf_path != "":
		sdf.load(sdf_path % i)
	else:
		gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt)
		unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=sdf, radiusFactor=1.0, exclude=FlagObstacle)
		extrapolateLsSimple(phi=sdf, distance=4, inside=True)
		sdf.setBound(value=5., boundaryWidth=4)
	if dim == 2:
		placeGrid2d(sdf,sdf_show,dstz=1) 
		sdf_show.createMesh(mesh)
	else:
		sdf.createMesh(mesh)

	if ref_path != "":
		ref_pp.load(ref_path % i)

		if ref_sdf_path != "":
			ref_sdf.load(ref_sdf_path % i)
		else:
			gridParticleIndex(parts=ref_pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt)
			unionParticleLevelset(parts=ref_pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=ref_sdf, radiusFactor=1.0, exclude=FlagObstacle)
			extrapolateLsSimple(phi=ref_sdf, distance=4, inside=True)
			ref_sdf.setBound(value=5., boundaryWidth=4)
		if dim == 2:
			placeGrid2d(ref_sdf,sdf_show,dstz=1) 
			sdf_show.createMesh(ref_mesh)
		else:
			ref_sdf.createMesh(ref_mesh)
	
	if src_path != "":
		src_pp.load(src_path % i)
		
		if src_sdf_path != "":
			src_sdf.load(src_sdf_path % i)
		else:
			gridParticleIndex(parts=src_pp, indexSys=src_gIdxSys, flags=gFlags, index=src_gIdx, counter=src_gCnt)
			unionParticleLevelset(parts=src_pp, indexSys=src_gIdxSys, flags=gFlags, index=src_gIdx, phi=src_sdf, radiusFactor=1.0, exclude=FlagObstacle)
			extrapolateLsSimple(phi=src_sdf, distance=4, inside=True)
			src_sdf.setBound(value=5., boundaryWidth=max(1,4//fac))
		interpolateGrid(src_sdf_high, src_sdf)
		src_sdf_high.multConst(fac)
		src_sdf_high.setBound(value=5., boundaryWidth=max(1,4))
		if dim == 2:
			placeGrid2d(src_sdf,sdf_show_small,dstz=1) 
			sdf_show_small.createMesh(src_mesh)
			placeGrid2d(src_sdf_high,sdf_show,dstz=1) 
			sdf_show.createMesh(src_mesh_high)
		else:
			src_sdf.createMesh(src_mesh)
			src_sdf_high.createMesh(src_mesh_high)

	s.step()
	if screenshot != "":
		gui.screenshot(screenshot % i if t > 1 else screenshot)
