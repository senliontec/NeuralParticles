from manta import *
import tools.global_tools
from param_helpers import *

guion = int(getParam("gui", 0)) != 0
pause = int(getParam("pause", 0)) != 0

in_path = getParam("in", "")

sdf_path = getParam("sdf", "")

res  = int(getParam("res", 150))

t = int(getParam("t", 50))
t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", t))

t = t_end - t_start

dim = int(getParam("dim", 2))

surface_path = getParam("out_surface", "")
foam_path = getParam("out_foam", "")

checkUnusedParams()

gs = vec3(res, res, 3 if dim == 2 else res)

s = Solver(name='solver', gridSize=gs, dim=3)
if dim == 2:
    s_tmp = Solver(name='tmp', gridSize=vec3(res,res,1), dim=2)
    sdf_tmp = s.create(LevelsetGrid)
    sdf_tmp.setBound(value=0., boundaryWidth=1)

pp = s.create(BasicParticleSystem)

sdf = s.create(LevelsetGrid)
sdf_inter = s.create(LevelsetGrid)
sdf.setBound(value=0., boundaryWidth=1)
mesh = s.create(Mesh)

gFlags   = s.create(FlagGrid)
gFlags.initDomain(4)
gFlags.fillGrid(TypeEmpty)

gIdxSys  = s.create(ParticleIndexSystem)
gIdx     = s.create(IntGrid)
gCnt     = s.create(IntGrid)

if guion:
    gui = Gui()
    gui.show()
    if pause: gui.pause()

for i in range(t_start,t_end):
    pp.load(in_path % i)
    pp.killRegion(gFlags, TypeObstacle) 
    
    if sdf_path != "":
        if dim == 2:
            sdf_tmp.load(sdf_path % i)
            placeGrid2d(sdf_tmp,sdf,dstz=1)
        else:
            sdf.load(sdf_path % i)
    else:
        gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt)
        unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=sdf, radiusFactor=1.0, exclude=FlagObstacle)
        extrapolateLsSimple(phi=sdf, distance=4, inside=True)
        sdf.setBound(value=5., boundaryWidth=4)

    blurRealGrid(sdf, sdf_inter, 1.5)
    sdf.copyFrom(sdf_inter)
    sdf.addConst(0.7)
    sdf.multConst(-1)
    maskParticles(pp, sdf)
    sdf.multConst(-1)

    sdf.createMesh(mesh)
    
    mesh.save(surface_path%i)
    pp.save(foam_path%i)

    s.step()
