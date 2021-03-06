from manta import *
import tools.global_tools
from param_helpers import *

guion = int(getParam("gui", 0)) != 0
pause = int(getParam("pause", 0)) != 0

in_path = getParam("in", "")

sdf_path = getParam("sdf", "")

res  = int(getParam("res", 150))

surface_fac = float(getParam("surface", 0))
blur_fac = float(getParam("blur", 0))

t = int(getParam("t", 50))
t_start = int(getParam("t_start", 0))
t_end = int(getParam("t_end", t))

t = t_end - t_start

dim = int(getParam("dim", 2))

bnd = int(getParam("bnd", 4))

surface_path = getParam("out_surface", "")
foam_path = getParam("out_foam", "")

eps = float(getParam("eps", 0.01))

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
sdf.setBound(value=5., boundaryWidth=1)
mesh = s.create(Mesh)

gFlags   = s.create(FlagGrid)
gFlags.initDomain(bnd)
gFlags.fillGrid(TypeEmpty)

gIdxSys  = s.create(ParticleIndexSystem)
gIdx     = s.create(IntGrid)
gCnt     = s.create(IntGrid)
#neighbor = s.create(ParticleNeighbors)

if guion:
    gui = Gui()
    gui.show()
    if pause: gui.pause()

for i in range(t_start,t_end):
    pp.load(in_path % i)
    #pp.killRegion(gFlags, TypeObstacle) 

    #gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt)
    #neighbor.update(pts=pp, indexSys=gIdxSys, index=gIdx, radius=eps, notiming=True)

    print("Particle Cnt: %d" % pp.pySize())
    #reduceParticles(pp, eps)
    #reduceParticlesNeighbors(pp, neighbor, 0)
    print("Particle Cnt Reduced: %d" % pp.pySize())
    
    if sdf_path != "":
        if dim == 2:
            sdf_tmp.load(sdf_path % i)
            placeGrid2d(sdf_tmp,sdf,dstz=1)
        else:
            sdf.load(sdf_path % i)
    else:
        gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt)
        unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=sdf, radiusFactor=1.0)#, exclude=FlagObstacle)
        extrapolateLsSimple(phi=sdf, distance=4, inside=True)
        sdf.setBound(value=5., boundaryWidth=bnd)
    if surface_fac > 0 or blur_fac > 0:
        if blur_fac > 0:
            blurRealGrid(sdf, sdf_inter, 1.5 * blur_fac)
            sdf.copyFrom(sdf_inter)
        sdf.addConst(surface_fac)
        sdf.multConst(-1)
        maskParticles(pp, sdf)
        sdf.multConst(-1)

    sdf.createMesh(mesh)
    
    mesh.save(surface_path%i)
    pp.save(foam_path%i)

    s.step()
