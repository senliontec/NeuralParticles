from manta import *
import tools.global_tools
from param_helpers import *

guion = int(getParam("gui", 1)) != 0
pause = int(getParam("pause", 0)) != 0

in_path = getParam("in", "")
out_path = getParam("out", "")

data_cnt = int(getParam("cnt", 1))
timesteps = int(getParam("t", 1))

dim = int(getParam("dim", 2))
res = int(getParam("res", 150))

checkUnusedParams()


gs = vec3(res, res, 1 if dim == 2 else res)
gs_show = vec3(res, res, 3 if dim == 2 else res)

s = Solver(name='sim', gridSize=gs, dim=dim)
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

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

for d in range(data_cnt):
    for t in range(timesteps):
        pp.load(in_path % (d,t))
        print(pp.pySize())

        gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt)
        unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=sdf, radiusFactor=1.0, exclude=FlagObstacle)
        extrapolateLsSimple(phi=sdf, distance=4, inside=True)
        sdf.setBound(value=5., boundaryWidth=4)
        if dim == 2:
            placeGrid2d(sdf,sdf_show,dstz=1) 
            sdf_show.createMesh(mesh)
        else:
            sdf.createMesh(mesh)

        sdf.save(out_path % (d,t))

        s.step()
