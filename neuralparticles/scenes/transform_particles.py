#import sys
#sys.path.append("tools/")

from manta import *
import tools.global_tools
from param_helpers import *

guion = int(getParam("gui", 1)) != 0
pause = int(getParam("pause", 0)) != 0

in_path = getParam("in", "")
out_path = getParam("out", "")

dim = int(getParam("dim", 2))
res = int(getParam("res", 150))
low_res = int(0.5*res)

curv = int(getParam("curv", 0)) != 0
peaks = int(getParam("peaks", 1))

t = int(getParam("t", 50))

bnd = int(getParam("bnd", 4))

mode = getParam("mode", "nor")
factor = getParam("fac", "1").split(",")

if dim == 2:
    if len(factor) == 1:
        factor = vec3(float(factor[0]),float(factor[0]),0)
    else:
        factor = vec3(float(factor[0]), float(factor[1]), 0)
elif dim == 3:
    if len(factor) == 1:
        factor = vec3(float(factor[0]))
    else:
        factor = vec3(float(factor[0]), float(factor[1]), float(factor[2]))

checkUnusedParams()

gs = vec3(res, res, res if dim == 3 else 1)
gs_low = vec3(low_res, low_res, low_res if dim == 3 else 1)
s = Solver(name="IISPH", gridSize=gs, dim=dim)
s_low = Solver(name="low", gridSize=gs_low, dim=dim)

pp = s.create(BasicParticleSystem)

pT = pp.create(PdataInt)        # particle type
pV = pp.create(PdataVec3)       # velocity
pD = pp.create(PdataReal)       # density
pP = pp.create(PdataReal)       # pressure

gIdxSys  = s.create(ParticleIndexSystem)
gIdx     = s.create(IntGrid)
gCnt     = s.create(IntGrid)

gFlags   = s.create(FlagGrid)

gFlags.initDomain(bnd)

sdf = s.create(LevelsetGrid)

trans = s.create(Vec3Grid)

low_sdf = s_low.create(LevelsetGrid)

grad = s_low.create(Vec3Grid)
if curv:
    curv = s_low.create(RealGrid)
if mode == 'cos':
    cos = s_low.create(RealGrid)

if dim==3:
    mesh     = s.create(Mesh)

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
    sdf.load(path + "_sdf.uni")

    '''if out_path != "":
        path = out_path % i
        pT.save(path + "_pt.uni")
        pV.save(path + "_pv.uni")
        pD.save(path + "_pd.uni")
        pP.save(path + "_pp.uni")'''

    extrapolateLsSimple(phi=sdf, distance=4, inside=True)
    if dim==3 and guion:
        sdf.createMesh(mesh)

    interpolateGrid(low_sdf, sdf)

    if mode == 'nor':
        getGradientGrid(grad, low_sdf)
    elif mode == 'const':
        grad.setConst(vec3(1.))
    elif mode == 'cos':
        getGradientGrid(grad, low_sdf)
        cosDisplacement(cos, grad, peaks)
        multGridVec(grad, cos)
    else:
        print("Mode '%s' not supported!" % mode)

    if curv:
        getCurvature(curv, low_sdf)
        multGridVec(grad, curv)

    interpolateGridVec3(trans, grad)
    trans.multConst(factor)

    mapGridToPartsVec3(source=trans, parts=pp, target=pV)

    s.step()

    pp.advect(pV, pT, exclude=FlagObstacle)    

    gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt)
    unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=sdf, radiusFactor=1.0, ptype=pT, exclude=FlagObstacle)
    
    extrapolateLsSimple(phi=sdf, distance=4, inside=True)

    if out_path != "":
        path = out_path % i
        pp.save(path + "_ps.uni")
        sdf.save(path + "_sdf.uni")
        
    if dim==3 and guion:
        sdf.createMesh(mesh)

    s.step()
