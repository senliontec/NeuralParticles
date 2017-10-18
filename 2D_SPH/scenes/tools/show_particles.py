from manta import *

from helpers import *
paramUsed = []

guion = True
pause = True

in_path = getParam("in", "", paramUsed)
out_path = getParam("out", "", paramUsed)

factor = float(getParam("factor", 0.1, paramUsed))

res  = int(getParam("res", 50, paramUsed))
bnd = int(getParam("bnd", 4, paramUsed))

t = int(getParam("t", 150, paramUsed))

checkUnusedParam(paramUsed)

gs = vec3(round(float(res)*3.2)+bnd*2, res*3+bnd*2, res+bnd* 2 if dim==3 else 1)

s = Solver(name='IISPH', gridSize=gs, dim=dim)
pp = s.create(BasicParticleSystem)
gFlags   = s.create(FlagGrid)

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()
	'''if dim == 3:
		gIdxSys = s.create(ParticleIndexSystem)
		gIdx = s.create(IntGrid)
		mesh = s.create(Mesh)
		levelset = s.create(LevelsetGrid)'''

for i in range(t):
	pp.load(in_path % i)
	'''if dim==3 and guion:
		unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=levelset, radiusFactor=1.0, ptype=pT)
		extrapolateLsSimple(phi=mesh['levelset'], distance=4, inside=True)
		mesh['levelset'].createMesh(mesh['mesh'])'''
	s.step()
