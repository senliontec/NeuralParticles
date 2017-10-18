from manta import *

from helpers import *
paramUsed = []

guion = True
pause = True

in_path = getParam("in", "", paramUsed)
res  = int(getParam("res", 15, paramUsed))

t = int(getParam("t", 50, paramUsed))

checkUnusedParam(paramUsed)

gs = vec3(res, res, 1)

s = Solver(name='IISPH', gridSize=gs, dim=2)
#pp = s.create(BasicParticleSystem)
pp = s.create(LevelsetGrid)
gFlags   = s.create(FlagGrid)

gFlags.initDomain(FlagFluid)

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

for i in range(t):
	pp.load(in_path % i)
	s.step()
