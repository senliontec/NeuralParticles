from manta import *

from helpers import *
paramUsed = []

guion = True
pause = True

in_path = getParam("in", "", paramUsed)
sdf_path = getParam("sdf", "", paramUsed)

res  = int(getParam("res", 150, paramUsed))

t = int(getParam("t", 50, paramUsed))

screenshot = getParam("scr", "", paramUsed)

checkUnusedParam(paramUsed)

gs = vec3(res, res, 1)

s = Solver(name='IISPH', gridSize=gs, dim=2)
pp = s.create(BasicParticleSystem)
if sdf_path != "":
	sdf = s.create(LevelsetGrid)
gFlags   = s.create(FlagGrid)

gFlags.initDomain(FlagFluid)

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

for i in range(t):
	pp.load(in_path % i)
	if sdf_path != "":
		sdf.load(sdf_path % i)
	s.step()
	if screenshot != "":
		gui.screenshot(screenshot % i)
