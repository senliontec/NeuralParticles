from manta import *
from tools.helpers import *
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
if in_path != "":
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
	if in_path != "":
		pp.load(in_path % i if t > 1 else in_path)
	if sdf_path != "":
		sdf.load(sdf_path % i if t > 1 else sdf_path)
	s.step()
	if screenshot != "":
		gui.screenshot(screenshot % i if t > 1 else screenshot)
