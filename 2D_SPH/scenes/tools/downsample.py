from manta import *

from helpers import *
paramUsed = []

guion = int(getParam("gui", 1, paramUsed)) != 0
pause = bool(getParam("pause", 0, paramUsed))

in_path = getParam("in", "", paramUsed)
out_path = getParam("out", "", paramUsed)

factor = float(getParam("factor", 0.1, paramUsed))

dim = int(getParam("dim", 2, paramUsed))
res  = int(getParam("res", 50, paramUsed))
bnd = int(getParam("bnd", 4, paramUsed))

t = int(getParam("t", 50, paramUsed))

checkUnusedParam(paramUsed)

gs = vec3(round(float(res)*3.2)+bnd*2, res*3+bnd*2, res+bnd* 2 if dim==3 else 1)

s = Solver(name='IISPH', gridSize=gs, dim=dim)
pp = s.create(BasicParticleSystem)

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

for i in range(t):
	pp.load(in_path % i)
