from manta import *
from tools.helpers import *
paramUsed = []

guion = True
pause = True

sdf_path = getParam("sdf", "", paramUsed)

res  = int(getParam("res", 50, paramUsed))

upres = int(getParam("upres", 3, paramUsed))

t = int(getParam("t", 50, paramUsed))
t_start = int(getParam("t_start", 0, paramUsed))
t_end = int(getParam("t_end", t, paramUsed))

t = t_end - t_start

screenshot = getParam("scr", "", paramUsed)

checkUnusedParam(paramUsed)

gs = vec3(res, res, 1)
gs_upres = vec3(upres, upres, 1)
gs_show = vec3(upres, upres, 3)

s_upres = Solver(name='upres', gridSize=gs_upres, dim=2)
s_show = Solver(name="show", gridSize=gs_show, dim=3)
s = Solver(name='IISPH', gridSize=gs, dim=2)

sdf_upres = s_upres.create(LevelsetGrid)

sdf_show = s_show.create(LevelsetGrid)
sdf_show.setBound(value=0., boundaryWidth=1)
mesh = s_show.create(Mesh)

flags_show = s.create(FlagGrid)
flags_show.initDomain()
flags_show.fillGrid(TypeEmpty)

sdf = s.create(LevelsetGrid)
gFlags   = s.create(FlagGrid)
gFlags.initDomain(FlagFluid)

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

for i in range(t_start,t_end):
	sdf.load(sdf_path % i if t > 1 else sdf_path)
	sdf.reinitMarching(flags=gFlags)
	interpolateGrid(sdf_upres, sdf)
	placeGrid2d(sdf_upres,sdf_show,dstz=1) 
	sdf_show.createMesh(mesh)
	s.step()
	if screenshot != "":
		gui.screenshot(screenshot % i if t > 1 else screenshot)
