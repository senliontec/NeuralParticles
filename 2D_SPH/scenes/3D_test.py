from manta import *
from tools.helpers import *

s = Solver(name="test", gridSize=vec3(6,5,4), dim=3)

l = s.create(LevelsetGrid)

l.setConst(1.)

l.save("test.uni")