from manta import *
import io
import time
from collections import OrderedDict
from tools.uniio import *
from tools.helpers import *

class NPZBuffer:
	def __init__(self, path):
		self.p = path
		self.c = 0
		self.arr_c = 0
		self.npz = readNumpy(path+"_%04d.npz"%self.c)

	def next(self):
		k = "arr_%d"%self.arr_c
		if k in self.npz:
			self.arr_c+=1
			return self.npz[k]
		else:
			self.arr_c=0
			self.c+=1
			path = self.p+"_%04d.npz"%self.c
			if not os.path.exists(path):
				return None
			self.npz = readNumpy(path)
			return self.next()

paramUsed = []

guion = True
pause = True

src_path = getParam("src", "", paramUsed)
ref_path = getParam("ref", "", paramUsed)

psize  = int(getParam("psize", 5, paramUsed))

t = int(getParam("t", 50, paramUsed))

screenshot = getParam("scr", "", paramUsed)

checkUnusedParam(paramUsed)

gs = vec3(psize, psize, 1)

s = Solver(name='patch', gridSize=gs, dim=2)

src_sdf = s.create(LevelsetGrid)
ref_sdf = s.create(LevelsetGrid)

flags = s.create(FlagGrid)
flags.initDomain(FlagFluid)

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

hdr = OrderedDict([	('dimX',psize),
			 		('dimY',psize),
					('dimZ',1),
					('gridType',17),
					('elementType',1),
					('bytesPerElement',4),
					('info',b'\0'*252),
					('dimT',0),
					('timestamp',(int)(time.time()*1e6))])
for i in range(10,t):
	src_buf = NPZBuffer(src_path%i)
	ref_buf = NPZBuffer(ref_path%i)
	while True:
		s_v = src_buf.next()
		r_v = ref_buf.next()
		if s_v is None:
			break

		#HACK: find better way!
		writeUni("tmp.uni", hdr, s_v)
		src_sdf.load("tmp.uni")
		writeUni("tmp.uni", hdr, r_v)
		ref_sdf.load("tmp.uni")
		s.step()
