from manta import *
import io
import time
from collections import OrderedDict
from tools.uniio import *
from tools.helpers import *

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

if t <= 1:
	src_buf = NPZBuffer(src_path)
	ref_buf = NPZBuffer(ref_path)
	pcnt = 0
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
		if screenshot != "":
			gui.screenshot(screenshot % pcnt)
		pcnt += 1
else:
	for i in range(0,t):
		src_buf = NPZBuffer(src_path%i)
		ref_buf = NPZBuffer(ref_path%i)
		pcnt = 0
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
			if screenshot != "":
				gui.screenshot(screenshot % (i, pcnt))
				pcnt += 1
