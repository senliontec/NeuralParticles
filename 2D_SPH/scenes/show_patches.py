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
ref2_path = getParam("ref2", "", paramUsed)

psize  = int(getParam("psize", 5, paramUsed))
hpsize = int(getParam("hpsize", psize, paramUsed))

t = int(getParam("t", 50, paramUsed))

screenshot = getParam("scr", "", paramUsed)

checkUnusedParam(paramUsed)

gs = vec3(psize, psize, 1)
gs_ref = vec3(hpsize, hpsize, 1)

s = Solver(name='patch', gridSize=gs, dim=2)
s_ref = Solver(name='hpatch', gridSize=gs_ref, dim=2)

src_sdf = s.create(LevelsetGrid)
ref_sdf = s_ref.create(LevelsetGrid)
if ref2_path != "":
	ref2_sdf = s_ref.create(LevelsetGrid)

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

ref_hdr = OrderedDict([	('dimX',hpsize),
			 		('dimY',hpsize),
					('dimZ',1),
					('gridType',17),
					('elementType',1),
					('bytesPerElement',4),
					('info',b'\0'*252),
					('dimT',0),
					('timestamp',(int)(time.time()*1e6))])

for i in range(0, t):
	src_buf = NPZBuffer(src_path if t == 1 else src_path%i)
	ref_buf = NPZBuffer(ref_path if t == 1 else ref_path%i)
	if ref2_path != "":
		ref2_buf = NPZBuffer(ref2_path)

	pcnt = 0
	while True:
		s_v = src_buf.next()
		r_v = ref_buf.next()
		if ref2_path != "":
			r2_v = ref2_buf.next()

		if s_v is None:
			break

		#HACK: find better way!
		writeUni("tmp.uni", hdr, s_v)
		src_sdf.load("tmp.uni")
		writeUni("tmp.uni", ref_hdr, r_v)
		ref_sdf.load("tmp.uni")
		if ref2_path != "":
			writeUni("tmp.uni", ref_hdr, r2_v)
			ref2_sdf.load("tmp.uni")

		s.step()
		if screenshot != "":
			gui.screenshot(screenshot % (pcnt if t == 1 else (i, pcnt)))
			pcnt += 1
