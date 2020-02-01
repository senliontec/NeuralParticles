# ----------------------------------------------------------------------------
#
# MantaFlow fluid solver framework
# Copyright 2017 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# GNU General Public License (GPL)
# http://www.gnu.org/licenses
#
# Implicit incompressible SPH (IISPH) simulation
#
# ----------------------------------------------------------------------------

import math

from manta import *
import tools.global_tools
from param_helpers import *
import numpy as np

from tools.IISPH import IISPH

guion = int(getParam("gui", 1)) != 0
pause = int(getParam("pause", 0)) != 0

out_path = getParam("out", "")

# fluid init parameters


# default solver parameters
dim   = int(getParam("dim", 2))                  # dimension
sres  = int(getParam("sres", 2))                 # sub-resolution per cell
dx    = 1.0/sres 								            # particle spacing (= 2 x radius)
res   = int(getParam("res", 150))                # reference resolution
bnd   = int(getParam("bnd", 4))                  # boundary cells
dens  = float(getParam("dens", 1000.0))          # density
avis  = int(getParam("vis", 1)) != 0             # aritificial viscosity
eta   = float(getParam("eta", 0.1))
fps   = int(getParam("fps", 30))
grav  = -float(getParam("grav", 9.8))
t 	  = float(getParam("t", 5.0))
sdt   = float(getParam("dt", 0))
circular_vel = float(getParam("circ", 0.))
wltstrength = float(getParam("wlt", 0.))
seed = int(getParam("seed", 235))

sm_arR = np.zeros((res if dim==3 else 1,res,res,1))
sm_arV = np.zeros((res if dim==3 else 1,res,res,3))

np.random.seed(seed)

if sdt <= 0:
	sdt = None

cube_cnt = int(getParam("c_cnt", 0))
cube = []
sphere_cnt = int(getParam("s_cnt", 0))
sphere = []

class Volume:
	def __init__(self, p, s):
		self.pos = p
		self.scale = s

def stringToCube(s):
	v = s.split(",")
	if dim == 2:
		if len(v) != 4:
			print("Wrong format of cube! Format has to be: x_pos,y_pos,x_scale,y_scale")
			exit()
		return Volume(vec3(float(v[0]), float(v[1]), 0), vec3(float(v[2]), float(v[3]), 1))
	else:
		if len(v) != 6:
			print("Wrong format of cube! Format has to be: x_pos,y_pos,z_pos,x_scale,y_scale,z_scale")
			exit()
		return Volume(vec3(float(v[0]), float(v[1]), float(v[2])), vec3(float(v[3]), float(v[4]), float(v[5])))

def stringToSphere(s):
	v = s.split(",")
	if dim == 2:
		if len(v) != 3:
			print("Wrong format of sphere! Format has to be: x_pos,y_pos,radius")
			exit()
		return Volume(vec3(float(v[0]), float(v[1]), 0), float(v[2]))
	else:
		if len(v) != 4:
			print("Wrong format of sphere! Format has to be: x_pos,y_pos,z_pos,radius")
			exit()
		return Volume(vec3(float(v[0]), float(v[1]), float(v[2])), float(v[3]))


for i in range(cube_cnt):
	cube.append(stringToCube(getParam("c%d"%i, "")))
for i in range(sphere_cnt):
	sphere.append(stringToSphere(getParam("s%d"%i, "")))

checkUnusedParams();

iisph = IISPH(res, dim, sres, bnd, dens, avis, eta, fps, sdt, grav)

mesh = {}
if dim==3:
	mesh['mesh']     = iisph.s.create(Mesh)
	mesh['levelset'] = iisph.s.create(LevelsetGrid)

out = {}
if out_path != "":
	out['frame'] = 0
	out['levelset'] = iisph.s.create(LevelsetGrid)
	out['dens'] = iisph.s.create(RealGrid)
	out['vel'] = iisph.s.create(Vec3Grid)
	out['pres'] = iisph.s.create(RealGrid)

init_phi = iisph.s.create(LevelsetGrid)
init_phi.setConst(999.)

for c in cube:
	fld = iisph.s.create(Box, center=c.pos*iisph.gs, size=c.scale*iisph.gs)
	init_phi.join(fld.computeLevelset())

for s in sphere:
	fld = iisph.s.create(Sphere, center=s.pos*iisph.gs, radius=s.scale*res)
	init_phi.join(fld.computeLevelset())

if cube_cnt == 0 and sphere_cnt == 0:
	fld = iisph.s.create(Box, center=iisph.gs*vec3(0.5,0.1,0.5), size=iisph.gs*vec3(1.0, 0.1,1))
	init_phi.join(fld.computeLevelset())

init_phi.setBound(bnd)
iisph.init_fluid(init_phi)

if not wltstrength == 0:
	wltnoise = NoiseField( parent=iisph.s, loadFromFile=False)
	# scale according to lowres sim , smaller numbers mean larger vortices
	wltnoise.posScale = vec3( int(1.0*iisph.gs.x) ) * 0.1
	wltnoise.posOffset = (vec3(np.random.rand(), np.random.rand(), np.random.rand())-0.5) * 10
	wltnoise.timeAnim = 0.1

	velNoise = iisph.s.create(Vec3Grid)
	w = iisph.s.create(RealGrid)
	w.setConst(1.)
	applyNoiseVec3(iisph.gFlags, velNoise, wltnoise, scale=wltstrength, weight=w)

	wltnoise2 = NoiseField( parent=iisph.s, loadFromFile=False)
	wltnoise2.posScale = wltnoise.posScale * 2.0
	wltnoise2.posOffset = (vec3(np.random.rand(), np.random.rand(), np.random.rand())-0.5) * 10
	wltnoise2.timeAnim = 0.1

	wltnoise3 = NoiseField( parent=iisph.s, loadFromFile=False)
	wltnoise3.posScale = wltnoise2.posScale * 2.0
	wltnoise3.posOffset = (vec3(np.random.rand(), np.random.rand(), np.random.rand())-0.5) * 10
	wltnoise3.timeAnim = 0.1
	applyNoiseVec3(iisph.gFlags, velNoise, wltnoise2, scale=wltstrength, weight=w)
	applyNoiseVec3(iisph.gFlags, velNoise, wltnoise3, scale=wltstrength, weight=w)

	if dim == 2:
		velNoise.multConst(vec3(1,1,0))

	iisph.apply_vel(velNoise)

if not circular_vel == 0:
	fillVelocityCircular(iisph.pV, iisph.pp, -circular_vel, vec3(res/2.,res/2.,res/2. if dim==3 else 0.5)) 

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

vmax = 0
pmax = 0
dmax = 0
while (iisph.s.timeTotal*fps < t): # main loop
	if out_path != "" and iisph.s.timeTotal*fps > out['frame']:
		path = out_path % out['frame']
		iisph.pp.save(path + "_ps.uni")
		iisph.pT.save(path + "_pt.uni")
		iisph.pV.save(path + "_pv.uni")
		iisph.pD.save(path + "_pd.uni")
		iisph.pP.save(path + "_pp.uni")

		vmax += iisph.pV.getMax()
		pmax += iisph.pP.getMax()
		dmax += iisph.pD.getMax()

		unionParticleLevelset(parts=iisph.pp, indexSys=iisph.gIdxSys, flags=iisph.gFlags, index=iisph.gIdx, phi=out['levelset'], radiusFactor=1.0, ptype=iisph.pT, exclude=FlagObstacle)
		extrapolateLsSimple(phi=out['levelset'], distance=4, inside=True)
		extrapolateLsSimple(phi=out['levelset'], distance=4, inside=False)
		out['levelset'].save(path + "_sdf.uni")
		
		copyGridToArrayLevelset(target=sm_arR, source=out['levelset'])
		np.savez_compressed(path + "_sdf.npz", sm_arR)

		mapPartsToGridVec3(flags=iisph.gFlags, target=out['vel'], parts=iisph.pp, source=iisph.pV)
		extrapolateVec3Simple(out['vel'], out['levelset'], 8)
		out['vel'].save(path + "_vel.uni")

		copyGridToArrayVec3(target=sm_arV, source=out['vel'])
		np.savez_compressed(path + "_vel.npz", sm_arV)

		mapPartsToGrid(flags=iisph.gFlags, target=out['dens'], parts=iisph.pp, source=iisph.pD)
		out['dens'].save(path + "_dens.uni")

		copyGridToArrayReal(target=sm_arR, source=out['dens'])
		np.savez_compressed(path + "_dens.npz", sm_arR)

		mapPartsToGrid(flags=iisph.gFlags, target=out['pres'], parts=iisph.pp, source=iisph.pP)
		out['pres'].save(path + "_pres.uni")

		copyGridToArrayReal(target=sm_arR, source=out['pres'])
		np.savez_compressed(path + "_pres.npz", sm_arR)

		out['frame']+=1

		if out['frame'] == t:
			break

	if dim==3 and guion:
		unionParticleLevelset(parts=iisph.pp, indexSys=iisph.gIdxSys, flags=iisph.gFlags, index=iisph.gIdx, phi=mesh['levelset'], radiusFactor=1.0, ptype=iisph.pT, exclude=FlagObstacle)
		extrapolateLsSimple(phi=mesh['levelset'], distance=4, inside=True)
		mesh['levelset'].createMesh(mesh['mesh'])

	iisph.update()
	'''if s.timeTotal*fps > i:
		i+=1
		gui.screenshot("2D_sph_%03d.png" % i)'''

print("mean max velocity: %d" % (vmax/t))
print("mean max pressure: %d" % (pmax/t))
print("mean max density: %d" % (dmax/t))