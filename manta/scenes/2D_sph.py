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
from tools.helpers import *
paramUsed = [];

guion = int(getParam("gui", 1, paramUsed)) != 0
pause = int(getParam("pause", 0, paramUsed)) != 0

out_path = getParam("out", "", paramUsed)

# fluid init parameters


# default solver parameters
dim   = int(getParam("dim", 2, paramUsed))                  # dimension
sres  = int(getParam("sres", 2, paramUsed))                 # sub-resolution per cell
dx    = 1.0/sres 								            # particle spacing (= 2 x radius)
res   = int(getParam("res", 150, paramUsed))                # reference resolution
bnd   = int(getParam("bnd", 4, paramUsed))                  # boundary cells
dens  = float(getParam("dens", 1000.0, paramUsed))          # density
avis  = int(getParam("vis", 1, paramUsed)) != 0             # aritificial viscosity
eta   = float(getParam("eta", 0.1, paramUsed))
fps   = int(getParam("fps", 30, paramUsed))
t_end = float(getParam("t_end", 5.0, paramUsed))
sdt   = float(getParam("dt", 0, paramUsed))
circular_vel = float(getParam("circ", 0., paramUsed))

if sdt <= 0:
	sdt = None

cube_cnt = int(getParam("c_cnt", 0, paramUsed))
cube = []

class Cube:
	def __init__(self, p, s):
		self.pos = p
		self.scale = s

def stringToCube(s):
	v = s.split(",")
	if dim == 2:
		if len(v) != 4:
			print("Wrong format of cube! Format have to be: x_pos,y_pos,x_scale,y_scale")
			exit()
		return Cube(vec3(float(v[0]), float(v[1]), 0), vec3(float(v[2]), float(v[3]), 1))
	else:
		print("3D stringtocube NYI!")
		exit()

for i in range(cube_cnt):
	cube.append(stringToCube(getParam("c%d"%i, "", paramUsed)))

checkUnusedParam(paramUsed);

backupSources(out_path)

gs   = vec3(res, res, res if dim==3 else 1)
grav = -9.8 * gs.y # gravity

s             = Solver(name='IISPH', gridSize=gs, dim=dim)
s.cfl         = 1
s.frameLength = 1.0/float(fps)
s.timestepMin = 0
s.timestepMax = s.frameLength
s.timestep    = s.frameLength

overFld = FlagFluid
overAll = FlagFluid|FlagObstacle

sph  = s.create(SphWorld, delta=dx, density=dens, g=(0,grav,0), eta=eta)
kern = s.create(CubicSpline, h=sph.delta)
print('h = {}, sr = {}'.format(kern.radius(), kern.supportRadius()))

pp = s.create(BasicParticleSystem)

dummyFlags = s.create(FlagGrid)
dummyFlags.initDomain(FlagFluid)

# acceleration data for particle neighbors
gIdxSys  = s.create(ParticleIndexSystem)
gIdx     = s.create(IntGrid)
gCnt     = s.create(IntGrid)
gFlags   = s.create(FlagGrid)
neighbor = s.create(ParticleNeighbors)

pT = pp.create(PdataInt)        # particle type
pV = pp.create(PdataVec3)       # velocity
pF = pp.create(PdataVec3)       # force
pD = pp.create(PdataReal)       # density
pP = pp.create(PdataReal)       # pressure

pDadv  = pp.create(PdataReal)   # density advected
pAii   = pp.create(PdataReal)   # a_ii
pDii   = pp.create(PdataVec3)   # d_ii
pDijPj = pp.create(PdataVec3)   # sum_j(d_ii*pj)

pDtmp = pp.create(PdataReal)
pVtmp = pp.create(PdataVec3)

mesh = {}
if dim==3:
	mesh['mesh']     = s.create(Mesh)
	mesh['levelset'] = s.create(LevelsetGrid)

out = {}
if out_path != "":
	out['frame'] = 0
	out['levelset'] = s.create(LevelsetGrid)
	out['dens'] = s.create(RealGrid)
	out['vel'] = s.create(Vec3Grid)
	out['pres'] = s.create(RealGrid)

init_phi = s.create(LevelsetGrid)

# boundary setup
gFlags.initDomain(bnd-1)

begin = pp.size()
sampleFlagsWithParticles(flags=gFlags, parts=pp, discretization=sres, randomness=0, ftype=FlagObstacle, notiming=True)
end = pp.size()
pT.setConstRange(s=FlagObstacle, begin=begin, end=end, notiming=True)

# obstacle
#generateBlock(vec3(0.766, 0.08, 0.5), vec3(0.08, 0.15, 0.4), FlagObstacle)

init_phi.setConst(999.)

for c in cube:
	fld = s.create(Box, center=c.pos*gs, size=c.scale*gs)
	fld.applyToGrid(grid=gFlags, value=FlagFluid, respectFlags=gFlags)
	init_phi.join(fld.computeLevelset())

if cube_cnt == 0:
	wltstrength = 0.6
	fld = s.create(Box, center=gs*vec3(0.5,0.2,1), size=gs*vec3(1.0, 0.03,1))
	fld.applyToGrid(grid=gFlags, value=FlagFluid, respectFlags=gFlags)
	init_phi.join(fld.computeLevelset())

	wltnoise = NoiseField( parent=s, loadFromFile=True)
	# scale according to lowres sim , smaller numbers mean larger vortices
	wltnoise.posScale = vec3( int(1.0*gs.x) ) * 0.5
	wltnoise.timeAnim = 0.1

	velNoise = s.create(Vec3Grid)
	w = s.create(RealGrid)
	w.setConst(1.)
	applyNoiseVec3(gFlags, velNoise, wltnoise, scale=wltstrength, weight=w)


	wltnoise2 = NoiseField( parent=s, loadFromFile=True)
	wltnoise2.posScale = wltnoise.posScale * 2.0
	wltnoise2.timeAnim = 0.1

	wltnoise3 = NoiseField( parent=s, loadFromFile=True)
	wltnoise3.posScale = wltnoise2.posScale * 2.0
	wltnoise3.timeAnim = 0.1
	applyNoiseVec3(gFlags, velNoise, wltnoise2, scale=wltstrength*0.6 , weight=w)
	applyNoiseVec3(gFlags, velNoise, wltnoise3, scale=wltstrength*0.6*0.6 , weight=w)

	fld = s.create(Box, center=gs*vec3(0.5,0.1,1), size=gs*vec3(1.0, 0.1,1))
	fld.applyToGrid(grid=gFlags, value=FlagFluid, respectFlags=gFlags)
	init_phi.join(fld.computeLevelset())

begin = pp.size()
sampleLevelsetWithParticles(phi=init_phi, flags=gFlags, parts=pp, discretization=sres, randomness=0)
end = pp.size()
pT.setConstRange(s=FlagFluid, begin=begin, end=end, notiming=True)


if circular_vel > 0:
	fillVelocityCircular(pVtmp, pp, -circular_vel, vec3(res/2.,res/2.,0.5))
	#grav = 0

if cube_cnt == 0:
	mapGridToPartsVec3(source=velNoise, parts=pp, target=pVtmp )

# fluid setup: dam
#generateBlock(vec3(0.766, 0.08, 0.5), vec3(0.08, 0.15, 0.4), FlagFluid)

sph.bindParticleSystem(p_system=pp, p_type=pT, p_neighbor=neighbor, notiming=True)
sph.updateSoundSpeed(math.sqrt(2.0*math.fabs(grav)*0.55*gs.y/eta), notiming=True)
pD.setConst(s=sph.density, notiming=True)
gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt, notiming=True)
neighbor.update(pts=pp, indexSys=gIdxSys, index=gIdx, radius=kern.supportRadius(), notiming=True)

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

while (s.timeTotal<t_end): # main loop
	sphComputeDensity(d=pD, k=kern, sph=sph, itype=overFld, jtype=overAll)
	sphComputeConstantForce(f=pF, v=vec3(0, grav*sph.mass, 0), sph=sph, itype=overFld, accumulate=False)
	sphComputeSurfTension(f=pF, k=kern, sph=sph, kappa=0.8, itype=overFld, jtype=overAll, accumulate=True)
	if(avis):
		sphComputeArtificialViscousForce(f=pF, v=pV, d=pD, k=kern, sph=sph, itype=overFld, jtype=overFld, accumulate=True)

	if sdt is None:
		adt = min(s.frameLength, kern.supportRadius()/sph.c)
		adt = sph.limitDtByVmax(dt=adt, h=kern.supportRadius(), vmax=pV.getMaxAbsValue(), a=0.4)
		s.adaptTimestepByDt(adt)
	else:
		s.adaptTimestepByDt(sdt)

	sphUpdateVelocity(v=pVtmp, vn=pV, f=pF, sph=sph, dt=s.timestep)
	sphComputeIisphDii(dii=pDii, d=pD, k=kern, sph=sph, dt=s.timestep, itype=overFld, jtype=overAll)

	pDadv.setConst(0)
	sphComputeDivergenceSimple(div=pDadv, v=pVtmp, k=kern, sph=sph, itype=overFld, jtype=overAll) # pDadv = div(v)
	pDadv.multConst(s=-s.timestep)                                                                # pDadv = - dt*div(v)
	pDadv.add(pD)                                                                                 # pDadv = pD - dt*div(v)
	pAii.setConst(0)
	sphComputeIisphAii(aii=pAii, d=pD, dii=pDii, k=kern, sph=sph, dt=s.timestep, itype=overFld, jtype=overAll)

	######################################################################
	# solve pressure
	pP.multConst(s=0.5)         # p = 0.5*p_prev
	d_avg, iters, d_err_th = sph.density, 0, sph.density*sph.eta/100.0
	while ((d_avg - sph.density)>d_err_th) or (iters<2):
		sphComputeIisphDijPj(dijpj=pDijPj, d=pD, p=pP, k=kern, sph=sph, dt=s.timestep, itype=overFld, jtype=overAll)

		pDtmp.setConst(0.0)
		sphComputeIisphP(p_next=pDtmp, p=pP, d_adv=pDadv, d=pD, aii=pAii, dii=pDii, dijpj=pDijPj, k=kern, sph=sph, dt=s.timestep, itype=overFld, jtype=overAll)
		pDtmp.clampMin(0.0)
		pP.copyFrom(pDtmp)

		pDtmp.setConst(0.0)
		sphComputeIisphD(d_next=pDtmp, d_adv=pDadv, d=pD, p=pP, dii=pDii, dijpj=pDijPj, k=kern, sph=sph, dt=s.timestep, itype=overFld, jtype=overAll)
		d_avg = pDtmp.sum(t=pT, itype=overFld)/cntPts(t=pT, itype=overFld)

		iters += 1

		# for the safety
		if iters>999:
			print('\tFail to converge: d_avg = {} (<{}), iters = {}'.format(d_avg, d_err_th+sph.density, iters))
			sys.exit(0)

	print('\td_avg = {} (<{}), iters = {}'.format(d_avg, d_err_th+sph.density, iters))
	######################################################################

	sphComputePressureForce(f=pF, p=pP, d=pD, k=kern, sph=sph, accumulate=False)
	sphUpdateVelocity(v=pV, vn=pVtmp, f=pF, sph=sph, dt=s.timestep)

	sphUpdatePosition(x=pp, v=pV, sph=sph, dt=s.timestep)
	gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt)
	neighbor.update(pts=pp, indexSys=gIdxSys, index=gIdx, radius=kern.supportRadius())

	if out_path != "" and s.timeTotal*fps > out['frame']:
		path = out_path % out['frame']
		pp.save(path + "_ps.uni")
		pT.save(path + "_pt.uni")
		pV.save(path + "_pv.uni")
		pD.save(path + "_pd.uni")
		pP.save(path + "_pP.uni")

		unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=out['levelset'], radiusFactor=1.0, ptype=pT, exclude=FlagObstacle)
		extrapolateLsSimple(phi=out['levelset'], distance=4, inside=True)
		out['levelset'].save(path + "_sdf.uni")

		mapPartsToGridVec3(flags=gFlags, target=out['vel'], parts=pp, source=pV)
		out['vel'].save(path + "_vel.uni")

		mapPartsToGrid(flags=gFlags, target=out['dens'], parts=pp, source=pD)
		out['dens'].save(path + "_dens.uni")

		mapPartsToGrid(flags=gFlags, target=out['pres'], parts=pp, source=pP)
		out['pres'].save(path + "_pres.uni")

		out['frame']+=1

	if dim==3 and guion:
		unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=mesh['levelset'], radiusFactor=1.0, ptype=pT, exclude=FlagObstacle)
		extrapolateLsSimple(phi=mesh['levelset'], distance=4, inside=True)
		mesh['levelset'].createMesh(mesh['mesh'])

	s.step()
