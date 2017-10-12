# ----------------------------------------------------------------------------
#
# MantaFlow fluid solver framework
# Copyright 2017 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# GNU General Public License (GPL)
# http://www.gnu.org/licenses
#
# Weakly compressible SPH (WCSPH) simulation
#
# ----------------------------------------------------------------------------

import math

guion = False
pause = True

# default solver parameters
params          = {}
params['dim']   = 2                  # dimension
params['sres']  = 2                  # sub-resolution per cell
params['dx']    = 1.0/params['sres'] # particle spacing (= 2 x radius)
params['res']   = 50                 # reference resolution
params['len']   = 1.0                # reference length
params['bnd']   = 4                  # boundary cells
params['gref']  = -9.8               # real-world gravity
params['dens']  = 1000.0             # density
params['avis']  = True               # aritificial viscosity
params['eta']   = 0.1
params['fps']   = 30
params['t_end'] = 5.0
params['sdt']   = None

scaleToManta   = float(params['res'])/params['len']
# NOTE: the original test uses 3.22; but, here it's slightly modified for the sake of convenience in discretization
params['gs']   = [round(float(params['res'])*3.2)+params['bnd']*2, params['res']*3+params['bnd']*2, params['res']+params['bnd']*2 if params['dim']==3 else 1]
params['grav'] = params['gref']*scaleToManta

s             = Solver(name='WCSPH', gridSize=vec3(params['gs'][0], params['gs'][1], params['gs'][2]), dim=params['dim'])
s.cfl         = 1
s.frameLength = 1.0/float(params['fps'])
s.timestepMin = 0
s.timestepMax = s.frameLength
s.timestep    = s.frameLength

sph  = s.create(SphWorld, delta=params['dx'], density=params['dens'], g=(0,params['grav'],0), eta=params['eta'])
kern = s.create(CubicSpline, h=sph.delta) # WCSPH uses 4*radius for the supporting radius (=h*2)
print('h = {}, sr = {}'.format(kern.radius(), kern.supportRadius()))

pp = s.create(BasicParticleSystem)

# acceleration data for particle neighbors
gIdxSys  = s.create(ParticleIndexSystem)
gIdx     = s.create(IntGrid)
gCnt     = s.create(IntGrid)
gFlags   = s.create(FlagGrid)
neighbor = s.create(ParticleNeighbors)

pT = pp.create(PdataInt)        # particle type
pV = pp.create(PdataVec3)       # velocity
pF = pp.create(PdataVec3)       # force
pP = pp.create(PdataReal)       # pressure
pD = pp.create(PdataReal)       # density

mesh = {}
if params['dim']==3:
	mesh['mesh']     = s.create(Mesh)
	mesh['levelset'] = s.create(LevelsetGrid)

# boundary setup
gFlags.initDomain(params['bnd']-1)

begin = pp.size()
sampleFlagsWithParticles(flags=gFlags, parts=pp, discretization=params['sres'], randomness=0, ftype=FlagObstacle, notiming=True)
end = pp.size()
pT.setConstRange(s=FlagObstacle, begin=begin, end=end, notiming=True)

# obstacle
a   = vec3(0.744*scaleToManta+params['bnd'], 0.161*0.5*scaleToManta+params['bnd'], 0.5*params['gs'][2] if (params['dim']==3) else 0)
b   = vec3(0.161*0.5*scaleToManta, 0.161*0.5*scaleToManta, 0.403*0.5*scaleToManta if (params['dim']==3) else params['gs'][2])
obs = s.create(Box, center=a, size=b)

begin = pp.size()
sampleShapeWithParticles(shape=obs, flags=gFlags, parts=pp, discretization=params['sres'], randomness=0, notiming=True)
end = pp.size()
pT.setConstRange(s=FlagObstacle, begin=begin, end=end, notiming=True)
obs.applyToGrid(grid=gFlags, value=FlagObstacle, respectFlags=gFlags)

# fluid setup: dam
dam_c = [2.606, 0.275, 0.5]
dam_s = [1.228*0.5, 0.55*0.5, 0.5]
a     = vec3(dam_c[0]*scaleToManta+params['bnd'], dam_c[1]*scaleToManta+params['bnd'], dam_c[2]*scaleToManta+params['bnd'] if (params['dim']==3) else 0)
b     = vec3(dam_s[0]*scaleToManta, dam_s[1]*scaleToManta, dam_s[2]*scaleToManta if (params['dim']==3) else params['gs'][2])
fld   = s.create(Box, center=a, size=b)

begin = pp.size()
sampleShapeWithParticles(shape=fld, flags=gFlags, parts=pp, discretization=params['sres'], randomness=0, notiming=True)
end = pp.size()
pT.setConstRange(s=FlagFluid, begin=begin, end=end, notiming=True)

sph.bindParticleSystem(p_system=pp, p_type=pT, p_neighbor=neighbor, notiming=True)
sph.updateSoundSpeed(math.sqrt(2.0*math.fabs(params['grav'])*0.4*scaleToManta/params['eta']), notiming=True)
pD.setConst(s=sph.density, notiming=True)
pP.setConst(s=0, notiming=True)
gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt, notiming=True)
neighbor.update(pts=pp, indexSys=gIdxSys, index=gIdx, radius=kern.supportRadius(), pT=pT, exclude=FlagObstacle, notiming=True)

bfkern = s.create(BndKernel, c=sph.c, h=sph.delta*0.5) # special for handling the boundary force

if guion:
	gui = Gui()
	gui.show()
	if pause: gui.pause()

while (s.timeTotal<params['t_end']): # main loop
	sphComputeDensity(d=pD, k=kern, sph=sph)
	sphComputePressure(p=pP, d=pD, sph=sph)
	pP.clampMin(0.0)

	sphComputeConstantForce(f=pF, v=vec3(0, params['grav']*sph.mass, 0), sph=sph, accumulate=False)
	sphComputePressureForce(f=pF, p=pP, d=pD, k=kern, sph=sph, accumulate=True)
	sphComputeBoundaryForce(f=pF, k=bfkern, sph=sph, accumulate=True)
	if(params['avis']): sphComputeArtificialViscousForce(f=pF, v=pV, d=pD, k=kern, sph=sph, accumulate=True)

	if params['sdt'] is None:
		adt = s.frameLength
		adt = sph.limitDtByVmax(dt=adt, h=kern.supportRadius(), vmax=pF.getMaxAbsValue()/sph.mass, a=0.25)
		adt = sph.limitDtByViscous(dt=adt, h=kern.supportRadius())
		s.adaptTimestepByDt(adt)
	else:
		s.adaptTimestepByDt(params['sdt'])

	sphUpdateVelocity(v=pV, vn=pV, f=pF, sph=sph, dt=s.timestep)

	sphUpdatePosition(x=pp, v=pV, sph=sph, dt=s.timestep)
	gridParticleIndex(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, counter=gCnt)
	neighbor.update(pts=pp, indexSys=gIdxSys, index=gIdx, radius=kern.supportRadius(), pT=pT, exclude=FlagObstacle)

	if params['dim']==3 and guion:
		unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=mesh['levelset'], radiusFactor=1.0, ptype=pT, exclude=FlagObstacle)
		extrapolateLsSimple(phi=mesh['levelset'], distance=4, inside=True)
		mesh['levelset'].createMesh(mesh['mesh'])

	s.step()
