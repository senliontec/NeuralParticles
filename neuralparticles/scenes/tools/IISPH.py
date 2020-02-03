from manta import *
import tools.global_tools
import numpy as np
import sys
import math

class IISPH:
    def __init__(self, 
                 res, 
                 dim=2, 
                 sres=2, 
                 bnd=4, 
                 dens=1000.0, 
                 avis=True,
                 eta=0.1,
                 fps=30,
                 sdt=None,
                 grav=-9.8):
        self.res = res
        self.dim = dim
        self.sres = sres
        self.eta = eta
        self.avis = avis
        self.sdt = sdt
        self.bnd = bnd

        self.gs = vec3(res, res, res if dim==3 else 1)
        self.grav = grav * self.gs.y

        dx = 1.0/sres

        self.s = Solver(name='IISPH_{}'.format(res), gridSize=self.gs, dim=dim)
        self.s.cfl         = 1
        self.s.frameLength = 1.0/float(fps)
        self.s.timestepMin = self.s.frameLength * 0.01
        self.s.timestepMax = self.s.frameLength
        self.s.timestep    = self.s.frameLength

        self.overFld = 1 # FlagFluid
        self.overAll = 1|2 # FlagFluid|FlagObstacle

        self.sph  = self.s.create(SphWorld, delta=dx, density=dens, g=(0,self.grav,0), eta=eta)
        self.kern = self.s.create(CubicSpline, h=self.sph.delta)
        print('h = {}, sr = {}'.format(self.kern.radius(), self.kern.supportRadius()))

        self.pp = self.s.create(BasicParticleSystem)

        #dummyFlags = self.s.create(FlagGrid) dfg
        #dummyFlags.initDomain(4)

        self.gIdxSys  = self.s.create(ParticleIndexSystem)
        self.gIdx     = self.s.create(IntGrid)
        self.gCnt     = self.s.create(IntGrid)
        self.gFlags   = self.s.create(FlagGrid)
        self.neighbor = self.s.create(ParticleNeighbors)

        # boundary setup
        self.gFlags.initDomain(self.bnd-1)
        
        self.pT = self.pp.create(PdataInt)        # particle type
        self.pV = self.pp.create(PdataVec3)       # velocity
        self.pF = self.pp.create(PdataVec3)       # force
        self.pD = self.pp.create(PdataReal)       # density
        self.pP = self.pp.create(PdataReal)       # pressure

        self.pDadv  = self.pp.create(PdataReal)   # density advected
        self.pAii   = self.pp.create(PdataReal)   # a_ii
        self.pDii   = self.pp.create(PdataVec3)   # d_ii
        self.pDijPj = self.pp.create(PdataVec3)   # sum_j(d_ii*pj)

        self.pDtmp = self.pp.create(PdataReal)
        self.pVtmp = self.pp.create(PdataVec3)
      
    def init_sph(self):
        self.sph.bindParticleSystem(p_system=self.pp, p_type=self.pT, p_neighbor=self.neighbor, notiming=True)
        self.sph.updateSoundSpeed(math.sqrt(2.0*math.fabs(-9.8 if self.grav == 0.0 else self.grav)*0.55*self.res/self.eta), notiming=True)
        self.pD.setConst(s=self.sph.density, notiming=True)
        gridParticleIndex(parts=self.pp, indexSys=self.gIdxSys, flags=self.gFlags, index=self.gIdx, counter=self.gCnt, notiming=True)
        self.neighbor.update(pts=self.pp, indexSys=self.gIdxSys, index=self.gIdx, radius=self.kern.supportRadius(), notiming=True)

    def init_fluid(self, init_phi):
        self.pp.clear()

        begin = self.pp.pySize()
        sampleFlagsWithParticles(flags=self.gFlags, parts=self.pp, discretization=self.sres, randomness=0, ftype=2)
        end = self.pp.pySize()

        self.pT.setConstRange(s=2, begin=begin, end=end, notiming=True)
        obstacle_cnt = end

        self.gFlags.updateFromLevelset(init_phi)
        begin = self.pp.pySize()
        sampleLevelsetWithParticles(phi=init_phi, flags=self.gFlags, parts=self.pp, discretization=self.sres, randomness=0)
        end = self.pp.pySize()
        self.pT.setConstRange(s=1, begin=begin, end=end, notiming=True)

        print("obstacle particle count: %d" % obstacle_cnt)
        print("fluid particle count: %d" % (end-obstacle_cnt))

        self.init_sph()
    
    def add_fluid(self, phi):
        self.gFlags.updateFromLevelset(phi)
        begin = self.pp.pySize()
        sampleLevelsetWithParticles(phi=phi, flags=self.gFlags, parts=self.pp, discretization=self.sres, randomness=0)
        end = self.pp.pySize()
        self.pT.setConstRange(s=1, begin=begin, end=end, notiming=True)

        self.init_sph()

  
    def init_pp(self, pp, pT, pV, pP, neighbor, minN=3, seed=345245):
        self.pp.clear()
        self.pp.readParticles(pp)
        #pp.printParts()
        #self.pp.printParts()
        #self.pV.copyFrom(pV)
        self.pT.copyFrom(pT)
        #self.pP.copyFrom(pP)

        #self.pV.multConst(vec3(1/3))
        #self.pP.multConst(1/9)
        
        reduceParticlesNeighbors(self.pp, neighbor,minN,seed)

        self.init_sph()

    def apply_vel(self, vel):
        mapGridToPartsVec3(source=vel, parts=self.pp, target=self.pV)

    def apply_pres(self, pres):
        mapGridToParts(source=pres, parts=self.pp, target=self.pP)

    def update(self):
        sphComputeDensity(d=self.pD, k=self.kern, sph=self.sph, itype=self.overFld, jtype=self.overAll)
        sphComputeConstantForce(f=self.pF, v=vec3(0, self.grav*self.sph.mass, 0), sph=self.sph, itype=self.overFld, accumulate=False)
        sphComputeSurfTension(f=self.pF, k=self.kern, sph=self.sph, kappa=0.8, itype=self.overFld, jtype=self.overAll, accumulate=True)
        if(self.avis):
    	    sphComputeArtificialViscousForce(f=self.pF, v=self.pV, d=self.pD, k=self.kern, sph=self.sph, itype=self.overFld, jtype=self.overFld, accumulate=True)
        
        if self.sdt is None:
            adt = min(self.s.frameLength, self.kern.supportRadius()/self.sph.c)
            adt = self.sph.limitDtByVmax(dt=adt, h=self.kern.supportRadius(), vmax=self.pV.getMaxAbs(), a=0.4)
            self.s.adaptTimestepByDt(adt)
        else:
            self.s.adaptTimestepByDt(self.sdt)

        sphUpdateVelocity(v=self.pVtmp, vn=self.pV, f=self.pF, sph=self.sph, dt=self.s.timestep)
        sphComputeIisphDii(dii=self.pDii, d=self.pD, k=self.kern, sph=self.sph, dt=self.s.timestep, itype=self.overFld, jtype=self.overAll)

        self.pDadv.setConst(0)
        sphComputeDivergenceSimple(div=self.pDadv, v=self.pVtmp, k=self.kern, sph=self.sph, itype=self.overFld, jtype=self.overAll) # pDadv = div(v)
        self.pDadv.multConst(s=-self.s.timestep)                                                                # pDadv = - dt*div(v)
        self.pDadv.add(self.pD)                                                                                 # pDadv = pD - dt*div(v)
        self.pAii.setConst(0)
        sphComputeIisphAii(aii=self.pAii, d=self.pD, dii=self.pDii, k=self.kern, sph=self.sph, dt=self.s.timestep, itype=self.overFld, jtype=self.overAll)

        ######################################################################
        # solve pressure
        self.pP.multConst(s=0.5)         # p = 0.5*p_prev
        d_avg, iters, d_err_th = self.sph.density, 0, self.sph.density*self.sph.eta/100.0
        prev_v = 0
        while ((d_avg - self.sph.density)>d_err_th) or (iters<2):
            prev_v = d_avg - self.sph.density

            sphComputeIisphDijPj(dijpj=self.pDijPj, d=self.pD, p=self.pP, k=self.kern, sph=self.sph, dt=self.s.timestep, itype=self.overFld, jtype=self.overAll)

            self.pDtmp.setConst(0.0)
            sphComputeIisphP(p_next=self.pDtmp, p=self.pP, d_adv=self.pDadv, d=self.pD, aii=self.pAii, dii=self.pDii, dijpj=self.pDijPj, k=self.kern, sph=self.sph, dt=self.s.timestep, itype=self.overFld, jtype=self.overAll)
            self.pDtmp.clampMin(0.0)
            self.pP.copyFrom(self.pDtmp)

            self.pDtmp.setConst(0.0)
            sphComputeIisphD(d_next=self.pDtmp, d_adv=self.pDadv, d=self.pD, p=self.pP, dii=self.pDii, dijpj=self.pDijPj, k=self.kern, sph=self.sph, dt=self.s.timestep, itype=self.overFld, jtype=self.overAll)
            d_avg = self.pDtmp.sum(t=self.pT, itype=self.overFld)/cntPts(t=self.pT, itype=self.overFld)
            iters += 1
            # for the safety
            if iters>200:
                if (d_avg - self.sph.density) < prev_v + self.sph.eta:
                    break
                else:
                    print('\tFail to converge: d_avg = {} (<{}), iters = {}'.format(d_avg, d_err_th+self.sph.density, iters))
                    sys.exit(0)

        print('\td_avg = {} (<{}), iters = {}'.format(d_avg, d_err_th+self.sph.density, iters))
        ######################################################################

        sphComputePressureForce(f=self.pF, p=self.pP, d=self.pD, k=self.kern, sph=self.sph, accumulate=False)
        sphUpdateVelocity(v=self.pV, vn=self.pVtmp, f=self.pF, sph=self.sph, dt=self.s.timestep)

        sphUpdatePosition(x=self.pp, v=self.pV, sph=self.sph, dt=self.s.timestep)
        gridParticleIndex(parts=self.pp, indexSys=self.gIdxSys, flags=self.gFlags, index=self.gIdx, counter=self.gCnt)
        self.neighbor.update(pts=self.pp, indexSys=self.gIdxSys, index=self.gIdx, radius=self.kern.supportRadius())

        self.s.step()
    #def synchronize(self, pos, vel):
