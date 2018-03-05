import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy
from scipy import interpolate

import sys
sys.path.append("manta/scenes/tools")
from helpers import *

def sdf_func(sdf):
    x_v = np.arange(0.5, sdf.shape[1]+0.5)
    y_v = np.arange(0.5, sdf.shape[0]+0.5)
    return lambda x: interpolate.interp2d(x_v, y_v, sdf)(x[0],x[1])

class ParticleGrid:
    def __init__(self, dimX, dimY, sub_dim, extrapol=5):
        self.particles = np.empty((0, 3))
        self.cells = np.ones((dimY, dimX),dtype="float32")*extrapol
        self.dimX = dimX
        self.dimY = dimY
        self.sub_dim = sub_dim**2
        self.extrapol = extrapol
    
    def sample_cell(self, pos):
        if pos[0] >= 0.5 and pos[0] <= self.dimX-0.5 and pos[1] >= 0.5 and pos[1] <= self.dimY-0.5:
            for i in range(self.sub_dim):
                r_p = pos + np.random.random((2,))-0.5
                self.particles = np.append(self.particles, np.array([[r_p[0], r_p[1], 0]]), axis=0)

    def set_sdf(self, pos, value):
        if pos[0] < self.dimX and pos[1] < self.dimY:
            c_x = int(pos[0])
            c_y = int(pos[1])
            self.cells[c_y,c_x] = min(value, self.cells[c_y,c_x])

    def sample_sdf(self):
        sdf_f = sdf_func(self.cells)
        for y in range(self.dimY):
            for x in range(self.dimX):
                if self.cells[y,x] <= 1.0:
                    self.sample_cell(np.array([x,y])+0.5)
                    del_i = []
                    for i in range(len(self.particles)-self.sub_dim, len(self.particles)):
                        p = self.particles[i,:2]
                        if sdf_f(p) > 0.0:
                            del_i.append(i)
                            #self.particles[i,:2] = p * radius/dis + center
                    self.particles = np.delete(self.particles, del_i, axis=0)

    def sample_sphere(self, center, radius):
        for y in range(int(-radius)-1-self.extrapol, int(radius)+1+self.extrapol):
            for x in range(int(-radius)-1-self.extrapol, int(radius)+1+self.extrapol):
                l = math.sqrt(x**2 + y**2)
                if l <= radius+self.extrapol:
                    self.set_sdf(np.array([x,y])+center, l-radius)

    def sample_quad(self, center, a, b):
        for y in range(int(-b)-1-self.extrapol, int(b)+1+self.extrapol):
            for x in range(int(-a)-1-self.extrapol, int(a)+1+self.extrapol):
                self.set_sdf(np.array([x,y])+center, -min(a-abs(x),b-abs(y)))

    def sample_cos_sphere(self, center, radius, cos_cnt, cos_amp):
        for y in range(int(-radius-cos_amp)-1-self.extrapol, int(radius+cos_amp)+1+self.extrapol):
            for x in range(int(-radius-cos_amp)-1-self.extrapol, int(radius+cos_amp)+1+self.extrapol):
                l = math.sqrt(x**2 + y**2)
                surf = (radius+math.cos(math.acos(x/l)*cos_cnt)*cos_amp) if l > 0 else radius
                if l == 0 or l <= surf:
                    self.set_sdf(np.array([x,y])+center, l-radius)

if __name__ == '__main__':
    fac = 3
    src_grid = ParticleGrid(50, 50, 2)
    ref_grid = ParticleGrid(src_grid.dimX*fac, src_grid.dimY*fac, 2)

    max_size = 10
    for i in range(10):
        pos = np.random.random((2,))*np.array([src_grid.dimX, src_grid.dimY])
        if random.random() < 0.2:
            a, b = random.random()*max_size, random.random()*max_size
            src_grid.sample_quad(pos, a, b)
            ref_grid.sample_quad(pos*fac, a*fac, b*fac)
        else:
            r = random.random()*max_size
            src_grid.sample_sphere(pos, r)
            #ref_grid.sample_cos_sphere(pos*fac, r*fac, max_size-r, 3)
            ref_grid.sample_sphere(pos*fac, r*fac)
    src_grid.sample_sdf()
    ref_grid.sample_sdf()

    plot_particles(src_grid.particles, [0,src_grid.dimX], [0,src_grid.dimY], 0.1, "src.png")
    plot_particles(ref_grid.particles, [0,ref_grid.dimX], [0,ref_grid.dimY], 0.1, "ref.png")

    plot_sdf(src_grid.cells, [0,src_grid.dimX], [0,src_grid.dimY], "src_sdf.png", src=src_grid.particles, s=0.1)
    plot_sdf(ref_grid.cells, [0,ref_grid.dimX], [0,ref_grid.dimY], "ref_sdf.png", src=ref_grid.particles, s=0.1)

    def in_surface(sdf):
        return np.where(abs(sdf) < 1.0)
    
    sdf_f = sdf_func(src_grid.cells)
    positions = src_grid.particles[in_surface(np.array([sdf_f(p) for p in src_grid.particles]))[0]]
    positions = positions[np.random.randint(len(positions), size=10)]
    print(positions)

    i = 0
    patch_size = 5
    for pos in positions:
        par = (extract_particles(src_grid.particles, pos, 1000, patch_size/2)[0] + 1) * patch_size/2
        tmp = np.array([[sdf_f(pos[:2]-patch_size/2+0.5+np.array([x,y]))[0] for x in range(patch_size)] for y in range(patch_size)])
        plot_sdf(tmp, [0,patch_size], [0,patch_size], "src_sdf_patch_%02d.png"%i, src=par, s=5)
        i+=1

    i = 0
    sdf_f = sdf_func(ref_grid.cells)
    patch_size *= fac
    for pos in positions * fac:
        par = (extract_particles(ref_grid.particles, pos, 1000, patch_size/2)[0] + 1) * patch_size/2
        tmp = np.array([[sdf_f(pos[:2]-patch_size/2+0.5+np.array([x,y]))[0] for x in range(patch_size)] for y in range(patch_size)])
        plot_sdf(tmp, [0,patch_size], [0,patch_size], "ref_sdf_patch_%02d.png"%i, src=par, s=5)
        i+=1

