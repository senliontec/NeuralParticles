import numpy as np
import matplotlib.pyplot as plt
import math
import random

class ParticleGrid:
    def __init__(self, dimX, dimY, sub_dim):
        self.particles = np.empty((0, 3))
        self.cells = np.ones((dimY, dimX),dtype="float32")
        self.dimX = dimX
        self.dimY = dimY
        self.sub_dim = sub_dim**2
    
    def sample_cell(self, pos, sd):
        if pos[0] < self.dimX and pos[1] < self.dimY:
            if self.cells[int(pos[1]),int(pos[0])] > 0.0:
                for i in range(self.sub_dim):
                    r_p = pos + np.random.random((2,))
                    self.particles = np.append(self.particles, np.array([[r_p[0], r_p[1], 0]]), axis=0)
            
            self.cells[int(pos[1]),int(pos[0])] = min(sd, self.cells[int(pos[1]),int(pos[0])])

    def sample_sphere(self, center, radius):
        for x in range(int(-radius), int(radius)):
            for y in range(int(-radius), int(radius)):
                l = math.sqrt(x**2 + y**2)
                if l <= radius:
                    self.sample_cell(np.array([x,y])+center, l-radius)

    def sample_quad(self, center, a, b):
        for x in range(int(-a), int(a)):
            for y in range(int(-b), int(b)):
                self.sample_cell(np.array([x,y]+center), -min(a-abs(x),b-abs(y)))

    def sample_cos_sphere(self, center, radius, cos_cnt, cos_amp):
        for x in range(int(-radius-cos_amp), int(radius+cos_amp)):
            for y in range(int(-radius-cos_amp), int(radius+cos_amp)):
                l = math.sqrt(x**2 + y**2)
                surf = (radius+math.cos(math.acos(x/l)*cos_cnt)*cos_amp) if l > 0 else radius
                if l == 0 or l <= surf:
                    self.sample_cell(np.array([x,y])+center, l-surf)

if __name__ == '__main__':
    fac = 3
    src_grid = ParticleGrid(50, 50, 2)
    ref_grid = ParticleGrid(src_grid.dimX*fac, src_grid.dimX*fac, 2)

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
            ref_grid.sample_cos_sphere(pos*fac, r*fac, max_size-r, 3)

    plt.scatter(src_grid.particles[:,0],src_grid.particles[:,1],s=0.9)
    plt.xlim([0,src_grid.dimX])
    plt.ylim([0,src_grid.dimY])
    plt.savefig("src.png")
    plt.clf()

    plt.scatter(ref_grid.particles[:,0],ref_grid.particles[:,1],s=0.1)
    plt.xlim([0,ref_grid.dimX])
    plt.ylim([0,ref_grid.dimY])
    plt.savefig("ref.png")