import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy
from scipy import interpolate

import warnings


def interpol_grid(grid):
    x_v = np.arange(0.5, grid.shape[2]+0.5)
    y_v = np.arange(0.5, grid.shape[1]+0.5)
    if grid.shape[0] == 1:
        z_v = np.array([0.0,1.0])
        return interpolate.RegularGridInterpolator((x_v, y_v, z_v), np.transpose(np.concatenate([grid,grid]),(2,1,0,3)), bounds_error=False, fill_value=1.0)
    else:
        z_v = np.arange(0.5, grid.shape[0]+0.5)
        return interpolate.RegularGridInterpolator((x_v, y_v, z_v), np.transpose(grid,(2,1,0,3)), bounds_error=False, fill_value=1.0)

def normals(sdf):
    g = np.gradient(np.squeeze(sdf))
    if sdf.shape[0] == 1:
        g.insert(0, np.zeros_like(g[0]))
        g = np.expand_dims(g,axis=1)
    g = np.expand_dims(g,axis=-1)
    g = np.concatenate([g[2],g[1],g[0]],axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nan_to_num(g/np.linalg.norm(g,axis=-1,keepdims=True))
    return result

def curvature(sdf):
    dif = np.gradient(normals(sdf))
    return (np.linalg.norm(dif[0],axis=-1)+np.linalg.norm(dif[1],axis=-1)+np.linalg.norm(dif[2],axis=-1))/3

import time
class ParticleIdxGrid:
    def __init__(self, particles, shape):
        #self.particles = particles
        self.shape = shape
        self.grid = np.empty(shape,dtype=object)
        self.grid[:] = [[[[] for x in range(shape[2])] for y in range(shape[1])] for z in range(shape[0])]
        
        for i in range(len(particles)):
            x,y,z = particles[i].astype(dtype="int32")
            if x >= 0 and x < self.shape[2] and y >= 0 and y < self.shape[1] and z >= 0 and z < self.shape[0]:
                self.grid[z,y,x].append(i)

    def get_cell(self, cell_idx):
        x,y,z = cell_idx.astype(dtype="int32")
        return self.grid[z,y,x]
    
    def get_range(self, c, r):
        sz, sy, sx = self.shape
        x0,y0,z0 = np.clip((c-r).astype('int32'), 0, [sx,sy,sz])
        x1,y1,z1 = np.clip((c+r).astype('int32'), 0, [sx,sy,sz])

        return [v for z in self.grid[z0:z1,y0:y1,x0:x1] for y in z for x in y for v in x]


class ParticleGrid:
    def __init__(self, dimX, dimY, dimZ, sub_dim, extrapol=5):
        self.particles = np.empty((0, 3))
        self.cells = np.ones((dimZ, dimY, dimX, 1),dtype="float32")*extrapol
        self.dimX = dimX
        self.dimY = dimY
        self.dimZ = dimZ
        self.sub_dim = sub_dim**(2 if dimZ == 1 else 3)
        self.extrapol = extrapol
    
    def sample_cell(self, cell_idx):
        if np.all(cell_idx >= 0) and np.all(cell_idx < [self.dimX, self.dimY, self.dimZ]):#cell_idx[0] >= 1 and cell_idx[0] <= self.dimX-1 and cell_idx[1] >= 1 and cell_idx[1] <= self.dimY-0.5 and cell_idx[2] >= 0.5 and cell_idx[2] <= self.dimZ-0.5:
            for i in range(self.sub_dim):
                rp =  cell_idx + np.random.random((3,))#-0.5
                if self.dimZ == 1:
                    rp[2] = 0.5
                self.particles = np.append(self.particles, [rp], axis=0)

    def set_sdf(self, cell_idx, value):
        if cell_idx[0] < self.dimX and cell_idx[1] < self.dimY and cell_idx[2] < self.dimZ:
            c_x, c_y, c_z = cell_idx
            self.cells[c_z,c_y,c_x,0] = min(value, self.cells[c_z,c_y,c_x,0])

    def sample_sdf(self):
        sdf_f = interpol_grid(self.cells)
        for z in range(self.dimZ):
            for y in range(self.dimY):
                for x in range(self.dimX):
                    if self.cells[z,y,x,0] < 1.0:
                        self.sample_cell(np.array([x,y,z]))
                        del_i = []
                        for i in range(len(self.particles)-self.sub_dim, len(self.particles)):
                            p = self.particles[i]
                            if sdf_f(p) > 0.0:
                                del_i.append(i)
                        self.particles = np.delete(self.particles, del_i, axis=0)

    def sample_sphere(self, center, radius):
        if self.dimZ == 1:
            center[2] = 0.5
        for z in range(int(-radius)-self.extrapol, int(radius)+1+self.extrapol) if self.dimZ != 1 else [0]:
            for y in range(int(-radius)-self.extrapol, int(radius)+1+self.extrapol):
                for x in range(int(-radius)-self.extrapol, int(radius)+1+self.extrapol):
                    idx = np.array([x,y,z])+center.astype('int32')
                    l = np.linalg.norm(idx + 0.5 - center)
                    if l <= radius+self.extrapol:
                        self.set_sdf(idx, l-radius)

    def sample_quad(self, center, a, b, c):
        if self.dimZ == 1:
            center[2] = 0.5
        for z in range(int(-c)-self.extrapol, int(c)+1+self.extrapol) if self.dimZ != 1 else [0]:
            for y in range(int(-b)-self.extrapol, int(b)+1+self.extrapol):
                for x in range(int(-a)-self.extrapol, int(a)+1+self.extrapol):
                    idx = np.array([x,y,z])+center.astype('int32')
                    p = np.abs(idx + 0.5 - center)
                    min_v = min(a-p[0],b-p[1])
                    if self.dimZ != 1:
                        min_v = min(min_v, c-p[2])
                    self.set_sdf(idx, -min_v)

    def sample_cos_sphere(self, center, radius, cos_cnt, cos_amp):
        if self.dimZ == 1:
            center[2] = 0.5
        for z in range(int(-radius)-self.extrapol, int(radius)+1+self.extrapol) if self.dimZ != 1 else [0]:
            for y in range(int(-radius)-self.extrapol, int(radius)+1+self.extrapol):
                for x in range(int(-radius)-self.extrapol, int(radius)+1+self.extrapol):
                    idx = np.array([x,y,z])+center.astype('int32')
                    p = idx + 0.5 - center
                    l = np.linalg.norm(p)
                    surf = (radius+math.cos(math.acos(p[0]/l)*cos_cnt)*cos_amp) if l > 0 else 0
                    if l <= surf+self.extrapol:
                        self.set_sdf(idx, l-surf)

class RandomParticles:
    def __init__(self, dimX, dimY, dimZ, sub_dim, fac_d, max_size, cnt, cube_prob, cos_displace=0, extrapol=5):
        self.dimX = dimX
        self.dimY = dimY
        self.dimZ = dimZ
        self.sub_dim = sub_dim
        self.fac_d = fac_d
        self.max_size = max_size
        self.cnt = cnt
        self.cube_prob = cube_prob
        self.extrapol = extrapol
        self.cos_displace = cos_displace
    
    def gen_random(self, pos=None, cube=None, a=None):
        self.pos = np.random.random((self.cnt,3)) * [self.dimX, self.dimY, self.dimZ if self.dimZ > 1 else 0] if pos is None else pos
        self.cube = np.random.random((self.cnt,)) < self.cube_prob if cube is None else cube
        self.a = 1+np.random.random((self.cnt,3)) * [self.max_size-1, self.max_size-1, self.max_size-1 if self.dimZ > 1 else 0] if a is None else a

    def get_grid(self):
        src_grid = ParticleGrid(self.dimX, self.dimY, self.dimZ, self.sub_dim, self.extrapol)
        ref_grid = ParticleGrid(int(self.dimX*self.fac_d), int(self.dimY*self.fac_d), int(self.dimZ*self.fac_d) if self.dimZ > 1 else 1, self.sub_dim, self.extrapol)
        for i in range(self.cnt):
            if self.cube[i]:
                src_grid.sample_quad(self.pos[i], self.a[i,0], self.a[i,1], self.a[i,2])
                ref_grid.sample_quad(self.pos[i] * self.fac_d, self.a[i,0] * self.fac_d, self.a[i,1] * self.fac_d, self.a[i,2] * self.fac_d)
            else:
                src_grid.sample_sphere(self.pos[i], self.a[i,0])
                if self.cos_displace > 0:
                    ref_grid.sample_cos_sphere(self.pos[i] * self.fac_d, self.a[i,0] * self.fac_d, 6, self.cos_displace)
                else:
                    ref_grid.sample_sphere(self.pos[i] * self.fac_d, self.a[i,0] * self.fac_d)
        src_grid.sample_sdf()
        ref_grid.sample_sdf()
        return src_grid, ref_grid

if __name__ == '__main__':

    from neuralparticles.tools.plot_helpers import plot_sdf, plot_particles
    from neuralparticles.tools.data_helpers import extract_particles

    rp = RandomParticles(50,50,1,2,3,10,5,0.5,3)

    rp.gen_random()
    src_grid, ref_grid = rp.get_grid()

    plot_particles(src_grid.particles, [0,src_grid.dimX], [0,src_grid.dimY], 0.1, "src.pdf")
    plot_particles(ref_grid.particles, [0,ref_grid.dimX], [0,ref_grid.dimY], 0.1, "ref.pdf")
    plot_particles(src_grid.particles, [0,src_grid.dimX], [0,src_grid.dimY], 0.1, "comp.pdf", ref_grid.particles/3)

    plot_sdf(src_grid.cells[0,:,:,0], [0,src_grid.dimX], [0,src_grid.dimY], "src_sdf.pdf", src=src_grid.particles, s=0.1)
    plot_sdf(ref_grid.cells[0,:,:,0], [0,ref_grid.dimX], [0,ref_grid.dimY], "ref_sdf.pdf", src=ref_grid.particles, s=0.1)

    def in_surface(sdf):
        return np.where(abs(sdf) < 1.0)

    def in_bound(pos, bnd_min, bnd_max):
        return np.where(np.all([np.all(bnd_min<=pos,axis=-1),np.all(pos<=bnd_max,axis=-1)],axis=0))
    
    patch_size = 5

    sdf_f = interpol_grid(src_grid.cells)
        
    particle_data_bound = src_grid.particles[in_bound(src_grid.particles[:,:2], patch_size/2, src_grid.dimX-(patch_size/2))]

    positions = particle_data_bound[in_surface(np.array([sdf_f(p) for p in particle_data_bound]))[0]]
    positions = positions[np.random.randint(len(positions), size=10)]

    i = 0
    for pos in positions:
        par = (extract_particles(src_grid.particles, pos, 1000, patch_size/2)[0] + 1) * patch_size/2
        tmp = np.array([[sdf_f(pos-patch_size/2+0.5+np.array([x,y,patch_size/2]))[0,0] for x in range(patch_size)] for y in range(patch_size)])
        plot_sdf(tmp, [0,patch_size], [0,patch_size], "src_sdf_patch_%02d.pdf"%i, src=par, s=5)
        i+=1

    i = 0
    sdf_f = interpol_grid(ref_grid.cells)
    patch_size *= 3
    for pos in positions * [3,3,1]:
        par = (extract_particles(ref_grid.particles, pos, 1000, patch_size/2)[0] + 1) * patch_size/2
        tmp = np.array([[sdf_f(pos-patch_size/2+0.5+np.array([x,y,patch_size/2]))[0,0] for x in range(patch_size)] for y in range(patch_size)])
        plot_sdf(tmp, [0,patch_size], [0,patch_size], "ref_sdf_patch_%02d.pdf"%i, src=par, s=5)
        i+=1

