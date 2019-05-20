#include "particle.h"
#include "randomstream.h"
#include "grid.h"
#include "levelset.h"
#include "pneighbors.h"

#include <random>
#include <algorithm>
#include <iterator>
#include <unordered_set>

#include <cmath>

namespace Manta
{
	//
	// helper
	//

	inline float poisson(int lambda, int k)
	{
		float res = pow(lambda, k) * exp(-lambda);
		std::cout << lambda << " " << k << "-> ";
		while(k > 0)
		{
			res /= k;
			k--;
		}
		std::cout << res << std::endl;
		return res;
	}

	struct Vec3iHash {
	public:
		size_t operator()(const Vec3i& v) const {
			return std::hash<std::string>()(v.toString());
		}
	};	


	//
	// kernels
	//

	KERNEL(pts) 
	void knReduceParticlesRandom(BasicParticleSystem &x, const float factor, RandomStream& rs)
	{
		if(rs.getFloat() > factor)
		{
			x.kill(idx);
		}
	}
	
	KERNEL(pts) 
	void knReduceParticlesPoisson(BasicParticleSystem &x, const ParticleNeighbors &n, const int factor, RandomStream& rs)
	{
		if(rs.getFloat() > poisson(factor, n.size(idx)))
		{
			x.kill(idx);
		}
	}

	KERNEL(pts) 
	void knReduceParticlesDense(BasicParticleSystem &x, const ParticleDataImpl<Real> &d, const float factor, RandomStream& rs)
	{
		if(rs.getFloat() > d[idx] * factor)
		{
			x.kill(idx);
		}
	}

	KERNEL(pts) 
	void knMaskParticles(BasicParticleSystem &x, const Grid<Real>& mask)
	{
		if(mask.getInterpolated(x[idx].pos) >= 0.)
		{
			x.kill(idx);
		}
	}

	KERNEL(pts)
	void knFillVelocityCircular(ParticleDataImpl<Vec3> &v, BasicParticleSystem &x, float magnitude, Vec3 center)
	{
		Vec3 tmp = x[idx].pos - center;
		v[idx] = magnitude * getNormalized(tmp);
	}

	KERNEL(bnd=1)
	void knCircularVelocityGrid(Grid<Vec3>& vel, float magnitude, Vec3 center)
	{
		Vec3 tmp = Vec3(i,j,k) - center;
		vel(i,j,k) = magnitude * getNormalized(tmp);
	}

	KERNEL(bnd=1) 
	void knCosDisplacement(Grid<Real>& displacement, const Grid<Vec3>& grid, float fac) {
		Vec3 v = getNormalized(grid(i,j,k));
		displacement(i,j,k) = std::cos(std::acos(v.x) * fac) * std::cos(std::acos(v.y) * fac);
		if(displacement.is3D()) displacement(i,j,k) *= std::cos(std::acos(v.z) * fac);
	}


	//
	// python functions
	//

	// reduce particle in a random way
	PYTHON() 
	void reduceParticlesRandom(BasicParticleSystem &x, const int factor, const int seed=23892489)
	{
		RandomStream rs((long)seed);
		knReduceParticlesRandom(x, 1.0f/factor, rs);
		x.doCompress();
	}

	PYTHON()
	void reduceParticlesPoisson(BasicParticleSystem &x, const ParticleNeighbors &n, const int factor, const int seed=23892489)
	{
		RandomStream rs((long)seed);
		//reduceParticlesPoisson(x, n, factor, seed);
		//x.doCompress();
		for(int idx = 0; idx < x.size(); idx++)
		{
			int cnt = 0;
			for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) 
			{
				if(x.isActive(n.neighborIdx(it)))
				{
					cnt++;
				}
			}
			if(rs.getFloat() > poisson(factor, cnt))
			{
				x.kill(idx);
			}
		}
		x.doCompress();
	}

	PYTHON()
	void reduceParticles(BasicParticleSystem &x, const float radius, const int seed=232345234)
	{
		Vec3i gSize = x.getParent()->getGridSize();

		std::unordered_set<Vec3i, Vec3iHash> used_cells = std::unordered_set<Vec3i, Vec3iHash>();
		gSize.x = int(gSize.x/radius);
		gSize.y = int(gSize.y/radius);
		gSize.z = int(gSize.z/radius);

		std::vector<int> idx = std::vector<int>(x.size());

		for(int i = 0; i < x.size(); i++)
		{
			idx[i] = i;
		}

		std::random_device rd;
		std::mt19937 g(rd());
		g.seed(seed);
	
		std::shuffle(idx.begin(), idx.end(), g);

		for(const auto i : idx)
		{
			if(x.isActive(i))
			{
				Vec3i ci = Vec3i(x[i].pos.x/radius, x[i].pos.y/radius, x[i].pos.z/radius);
				if(!used_cells.insert(ci).second)
				{
					x.kill(i);
				}
			}
		}
		x.doCompress();
	}

	PYTHON()
	void reduceParticlesNeighbors(BasicParticleSystem &x, const ParticleNeighbors &n, const int minN=3, const int seed=23892489)
	{
		std::vector<int> v = std::vector<int>(x.size());

		for(int i = 0; i < x.size(); i++)
		{
			v[i] = i;
		}
 
		std::random_device rd;
		std::mt19937 g(rd());
		g.seed(seed);
	
		std::shuffle(v.begin(), v.end(), g);

		for(const auto i : v)
		{
			if(x.isActive(i))
			{
				if(n.size(i) < minN){
					x.kill(i);
				}
				for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(i); it!=n.end(i); ++it) 
				{
					const int idx = n.neighborIdx(it);
					if(idx != i) x.kill(idx);
				}
			}
		}
		x.doCompress();
	}

	PYTHON()
	void reduceParticlesNeighborsDens(BasicParticleSystem &x, const ParticleNeighbors &n, const ParticleDataImpl<Real> &d, float r, const float factor=1.0, const int minN=3, const int seed=23892489)
	{
		std::vector<int> v = std::vector<int>(x.size());

		for(int i = 0; i < x.size(); i++)
		{
			v[i] = i;
		}
 
		std::random_device rd;
		std::mt19937 g(rd());
		g.seed(seed);
	
		std::shuffle(v.begin(), v.end(), g);

		r *= d.sumMagnitude()/d.size();//d.getMinValue();

		for(const auto i : v)
		{
			if(x.isActive(i))
			{
				if(n.size(i) < minN){
					x.kill(i);
				}
				for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(i); it!=n.end(i); ++it) 
				{
					const int idx = n.neighborIdx(it);
					if(idx != i && n.length(it) < pow(r/d[i], factor)) 
						x.kill(idx);
				}
			}
		}
		x.doCompress();
	}
		
	PYTHON()
	void reduceParticlesDens(BasicParticleSystem &x, const ParticleDataImpl<Real> &d, const float factor, const int seed=23892489)
	{		
		float avg = d.sumMagnitude()/d.size();
		RandomStream rs((long)seed);
		knReduceParticlesDense(x, d, 1.0f / (avg * factor), rs);
		x.doCompress();
	}
		
	PYTHON()
	void maskParticles(BasicParticleSystem &x, const Grid<Real>& mask)
	{		
		knMaskParticles(x, mask);
		x.doCompress();
	}

	PYTHON()
	void fillVelocityCircular(ParticleDataImpl<Vec3> &v, BasicParticleSystem &x, float magnitude, Vec3 center)
	{
		knFillVelocityCircular(v, x, magnitude, center);
	}

	PYTHON()
	void circularVelGrid(Grid<Vec3>& vel, float magnitude, Vec3 center)
	{
		knCircularVelocityGrid(vel, magnitude, center);
	}

	PYTHON() 
	void placeGrid2d(Grid<Real>& src, Grid<Real>& dst, int dstz) 
	{
		FOR_IJK(src) 
		{
			if(!dst.isInBounds(Vec3i(i,j,dstz))) continue;
			dst(i,j,dstz) = src(i,j,0);
		}
	}


	PYTHON() 
	void cosDisplacement(Grid<Real> &disp, const Grid<Vec3> &grid, float fac) {
		knCosDisplacement(disp, grid, fac);
	}

	/*PYTHON() 
	void getPointsOnSurface(std::vector<IndexInt> &onSurface, const BasicParticleSystem &x, const Grid<Real> &g, float thres) {
		if(std::abs(g.getInterpolated(x[idx].pos)) < thres) {
			onSurface.push_back(idx);
		}
	}

	PYTHON() 
	void getParticlePatches(PyArrayContainer patches, const BasicParticleSystem &x, const ParticleNeighbors &n, const Grid<Real> &g, float thres, int part_cnt, float pad_val) {
		std::vector<IndexInt> patchCenters;
		getPointsOnSurface(patchCenters, x, g, thres);
 
		std::random_device rd;
		std::mt19937 gen(rd());
		gen.seed(23456);
	
		std::shuffle(patchCenters.begin(), patchCenters.end(), gen);

		int j = 0;
		for(const auto idx : patchCenters)
		{
			int i = 0;
			for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(i) && i < part_cnt; ++it, ++i) 
			{
				patches[j * part_cnt + i] = x[n.neighborIdx(it)].pos;
			}		
			for(; i < part_cnt; i++) {
				patches[j * part_cnt + i] = pad_val;
			}
			j++;
		}
	}
	PYTHON()
	void extractSurfacePatches(const LevelsetGrid& phi0, const LevelsetGrid& phi1, float* patches, const int maxCnt, int* patchCnt, const float tol = 0.1f)
	{
		// TODO: use second levelset grid as reference + respect also scale!
		for(int z = 0; z < levelset.getSizeZ(); z++)
		{
			for(int y = 0; y < levelset.getSizeY(); y++)
			{
				for(int x = 0; x < levelset.getSizeX(); x++)
				{
					if(abs(levelset.get(x,y,z)) < tol)
					{
						for(int k = -2; k < 3; k++)
						{
							if(!levelset.is3D()) k = 0;
							for(int j = -2; j < 3; j++)
							{
								for(int i = -2; i < 3; i++)
								{
									if(*patchCnt < maxCnt)
									{
										patches[*patchCnt++] = levelset.get(x+i,y+j,z+k);
									}
									else
									{
										std::cout << "patches array too small!" << std::endl;
									}
								}
							}
							if(!levelset.is3D()) break;
						}
					}
				}
			}
		}
	}*/
}