#include "particle.h"
#include "randomstream.h"
#include "grid.h"
#include "levelset.h"
#include "pneighbors.h"

#include <random>
#include <algorithm>
#include <iterator>

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
	void knFillVelocityCircular(ParticleDataImpl<Vec3> &v, BasicParticleSystem &x, float magnitude, Vec3 center)
	{
		Vec3 tmp = x[idx].pos - center;

		v[idx] = magnitude * tmp/normalize(tmp);
	}

	KERNEL(bnd=1) void knCosDisplacement(Grid<Real>& displacement, const Grid<Vec3>& grid, float fac) {
		Vec3 v = grid(i,j,k);
		float l = norm(v);
		if(l > VECTOR_EPSILON){
			displacement(i,j,k) = std::cos(std::acos(v.x/l) * fac);// * std::cos(std::acos(v.z/l) * fac);
		} else {
			displacement(i,j,k) = 0.;
		}
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
	void fillVelocityCircular(ParticleDataImpl<Vec3> &v, BasicParticleSystem &x, float magnitude, Vec3 center)
	{
		knFillVelocityCircular(v, x, magnitude, center);
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


	PYTHON() void cosDisplacement(Grid<Real> &disp, const Grid<Vec3> &grid, float fac) {
		knCosDisplacement(disp, grid, fac);
	}
	/*PYTHON()
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