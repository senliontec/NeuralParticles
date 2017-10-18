#include "particle.h"
#include "randomstream.h"
#include "grid.h"
#include "levelset.h"

namespace Manta
{
	KERNEL(pts) 
	void knReduceParticlesRandom(BasicParticleSystem &x, const float factor, RandomStream& rs)
	{
		if(rs.getFloat() > factor)
		{
			x.kill(idx);
		}
	}

	// reduce particle in a random way
	PYTHON() 
	void reduceParticlesRandom(BasicParticleSystem &x, const int factor, const int seed=23892489)
	{
		RandomStream rs((long)seed);
		knReduceParticlesRandom(x, 1.0f/factor, rs);
		x.doCompress();
	}

	PYTHON()
	void extractSurfacePatches(const LevelsetGrid& levelset, float* patches, const int maxCnt, int* patchCnt, const float tol = 0.1f)
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
	}
}