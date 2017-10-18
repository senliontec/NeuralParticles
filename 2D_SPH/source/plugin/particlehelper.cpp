#include "particle.h"
#include "randomstream.h"

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
	void extractSurfacePatches()
	{
		
	}
}