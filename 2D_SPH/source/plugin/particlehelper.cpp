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
	void reduceParticlesRandom(BasicParticleSystem &x, const int factor)
	{
		RandomStream rs(23892489l);
		knReduceParticlesRandom(x, 1.0f/factor, rs);
		x.doCompress();
	}
}