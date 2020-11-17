#pragma once
#ifndef _FLUIDSIM_H_
#define _FLUIDSIM_H_
#include "lib/mathlib.h"
#include "Particle.h"

class FluidSim {
public:
	FluidSim(Vec3 min, Vec3 max, size_t num_particles) :
		min(min),
		max(max),
		num_particles(num_particles) { }
	~FluidSim() {}

	Vec3 min;  // Defines min point of simulation box
	Vec3 max;  // Defines max point of simulation box
	std::vector<Particle> particles; // list of all particles
	size_t num_particles;

	void update();
private:

};

#endif // !_FLUIDSIM_H_
