#pragma once
#ifndef _PARTICLE_H_
#define _PARTICLE_H_
#include "lib/mathlib.h"
#include "lib/spectrum.h"

class Particle {
public:
	Particle() {}
	Particle(Vec3 pos) : position(pos) {}
	Particle(Vec3 pos, Spectrum col) : position(pos), color(col) {}
	~Particle() {}
	Vec3 position;
	Vec3 velocity;
	Vec3 force;
	float mass;
	float density;
	float pressure;
	Spectrum color;

private:
		

};

#endif // !_PARTICLE_H_
