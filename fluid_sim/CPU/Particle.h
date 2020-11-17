#pragma once
#ifndef _PARTICLE_H_
#define _PARTICLE_H_
#include "lib/mathlib.h"
#include "lib/spectrum.h"
#include <GL/glut.h>
#define SPHERE_RINGS 12
#define SPHERE_SECTORS 24
#define DEFAULT_MASS (1e-5)
#define DEFAULT_DENSITY 1.f
#define DEFAULT_PRESSURE 0.f
#define DEFAULT_RADIUS (1e-6)

class Particle {
public:
	Particle(Vec3 pos, float rad) : 
	position(pos),
	velocity(Vec3(0.f)),
	force(Vec3(0.f)),
	mass(DEFAULT_MASS),
	density(DEFAULT_DENSITY),
	pressure(DEFAULT_PRESSURE),
	radius(rad),
	color(Spectrum(1.f, 0.f, 0.f)) {
        float const R = 1. / (float)(SPHERE_RINGS - 1);
        float const S = 1. / (float)(SPHERE_SECTORS - 1);
        int r, s;

        vertices.resize(SPHERE_RINGS * SPHERE_SECTORS * 3);
        normals.resize(SPHERE_RINGS * SPHERE_SECTORS * 3);
        texcoords.resize(SPHERE_RINGS * SPHERE_SECTORS * 2);
        std::vector<GLfloat>::iterator v = vertices.begin();
        std::vector<GLfloat>::iterator n = normals.begin();
        std::vector<GLfloat>::iterator t = texcoords.begin();
        for (r = 0; r < SPHERE_RINGS; r++) {
            for (s = 0; s < SPHERE_SECTORS; s++) {
                float const y = sin(-PI_F/2 + PI_F * r * R);
                float const x = cos(2 * PI_F * s * S) * sin(PI_F * r * R);
                float const z = sin(2 * PI_F * s * S) * sin(PI_F * r * R);

                *t++ = s * S;
                *t++ = r * R;

                *v++ = x * radius;
                *v++ = y * radius;
                *v++ = z * radius;

                *n++ = x;
                *n++ = y;
                *n++ = z;
            }
        }

        indices.resize(SPHERE_RINGS * SPHERE_SECTORS * 4);
        std::vector<GLushort>::iterator i = indices.begin();
        for (r = 0; r < SPHERE_RINGS - 1; r++) {
            for (s = 0; s < SPHERE_SECTORS - 1; s++) {
                *i++ = r * SPHERE_SECTORS + s;
                *i++ = r * SPHERE_SECTORS + (s + 1);
                *i++ = (r + 1) * SPHERE_SECTORS + (s + 1);
                *i++ = (r + 1) * SPHERE_SECTORS + s;
            }
        }
	}

	~Particle() {}

	Vec3 position;
	Vec3 velocity;
	Vec3 force;
	float mass;
	float density;
	float pressure;
	float radius;
	Spectrum color;
	void draw();
private:
	std::vector<GLfloat> vertices;
	std::vector<GLfloat> normals;
	std::vector<GLfloat> texcoords;
	std::vector<GLushort> indices;
};

#endif // !_PARTICLE_H_
