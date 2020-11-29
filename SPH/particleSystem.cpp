/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 // OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include "particleSystem.h"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, float3 boxDims, bool bUseOpenGL) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_hPos(0),
    m_boxDims(boxDims),
    m_timer(NULL),
    m_solverIterations(1) {
    // set simulation parameters
    m_params.particleRadius = 1.0f / 64.0f;
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;
    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.boxMin.x = -boxDims.x / 2;
    m_params.boxMin.y = -boxDims.y / 2;
    m_params.boxMin.z = -boxDims.z / 2;
    m_params.boxMax.x = boxDims.x / 2;
    m_params.boxMax.y = boxDims.y / 2;
    m_params.boxMax.z = boxDims.z / 2;

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem() {
    _finalize();
    m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size) {
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// create a color ramp
void colorRamp(float t, float* r) {
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors - 1);
    int i = (int)t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i + 1][0], u);
    r[1] = lerp(c[i][1], c[i + 1][1], u);
    r[2] = lerp(c[i][2], c[i + 1][2], u);
}

void
ParticleSystem::_initialize(int numParticles) {
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles * 4];
    memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));

    m_particles.resize(m_numParticles);

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    if (m_bUseOpenGL) {
        m_posVbo = createVBO(memSize);
    }
    else {
        checkCudaErrors(cudaMalloc((void**)&m_cudaPosVBO, memSize));
    }

    if (m_bUseOpenGL) {
        m_colorVBO = createVBO(m_numParticles * 4 * sizeof(float));

        // fill color buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
        float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float* ptr = data;

        for (uint i = 0; i < m_numParticles; i++) {
            float t = i / (float)m_numParticles;
#if 0
            * ptr++ = rand() / (float)RAND_MAX;
            *ptr++ = rand() / (float)RAND_MAX;
            *ptr++ = rand() / (float)RAND_MAX;
#else
            colorRamp(t, ptr);
            ptr += 3;
#endif
            * ptr++ = 1.0f;
        }

        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    else {
        checkCudaErrors(cudaMalloc((void**)&m_cudaColorVBO, sizeof(float) * numParticles * 4));
    }

    sdkCreateTimer(&m_timer);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize() {
    assert(m_bInitialized);

    delete[] m_hPos;
    m_particles.clear();

    if (m_bUseOpenGL) {
        glDeleteBuffers(1, (const GLuint*)&m_posVbo);
        glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
    }
    else {
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
    }
}

// https://matthias-research.github.io/pages/publications/sca03.pdf
// https://lucasschuermann.com/writing/implementing-sph-in-2d
float ParticleSystem::pressure_ideal_gas(const Particle& p) {
    return GAS_CONSTANT* (p.density - REST_DENS);
}

// https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf
// https://en.wikipedia.org/wiki/Tait_equation
float ParticleSystem::pressure_tait_eq(const Particle& p) {
    const float cs = 88.5f;
    const float gamma = 7.f;
    const float B = (REST_DENS * cs * cs) / gamma;
    float quot = p.density / REST_DENS;
    float quot_exp_gamma = quot * quot * quot * quot * quot * quot * quot;
    return B * ((quot_exp_gamma) - 1.f);
}

//float ParticleSystem::guass_kernel(float3 rij, float h) {
//    float sigma = 1.f / (std::pow(PI_F, 1.5f) * h * h * h);
//    float q = norm(rij) / h;
//    if (q < 3.0f) {
//        return sigma * std::expf(-q * q);
//    }
//    else {
//        return 0.f;
//    }
//}
//
//float3 ParticleSystem::guass_kernel_gradient(float3 rij, float h) {
//    float3 grad = unit(rij);
//    float n = norm(rij);
//    float sigma = 1.f / (std::pow(PI_F, 1.5f) * h * h * h);
//    float q = n / h;
//    if (n > 1e-20f && q < 3.0f) {
//        float dq = -2.0f * q * sigma * std::expf(-q * q);
//        grad.x *= dq;
//        grad.y *= dq;
//        grad.z *= dq;
//        /*if (grad.x != grad.x || grad.y != grad.y || grad.z != grad.z) {
//            float3 u = unit(rij);
//            printf("grad <%f, %f, %f>, dq %f, q %f, sigma %f, unit <%f, %f, %f>\n", grad.x, grad.y, grad.z, dq, q, sigma, u.x, u.y, u.z);
//        }*/
//        return grad;
//    }
//    else {
//        return { 0.f,0.f,0.f };
//    }
//}

void ParticleSystem::computeDensities() {
    const float POLY6 = 315.f / (65.f * PI_F * std::pow(m_H, 9.f));

    for (Particle& pi : m_particles) {
        pi.density = 0.f;
        for (Particle& pj : m_particles) {
            Vector3f rij = pi.position - pj.position;
            float r2 = rij.squaredNorm();
            if (r2 < HSQ) {
                pi.density += pj.mass * POLY6 * std::pow(HSQ - r2, 3.f);
            }
        }
        pi.pressure = std::max(0.f, pressure_ideal_gas(pi));
    }
}

void ParticleSystem::computeForces() {
    const float SPIKY_GRAD = -45.f / (PI_F * std::pow(m_H, 6.f));
    const float VISC_LAP = 45.f / (PI_F * std::pow(m_H, 6.f));
    for (Particle& pi : m_particles) {
        Vector3f fpress = { 0.f, 0.f, 0.f };
        Vector3f fvisc = { 0.f, 0.f, 0.f };
        Vector3f fgrav = { 0.f, GRAVITY * 11000 * pi.density, 0.f };
        for (Particle& pj : m_particles) {
            if (pi.index == pj.index) continue;
            Vector3f rij = pi.position - pj.position;
            float r = rij.norm();
            if (r < m_H) {
                fpress += -rij.normalized() * pj.mass * (pi.pressure + pj.pressure) / (2.f * pj.density) * SPIKY_GRAD * std::pow(m_H - r, 2.f);
                fvisc += VISC * pj.mass * (pj.velocity - pi.velocity) / pj.density * VISC_LAP * (m_H - r);
            }
        }
        pi.force = fpress + fvisc + fgrav;
    }
}

void ParticleSystem::integrate(float deltaTime) {
    for (Particle& p : m_particles) {
        Vector3f accel = p.force / p.density;
        p.velocity += deltaTime * accel;
        p.position += deltaTime * p.velocity;

        // bounds check in X
        if (p.position.x() - EPS_F < m_params.boxMin.x) {
            p.position.x() = m_params.boxMin.x + EPS_F;
            p.velocity.x() *= -.75f;  // reverse direction
        }
        if (p.position.x() + EPS_F > m_params.boxMax.x) {
            p.position.x() = m_params.boxMax.x - EPS_F;
            p.velocity.x() *= -.75f;  // reverse direction
        }

        // bounds check in Y
        if (p.position.y() - EPS_F < m_params.boxMin.y) {
            p.position.y() = m_params.boxMin.y + EPS_F;
            p.velocity.y() *= -.75f;  // reverse direction
        }
        if (p.position.y() + EPS_F > m_params.boxMax.y) {
            p.position.y() = m_params.boxMax.y - EPS_F;
            p.velocity.y() *= -.75f;  // reverse direction
        }

        // bounds check in Z
        if (p.position.z() - EPS_F < m_params.boxMin.z) {
            p.position.z() = m_params.boxMin.z + EPS_F;
            p.velocity.z() *= -.75f;  // reverse direction
        }
        if (p.position.z() + EPS_F > m_params.boxMax.z) {
            p.position.z() = m_params.boxMax.z - EPS_F;
            p.velocity.z() *= -.75f;  // reverse direction
        }

        m_hPos[p.index * 4 + 0] = p.position.x();
        m_hPos[p.index * 4 + 1] = p.position.y();
        m_hPos[p.index * 4 + 2] = p.position.z();
        m_hPos[p.index * 4 + 3] = 1.0f;
    }
}

unsigned long long ParticleSystem::get_Z_index(Particle p) {
    // find the section of the grid the particle is in
    float posX = p.position.x() - m_params.boxMin.x;
    float posY = p.position.y() - m_params.boxMin.y;
    float posZ = p.position.z() - m_params.boxMin.z;
    uint x_grid = floor((posX / m_boxDims.x) * m_z_grid_dim);
    uint y_grid = floor((posY / m_boxDims.y) * m_z_grid_dim);
    uint z_grid = floor((posZ / m_boxDims.z) * m_z_grid_dim);
    
    // interleave the grid indices to find the final z-index
    // x bits
    x_grid = (x_grid | (x_grid << 16)) & 0x030000FF;
    x_grid = (x_grid | (x_grid << 8)) & 0x0300F00F;
    x_grid = (x_grid | (x_grid << 4)) & 0x030C30C3;
    x_grid = (x_grid | (x_grid << 2)) & 0x09249249;
    // y bits
    y_grid = (y_grid | (y_grid << 16)) & 0x030000FF;
    y_grid = (y_grid | (y_grid << 8)) & 0x0300F00F;
    y_grid = (y_grid | (y_grid << 4)) & 0x030C30C3;
    y_grid = (y_grid | (y_grid << 2)) & 0x09249249;
    // z bits
    z_grid = (z_grid | (z_grid << 16)) & 0x030000FF;
    z_grid = (z_grid | (z_grid << 8)) & 0x0300F00F;
    z_grid = (z_grid | (z_grid << 4)) & 0x030C30C3;
    z_grid = (z_grid | (z_grid << 2)) & 0x09249249;

    return x_grid | (y_grid << 1) | (z_grid << 2);
}


bool cmpParticles(Particle pi, Particle pj) {
    return (pi.zindex < pj.zindex);
}

void ParticleSystem::constructGridArray() {
    printf("Grid dim: %d\n", m_z_grid_dim);
    for (Particle& p : m_particles) {
        p.zindex = get_Z_index(p);
        printf("Particle %d is in z index %d\n", p.index, p.zindex);

    }
    // sort the particles vector according to the z index
    std::sort(m_particles.begin(), m_particles.end(), cmpParticles);
}


// step the simulation
void
ParticleSystem::update(float deltaTime) {
    assert(m_bInitialized);

    for (int iter = 0; iter < m_solverIterations; iter++) {
        // place particles into their grid indices and sort particles according to cell indices
        constructGridArray();

        // N^2 algorithm for calculating density for each particle, computes pressure as well
        computeDensities();

        // computes pressure and gravity force contribution on each particle
        computeForces();

        // integrates velocity and position based on forces
        integrate(deltaTime);
        /*for (uint i = 0; i < m_particles.size(); i++) {
            Particle& p = m_particles.at(i);
            printf("particle %u mass %f, density %f, pressure %f, position <%f, %f, %f>, force <%f, %f, %f>\n",
                p.index, p.mass, p.density, p.pressure, p.position.x, p.position.y, p.position.z, p.force.x, p.force.y, p.force.z);
        }*/
    }

    // update the vertex buffer object
    updatePosVBO();
}

void
ParticleSystem::dumpParticles(uint start, uint count) {
    // debug

    for (uint i = start; i < start + count; i++) {
        //        printf("%d: ", i);
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i * 4 + 0], m_hPos[i * 4 + 1], m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
    }
}

void ParticleSystem::updatePosVBO() {
    if (m_bUseOpenGL) {
        glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, m_numParticles * 4 * sizeof(float), m_hPos);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    else {
    }
}

inline float frand() {
    return rand() / (float)RAND_MAX;
}

void
ParticleSystem::initGrid(uint* size, float spacing, float jitter, uint numParticles) {
    srand(1973);

    for (uint z = 0; z < size[2]; z++) {
        for (uint y = 0; y < size[1]; y++) {
            for (uint x = 0; x < size[0]; x++) {
                uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

                if (i < numParticles) {
                    float w, h, d;
                    w = m_boxDims.x;
                    h = m_boxDims.y;
                    d = m_boxDims.z;
                    Particle& p = m_particles.at(i);
                    p.index = i;
                    p.position = { (spacing * x) + m_params.particleRadius + m_params.boxMin.x + (w * frand() - w / 2) * jitter ,
                                   (spacing * y) + m_params.particleRadius + m_params.boxMin.y + (h * frand() - h / 2) * jitter ,
                                   (spacing * z) + m_params.particleRadius + m_params.boxMin.z + (d * frand() - d / 2) * jitter };
                    p.velocity = { 0.f,0.f,0.f };
                    p.force = { 0.f,0.f,0.f };
                    p.mass = MASS;
                    p.density = 0.f;
                    p.pressure = 0.f;
                    p.radius = m_params.particleRadius;
                    m_hPos[p.index * 4 + 0] = p.position.x();
                    m_hPos[p.index * 4 + 1] = p.position.y();
                    m_hPos[p.index * 4 + 2] = p.position.z();
                    m_hPos[p.index * 4 + 3] = 1.0f;
                }
            }
        }
    }
    updatePosVBO();
}

void
ParticleSystem::reset(ParticleConfig config) {
    switch (config) {
    default:
    case CONFIG_RANDOM:
    {
        int p = 0, v = 0;

        for (uint i = 0; i < m_numParticles; i++) {
            float w, h, d;
            w = m_boxDims.x;
            h = m_boxDims.y;
            d = m_boxDims.z;
            Particle& p = m_particles.at(i);
            p.index = i;
            p.position = { w * frand() - w / 2,
                           h * frand() - h / 2,
                           d * frand() - d / 2 };
            p.velocity = { 0.f,0.f,0.f };
            p.force = { 0.f,0.f,0.f };
            p.mass = MASS;
            p.density = 0.f;
            p.pressure = 0.f;
            p.radius = m_params.particleRadius;
            m_hPos[p.index * 4 + 0] = p.position.x();
            m_hPos[p.index * 4 + 1] = p.position.y();
            m_hPos[p.index * 4 + 2] = p.position.z();
            m_hPos[p.index * 4 + 3] = 1.f;
        }
    }
    break;

    case CONFIG_GRID:
    {
        float jitter = m_params.particleRadius * 0.01f;
        uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
        uint gridSize[3];
        gridSize[0] = gridSize[1] = gridSize[2] = s;
        initGrid(gridSize, m_params.particleRadius * 2.0f, jitter, m_numParticles);
    }
    break;
    }
    updatePosVBO();
}

void
ParticleSystem::addSphere(int start, float* pos, float* vel, int r, float spacing) {
    uint index = start;
    float w, h, d;
    w = m_boxDims.x;
    h = m_boxDims.y;
    d = m_boxDims.z;
    std::sort(m_particles.begin(), m_particles.end(),
        [](const Particle e0, const Particle e1) {
            return e0.index < e1.index;
        });

    for (int z = -r; z <= r; z++) {
        for (int y = -r; y <= r; y++) {
            for (int x = -r; x <= r; x++) {
                float dx = x * spacing;
                float dy = y * spacing;
                float dz = z * spacing;
                float l = sqrtf(dx * dx + dy * dy + dz * dz);
                float jitter = m_params.particleRadius * 0.01f;

                if ((l <= m_params.particleRadius * 2.0f * r) && (index < m_numParticles)) {
                    Particle& p = m_particles.at(index);
                    p.position = { pos[0] + dx + (w * frand() - w / 2) * jitter,
                                   pos[1] + dy + (h * frand() - h / 2) * jitter,
                                   pos[2] + dz + (d * frand() - d / 2) * jitter};
                    m_hPos[p.index * 4 + 0] = p.position.x();
                    m_hPos[p.index * 4 + 1] = p.position.y();
                    m_hPos[p.index * 4 + 2] = p.position.z();
                    m_hPos[p.index * 4 + 3] = 1.f;
                    index++;

                }
            }
        }
    }
    updatePosVBO();
}
