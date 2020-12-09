// implementation of kernels

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "particleSystem.cuh"

__device__ void computePressureIdeal(Particle* p) {
	p->pressure = fmaxf(0.f, GAS_CONSTANT * (p->density - REST_DENS));
}

__device__ void computePressureTait(Particle* p) {
	const float cs = 88.5f;
	const float gamma = 7.f;
	const float B = (REST_DENS * cs * cs) / gamma;
	float quot = p->density / REST_DENS;
	float quot_exp_gamma = quot * quot * quot * quot * quot * quot * quot;
	p->pressure = B * ((quot_exp_gamma)-1.f);
}

__device__ void computeDensity(Particle* pi, const Particle* pj) {
	const float POLY6 = 315.f / (65.f * PI_F * powf(m_H, 9.f));
	Vector3f rij = pi->position - pj->position;
	float r2 = rij.squaredNorm();
	if (r2 < HSQ) {
		pi->density += pj->mass * POLY6 * powf(HSQ - r2, 3.f);
	}
}

__device__ void computeForce(Particle* pi, const Particle* pj) {
	const float SPIKY_GRAD = -45.f / (PI_F * powf(m_H, 6.f));
	const float VISC_LAP = 45.f / (PI_F * powf(m_H, 6.f));
	Vector3f rij = pi->position - pj->position;
	float r = rij.norm();
	if (r < m_H) {
		pi->force_press += -rij.normalized() * pj->mass * (pi->pressure + pj->pressure) / (2.f * pj->density) * SPIKY_GRAD * powf(m_H - r, 2.f);
		pi->force_visc += VISC * pj->mass * (pj->velocity - pi->velocity) / pj->density * VISC_LAP * (m_H - r);
	}
}

__device__ void computeCollision(Particle* pi, const Particle* pj) {
	Vector3f vij, rij;
	float dij;
	vij = pi->velocity - pj->velocity;
	rij = pi->position - pj->position;
	dij = rij.norm();
	if (dij <= COLLISION_PARAM * 2 * pi->radius && rij.dot(vij) < 0) {
		pi->delta_velocity += (pj->mass * (1.f + RESTITUTION)) * (rij.dot(vij) / (dij * dij)) * rij;
		pi->collision_count++;
	}
}

__global__ void kernelComputeDensities() {
	return;
}

__global__ void kernelComputeForces(Particle *dev_particles, Grid_item *dev_B, Grid_item *dev_B_prime) {
	__shared__ Particle batch[GRID_COMPACT_WIDTH];
	uint blockid = blockIdx.x;
	uint particleid = dev_B_prime[blockid].start + threadIdx.x;
	Particle pi = dev_particles[particleid];
	pi.force_press = { 0.f, 0.f, 0.f };
	pi.force_visc = { 0.f, 0.f, 0.f };
	//for each neighbor block in dev_B
		// for each particle in block
			// bring in particles into batch
			// sync threads
			for (int j = 0; j < GRID_COMPACT_WIDTH; j++) {
				Particle pj = dev_particles[j];
				computeForce(&pi, &pj);
			}
		//
	//
	return;
}

__global__ void kernelComputeCollisions() {
	return;
}

extern "C" {
	void cudaInit(int argc, char** argv) {
		int devID;

		devID = findCudaDevice(argc, (const char**)argv);

		if (devID < 0) {
			printf("No CUDA devices found, exiting\n");
			exit(EXIT_SUCCESS);
		}
	}

	void allocateArray(void** devPtr, size_t size) {
		checkCudaErrors(cudaMalloc(devPtr, size));
	}

	void freeArray(void* devPtr) {
		checkCudaErrors(cudaFree(devPtr));
	}

	void threadSync() {
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void copyArrayFromDevice(void* host, const void* device, size_t size) {
		checkCudaErrors(cudaMemcpy((char*)host, device, size, cudaMemcpyDeviceToHost));
	}
	
	void copyArrayToDevice(void* device, const void* host, size_t size) {
		checkCudaErrors(cudaMemcpy((char*)device, host, size, cudaMemcpyHostToDevice));
	}

	void cudaComputeDensities(Particle* dev_particles) {
		return;
	}

	void cudaComputeForces(Particle* dev_particles) {
		return;
	}

	void cudaParticleCollisions(Particle* dev_particles) {
		return;
	}

}