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

// coord components must be 10 bits
__device__ uint coord2zIndex(Vector3i coord) {
    uint x_grid = coord.x();
    uint y_grid = coord.y();
    uint z_grid = coord.z();

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

__device__ uint get_Z_index(Particle p, SimParams *params) {
    // find the section of the grid the particle is in
    float posX = p.position.x() - params->boxMin.x;
    float posY = p.position.y() - params->boxMin.y;
    float posZ = p.position.z() - params->boxMin.z;
    Vector3i coord;
    coord.x() = floor((posX / params->boxDims.x) * params->gridDim);
    coord.y() = floor((posY / params->boxDims.y) * params->gridDim);
    coord.z() = floor((posZ / params->boxDims.z) * params->gridDim);
    return coord2zIndex(coord);
}

// 30  bit number -> 10 bit number
__device__ uint collapseEvery3(uint x) {
    uint res = 0;
    for (int i = 0; i < 10; i++) {
        uint b = ((x >> i * 3) & 1);
        res = res | (b << i);
    }
    return res;
}

__device__ dim3 zIndex2coord(uint z_index) {
    uint x = collapseEvery3(z_index);
    uint y = collapseEvery3(z_index >> 1);
    uint z = collapseEvery3(z_index >> 2);
		dim3 res;
		res.x = x;
		res.y = y;
		res.z = z;
    return res;
}

// return if the given coordinates are in the bounds of the sim grid
__device__ int inGrid(int x, int y, int z, SimParams *params) {
	int gridMax = params->gridDim;
	return (x < gridMax) && (y < gridMax) && (z < gridMax) && (x >= 0) && (y >= 0) && (z >= 0);
}

__global__ void kernelComputeDensities() {
	return;
}

__global__ void kernelComputeForces(Particle *dev_particles, Grid_item *dev_B, Grid_item *dev_B_prime, SimParams *params) {
	__shared__ Particle batch[GRID_COMPACT_WIDTH];
	__shared__ int copied = 0;
	uint blockid = blockIdx.x;
	uint particleid = dev_B_prime[blockid].start + threadIdx.x;
	Particle pi = dev_particles[particleid];
	pi.force_press = { 0.f, 0.f, 0.f };
	pi.force_visc = { 0.f, 0.f, 0.f };

	// for each neighboring grid block
	dim3 coords = zIndex2coord(pi.zindex);
	for (int dx = -1; dx < 2; dx++) {
		for (int dy = -1; dy < 2; dy++) {
			for (int dz = -1; dz < 2; dz++) {
				int neighborx = coords.x + dx;
				int neighbory = coords.y + dy;
				int neighborz = coords.z + dz;
				// check coordinates are valid
				if (!inGrid(neighborx, neighbory, neighborz, params)) {
					continue;
				}
				// get the zindex of the block
				int blockZIndex = coord2zIndex(Vector3i(neighborx, neighbory, neighborz));
				Grid_item neighbor_block = dev_B_prime[blockZIndex];
				int start = neighbor_block.start;
				int nParticles = neighbor_block.nParticles;

				while (copied < nParticles) {
					// in batches, copy particles into the batch array for processing
					int toCopyDex = copied + threadIdx.x;
					if (toCopyDex < nParticles) {
						batch[threadIdx.x] = dev_particles[start + toCopyDex];
						atomicAdd(&copied, 1); // count the particle as copied
					}
					__syncthreads();
					// each active thread computes values for itself from its neighbors
					for (int j = 0; j < copied%GRID_COMPACT_WIDTH; j++) {
						Particle pj = dev_particles[j];
						computeForce(&pi, &pj);
					}
				}
			}
		}
	}
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
