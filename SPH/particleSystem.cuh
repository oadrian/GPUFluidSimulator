// header for c function definitions and cuda kernels
#include "particles_kernel.cuh"
extern "C"
{
	void cudaInit(int argc, char** argv);

	void allocateArray(void** devPtr, size_t size);
	void freeArray(void* devPtr);

	void threadSync();

	void copyArrayFromDevice(void *host, const void *device, size_t size);
	void copyArrayToDevice(void* device, const void* host, size_t size);

	void cudaComputeDensities(Particle *dev_particles);
	void cudaComputeForces(Particle* dev_particles);
	void cudaParticleCollisions(Particle* dev_particles);
}