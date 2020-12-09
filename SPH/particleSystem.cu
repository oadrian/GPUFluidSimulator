// implementation of kernels

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "particleSystem.cuh"


__global__ void kernelComputeDensities() {
	return;
}

__global__ void kernelComputeForces() {
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

	void allocateArray(void** devPtr, int size) {
		checkCudaErrors(cudaMalloc(devPtr, size));
	}

	void freeArray(void* devPtr) {
		checkCudaErrors(cudaFree(devPtr));
	}

	void threadSync() {
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void copyArrayFromDevice(void* host, const void* device, int size) {
		checkCudaErrors(cudaMemcpy((char*)host, device, size, cudaMemcpyDeviceToHost));
	}
	
	void copyArrayToDevice(void* device, const void* host, int size) {
		checkCudaErrors(cudaMemcpy((char*)device, host, size, cudaMemcpyHostToDevice));
	}

	void cudaComputeDensities() {
		return;
	}

	void cudaComputeForces() {
		return;
	}

	void particleCollisions() {
		return;
	}

}