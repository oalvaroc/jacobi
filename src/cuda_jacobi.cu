#include "cuda_jacobi.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

__global__ void jacobi_step(const float *src, float *dst, int *not_converged,
                            size_t order, double threshold) {
	const size_t row = blockIdx.y + 1;
	const size_t col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (col >= order - 1) {
		return;
	}

	const size_t center = row * order + col;
	const float value = 0.25f * (src[center - order] + src[center + order] +
	                             src[center + 1] + src[center - 1]);

	if (fabsf(value - src[center]) >= threshold) {
		*not_converged = 1;
	}
	dst[center] = value;
}

static void check(cudaError_t err, const char *operation) {
	if (err != cudaSuccess) {
		std::fprintf(stderr, "CUDA error: %s: %s\n", operation, cudaGetErrorString(err));
		std::exit(EXIT_FAILURE);
	}
}

static void *device_alloc(size_t bytes) {
	void *ptr = nullptr;
	check(cudaMalloc(&ptr, bytes), "cudaMalloc");
	return ptr;
}

jacobi_result cuda_jacobi(float *grid, size_t order, size_t n_iter, double epsilon,
                          unsigned block_size) {
	const size_t bytes = order * order * sizeof(float);
	const size_t interior = order - 2;
	const double threshold = epsilon > 0.0 ? epsilon : HUGE_VAL;
	const dim3 blocks((interior + block_size - 1) / block_size, interior);
	const dim3 threads(block_size);

	float *src = static_cast<float *>(device_alloc(bytes));
	float *dst = static_cast<float *>(device_alloc(bytes));
	int *not_converged = static_cast<int *>(device_alloc(sizeof(int)));

	check(cudaMemcpy(src, grid, bytes, cudaMemcpyHostToDevice), "upload");
	check(cudaMemcpy(dst, grid, bytes, cudaMemcpyHostToDevice), "upload");

	size_t k = 0;
	bool converged = false;
	while (k < n_iter && !converged) {
		k++;

		check(cudaMemset(not_converged, 0, sizeof(int)), "cudaMemset");

		if (interior > 0) {
			jacobi_step<<<blocks, threads>>>(src, dst, not_converged, order, threshold);
			check(cudaGetLastError(), "jacobi_step");
		}

		std::swap(src, dst);

		if (epsilon > 0.0) {
			int flag = 0;
			check(cudaMemcpy(&flag, not_converged, sizeof(int), cudaMemcpyDeviceToHost), "download");
			converged = flag == 0;
		}
	}

	check(cudaMemcpy(grid, src, bytes, cudaMemcpyDeviceToHost), "download");

	check(cudaFree(src), "cudaFree");
	check(cudaFree(dst), "cudaFree");
	check(cudaFree(not_converged), "cudaFree");

	return {k, converged};
}
