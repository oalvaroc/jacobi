#pragma once

#include <cstddef>

struct jacobi_result {
	size_t iter;
	bool converged;
};

jacobi_result cuda_jacobi(float *grid, size_t order, size_t n_iter, double epsilon,
                          unsigned block_size);
