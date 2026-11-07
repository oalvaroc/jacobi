#include "cli.hpp"
#include "mat.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <print>
#include <random>
#include <cstddef>
#include <omp.h>

struct jacobi_result {
	size_t iter;
	bool converged;
};

static jacobi_result omp_jacobi(mat2d<float>& mat, size_t n_iter, double epsilon) {
	mat2d<float> new_mat = mat;
	const size_t last = mat.order() - 1;

	size_t k = 0;
	while (k < n_iter) {
		k++;

		float diff = 0.0;
		#pragma omp parallel for schedule(static) reduction(max:diff)
		for (size_t row = 1; row < last; row++) {
			for (size_t col = 1; col < last; col++) {
				float value = 0.25f * (mat[row - 1, col] + mat[row + 1, col] +
					               mat[row, col + 1] + mat[row, col - 1]);

				diff = std::max(diff, std::fabs(value - mat[row, col]));
				new_mat[row, col] = value;
			}
		}

		new_mat.swap(mat);

		if (diff < epsilon) {
			return {k, true};
		}
	}
	return {k, false};
}

int main(int argc, char **argv) {
	cli_args args{};

	int err = parse_args(argc, argv, &args,
	                     "Iterative Jacobi method for a square matrix of order N, using OpenMP.\n"
	                     "Set the number of threads with OMP_NUM_THREADS.");
	if (err) {
		return err;
	}

	std::mt19937 gen;
	gen.seed(args.seed);
	std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

	auto mat = mat2d<float>::make_random(args.order, [&gen, &uniform](){
		return uniform(gen);
	});

	auto start = std::chrono::steady_clock::now();
	jacobi_result result = omp_jacobi(mat, args.iter, args.epsilon);
	std::chrono::duration<double> delta = (std::chrono::steady_clock::now() - start);

	std::println("elapsed={} iter={}/{} converged={} threads={}",
	             delta, result.iter, args.iter, result.converged, omp_get_max_threads());

	return 0;
}
