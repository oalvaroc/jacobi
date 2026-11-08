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

static float sweep_row(const float *src, float *dst, size_t order, size_t row) {
	const float *above = src + (row - 1) * order;
	const float *center = src + row * order;
	const float *below = src + (row + 1) * order;
	float *out = dst + row * order;

	float diff = 0.0f;
	#pragma omp simd reduction(max:diff)
	for (size_t col = 1; col < order - 1; col++) {
		float value = 0.25f * (above[col] + below[col] +
		                       center[col + 1] + center[col - 1]);
		float delta = std::fabs(value - center[col]);

		diff = diff > delta ? diff : delta;
		out[col] = value;
	}
	return diff;
}

static jacobi_result omp_simd_jacobi(mat2d<float>& mat, size_t n_iter, double epsilon) {
	mat2d<float> new_mat = mat;
	const size_t order = mat.order();
	const size_t last = order - 1;

	size_t k = 0;
	while (k < n_iter) {
		k++;

		const float *src = mat.data();
		float *dst = new_mat.data();

		float diff = 0.0;
		#pragma omp parallel for schedule(static) reduction(max:diff)
		for (size_t row = 1; row < last; row++) {
			diff = std::max(diff, sweep_row(src, dst, order, row));
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
	                     "Iterative Jacobi method for a square matrix of order N, using OpenMP\n"
	                     "threads and SIMD. Set the number of threads with OMP_NUM_THREADS.");
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
	jacobi_result result = omp_simd_jacobi(mat, args.iter, args.epsilon);
	std::chrono::duration<double> delta = (std::chrono::steady_clock::now() - start);

	std::println("elapsed={} iter={}/{} converged={} threads={}",
	             delta, result.iter, args.iter, result.converged, omp_get_max_threads());

	if (args.output && !mat.save(args.output)) {
		std::println(stderr, "failed to write the matrix to '{}'", args.output);
		return EXIT_FAILURE;
	}

	return 0;
}
