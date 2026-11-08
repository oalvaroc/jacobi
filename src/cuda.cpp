#include "cli.hpp"
#include "cuda_jacobi.hpp"
#include "mat.hpp"
#include <chrono>
#include <print>
#include <random>

int main(int argc, char **argv) {
	cli_args args{};
	size_t block_size;

	int err = parse_args(argc, argv, &args,
	                     "Iterative Jacobi method for a square matrix of order N, using CUDA.",
	                     &block_size);
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
	jacobi_result result = cuda_jacobi(mat.data(), mat.order(), args.iter, args.epsilon,
	                                   static_cast<unsigned>(block_size));
	std::chrono::duration<double> delta = (std::chrono::steady_clock::now() - start);

	std::println("elapsed={} iter={}/{} converged={} block_size={}",
	             delta, result.iter, args.iter, result.converged, block_size);

	if (args.output && !mat.save(args.output)) {
		std::println(stderr, "failed to write the matrix to '{}'", args.output);
		return EXIT_FAILURE;
	}

	return 0;
}
