#include "mat.hpp"
#include <algorithm>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cmath>
#include <limits>
#include <print>
#include <random>
#include <string>
#include <system_error>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

#define DEFAULT_ITER 10
#define DEFAULT_SEED 42
#define DEFAULT_EPSILON 1e-6

struct cli_args {
	size_t order;
	size_t iter;
	size_t seed;
	double epsilon;
};

struct jacobi_result {
	size_t iter;
	bool converged;
};

template <typename T>
static bool parse_number(const char *str, T& out) {
	const char *end = str + std::strlen(str);
	auto [ptr, ec] = std::from_chars(str, end, out);
	return ec == std::errc{} && ptr == end;
}

static void help(const std::string& prog) {
	std::print("usage: {} OPTIONS\n\n", prog);
	std::print("Iterative Jacobi method for a square matrix of order N.\n\n");
	std::println("Options:");
	std::println("    -n, --order    Square matrix order");
	std::println("    -i, --iter     Number of iterations (default=10)");
	std::println("    -s, --seed     RNG seed (default=42)");
	std::println("    -e, --epsilon  Tolerance (default={})", DEFAULT_EPSILON);
	std::println("    -h, --help     Show this message and exit");
}

static int parse_args(int argc, char **argv, cli_args *args) {
	static option opts[] = {
		{"order",   required_argument, NULL, 'n'},
		{"iter",    required_argument, NULL, 'i'},
		{"seed",    required_argument, NULL, 's'},
		{"epsilon", required_argument, NULL, 'e'},
		{"help",    no_argument,       NULL, 'h'},
		{NULL,      0,                 NULL,  0 }
	};

	if (argc == 1) {
		help(argv[0]);
		return EXIT_FAILURE;
	}

	assert(args != nullptr);

	args->order = 0;
	args->iter = DEFAULT_ITER;
	args->seed = DEFAULT_SEED;
	args->epsilon = DEFAULT_EPSILON;

	int opt;
	while ((opt = getopt_long(argc, argv, "hn:i:s:e:", opts, NULL)) > 0) {
		switch (opt) {
		case 'h':
			help(argv[0]);
			return EXIT_FAILURE;
		case 'n':
			if (!parse_number(optarg, args->order)) {
				std::println(stderr, "invalid order: '{}'", optarg);
				return EXIT_FAILURE;
			}

			if (args->order < 2) {
				std::println(stderr, "matrix order must be at least 2");
				return EXIT_FAILURE;
			}

			if (args->order > std::numeric_limits<size_t>::max() / args->order) {
				std::println(stderr, "matrix order is too large: {}", args->order);
				return EXIT_FAILURE;
			}
			break;
		case 'i':
			if (!parse_number(optarg, args->iter)) {
				std::println(stderr, "invalid number of iterations: '{}'", optarg);
				return EXIT_FAILURE;
			}

			if (args->iter < 1) {
				std::println(stderr, "number of iterations must be at least 1");
				return EXIT_FAILURE;
			}
			break;
		case 'e':
			if (!parse_number(optarg, args->epsilon)) {
				std::println(stderr, "invalid epsilon: '{}'", optarg);
				return EXIT_FAILURE;
			}

			if (!std::isfinite(args->epsilon) || args->epsilon < 0.0) {
				std::println(stderr, "epsilon must be finite and not negative");
				return EXIT_FAILURE;
			}
			break;
		case 's':
			if (!parse_number(optarg, args->seed)) {
				std::println(stderr, "invalid seed: '{}'", optarg);
				return EXIT_FAILURE;
			}
			break;
		default:
			return EXIT_FAILURE;
		}
	}

	if (args->order == 0) {
		std::println(stderr, "missing required option: -n, --order");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

static jacobi_result serial_jacobi(mat2d<float>& mat, size_t n_iter, double epsilon) {
	mat2d<float> new_mat = mat;

	size_t k = 0;
	while (k < n_iter) {
		k++;

		float diff = 0.0;
		for (size_t row = 1; row < mat.order() - 1; row++) {
			for (size_t col = 1; col < mat.order() - 1; col++) {
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

	int err = parse_args(argc, argv, &args);
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
	jacobi_result result = serial_jacobi(mat, args.iter, args.epsilon);
	std::chrono::duration<double> delta = (std::chrono::steady_clock::now() - start);

	std::println("elapsed={} iter={}/{} converged={}",
	             delta, result.iter, args.iter, result.converged);

	return 0;
}
