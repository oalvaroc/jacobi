#pragma once

#include <cassert>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <limits>
#include <print>
#include <string>
#include <system_error>

#define DEFAULT_ITER 10
#define DEFAULT_SEED 42
#define DEFAULT_EPSILON 1e-6

struct cli_args {
	size_t order;
	size_t iter;
	size_t seed;
	double epsilon;
};

template <typename T>
bool parse_number(const char *str, T& out) {
	const char *end = str + std::strlen(str);
	auto [ptr, ec] = std::from_chars(str, end, out);
	return ec == std::errc{} && ptr == end;
}

inline void help(const std::string& prog, const char *description) {
	std::print("usage: {} OPTIONS\n\n", prog);
	std::print("{}\n\n", description);
	std::println("Options:");
	std::println("    -n, --order    Square matrix order");
	std::println("    -i, --iter     Number of iterations (default={})", DEFAULT_ITER);
	std::println("    -s, --seed     RNG seed (default={})", DEFAULT_SEED);
	std::println("    -e, --epsilon  Tolerance (default={})", DEFAULT_EPSILON);
	std::println("    -h, --help     Show this message and exit");
}

inline int parse_args(int argc, char **argv, cli_args *args, const char *description) {
	static option opts[] = {
		{"order",   required_argument, NULL, 'n'},
		{"iter",    required_argument, NULL, 'i'},
		{"seed",    required_argument, NULL, 's'},
		{"epsilon", required_argument, NULL, 'e'},
		{"help",    no_argument,       NULL, 'h'},
		{NULL,      0,                 NULL,  0 }
	};

	if (argc == 1) {
		help(argv[0], description);
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
			help(argv[0], description);
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
