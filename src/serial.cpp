#include <cassert>
#include <exception>
#include <print>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <cstdlib>
#include <getopt.h>

#define DEFAULT_SEED 42

struct cli_args {
	size_t order;
	size_t seed;
};

static void help(const std::string& prog) {
	std::print("usage: {} OPTIONS\n\n", prog);
	std::print("Iterative Jacobi method for a square matrix of order N.\n\n");
	std::println("Options:");
	std::println("    -n, --order    Square matrix order");
	std::println("    -s, --seed     RNG seed (default=42)");
	std::println("    -h, --help     Show this message and exit");
}

static int parse_args(int argc, char **argv, cli_args *args) {
	static option opts[] = {
		{"order", required_argument, NULL, 'n'},
		{"seed",  required_argument, NULL, 's'},
		{"help",  no_argument,       NULL, 'h'},
		{NULL, 0, NULL, 0}
	};

	if (argc == 1) {
		help(argv[0]);
		return EXIT_FAILURE;
	}

	assert(args != nullptr);

	args->order = 0;
	args->seed = DEFAULT_SEED;

	int opt;
	while ((opt = getopt_long(argc, argv, "hn:s:", opts, NULL)) > 0) {
		switch (opt) {
		case 'h':
			help(argv[0]);
			return EXIT_FAILURE;
		case 'n': {
			try {
				args->order = std::stoll(optarg);
			} catch (std::exception& e) {
				std::println(stderr, "failed to parse order: {}", e.what());
				return EXIT_FAILURE;
			}

			if (args->order < 2) {
				std::println("matrix order must be a number greater than 2");
				return EXIT_FAILURE;
			}
			break;
		}
		case 'r':
			try {
				args->seed = std::stoll(optarg);
			} catch (std::exception& e) {
				std::println(stderr, "failed to parse seed: {}", e.what());
				return EXIT_FAILURE;
			}
			break;
		case '?':
			break;
		default:
			return EXIT_FAILURE;
		}
	}

	if (args->order == 0) {
		std::println("missing required option: -n, --order\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
	cli_args args{};

	int err = parse_args(argc, argv, &args);
	if (err) {
		return err;
	}

	return 0;
}
