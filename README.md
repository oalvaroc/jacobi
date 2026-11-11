# Jacobi

Solves the 2D Laplace equation on a square grid with the Jacobi iterative method, and compares a sequential implementation with OpenMP and CUDA versions. Each iteration replaces every interior point with the average of its four neighbours; the boundary stays fixed.

| Program | Implementation |
|---|---|
| `serial` | Sequential reference |
| `omp` | OpenMP, parallel loop over rows |
| `omp_simd` | `omp` plus SIMD vectorization of the column loop |
| `cuda` | One GPU thread per cell, configurable block size |

The report (in Portuguese) with the validation and performance analysis is in [`docs/relatorio.md`](docs/relatorio.md).

## Building

Requires CMake 3.18+ and a C++23 compiler with OpenMP. The `cuda` program is built only if a CUDA compiler is found.

```sh
cmake -B build

# or use a specific C++ compiler
# cmake -B build -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_COMPILER=clang++

cmake --build build
```

## Running

```sh
build/src/serial -n 1024 -i 1000 -e 1e-4
OMP_NUM_THREADS=8 build/src/omp -n 1024 -i 1000
build/src/cuda -n 1024 -i 1000 -b 256
```

| Option | Meaning |
|---|---|
| `-n` | Grid order (required) |
| `-i` | Maximum number of iterations (default 10) |
| `-e` | Stop when the largest change in an iteration is below this value; 0 always runs every iteration (default 1e-6) |
| `-s` | Seed of the random initial grid (default 42) |
| `-o` | Write the final grid to a file as raw float32 |
| `-b` | Threads per CUDA block, `cuda` only (default 256) |

Each program prints the elapsed time, the number of iterations and whether it converged.

## Benchmark, validation and report

The scripts need Python 3; `compare.py` needs numpy and `plot.py` needs matplotlib.

```sh
# run every program over grid sizes, thread counts and block sizes; write a CSV to results/
scripts/benchmark.py --iter 500 --matrix-dir results/matrices

# compare final grids with the serial one: largest absolute and relative error
scripts/compare.py results/matrices/serial_n1024.bin results/matrices/cuda_n1024_b256.bin

# plot a benchmark CSV into docs/figures
scripts/plot.py results/benchmark_<timestamp>.csv

# build the report PDF (needs pandoc and tectonic)
make -C docs
```
