#!/usr/bin/env python3
"""Run the Jacobi programs across configurations and save the timings to a CSV file."""

import argparse
import csv
import datetime
import itertools
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROGRAMS = ("serial", "omp", "omp_simd")
OUTPUT_RE = re.compile(r"elapsed=(\S+)s iter=(\d+)/(\d+) converged=(true|false)")
FIELDS = (
    "timestamp",
    "host",
    "program",
    "order",
    "threads",
    "iterations_requested",
    "iterations",
    "converged",
    "epsilon",
    "seed",
    "repetition",
    "elapsed_s",
)


def default_threads() -> list[int]:
    """Lists the default thread counts.

    Returns:
        The powers of two up to the number of logical CPUs.
    """
    count = os.cpu_count() or 1
    threads = [1]
    while threads[-1] * 2 <= count:
        threads.append(threads[-1] * 2)
    return threads


def parse_args() -> argparse.Namespace:
    """Parses the command line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs", nargs="+", choices=PROGRAMS, default=list(PROGRAMS))
    parser.add_argument("--orders", nargs="+", type=int, default=[256, 1024, 4096, 8192],
                        help="matrix orders (default: %(default)s)")
    parser.add_argument("--threads", nargs="+", type=int, default=default_threads(),
                        help="thread counts for the OpenMP programs (default: %(default)s)")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="runs per configuration (default: %(default)s)")
    parser.add_argument("--iter", type=int, default=100,
                        help="iterations per run (default: %(default)s)")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="tolerance; 0 always runs every iteration (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: %(default)s)")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build",
                        help="CMake build directory with the programs (default: build)")
    parser.add_argument("--output", type=Path,
                        help="CSV file to create (default: results/benchmark_<timestamp>.csv)")
    return parser.parse_args()


def make_runs(args: argparse.Namespace) -> list[tuple[str, int, int, int]]:
    """Lists every run of the benchmark in shuffled order.

    The serial program only runs with one thread. Shuffling keeps slow drift, such as
    the machine heating up, from always affecting the same configuration.

    Args:
        args: The parsed command line options.

    Returns:
        A (program, order, threads, repetition) tuple for each run.
    """
    runs = []
    for program, order, repetition in itertools.product(
        args.programs, args.orders, range(1, args.repetitions + 1)
    ):
        for threads in [1] if program == "serial" else args.threads:
            runs.append((program, order, threads, repetition))
    random.Random(args.seed).shuffle(runs)
    return runs


def run_once(
    program: str, order: int, threads: int, args: argparse.Namespace
) -> dict[str, int | float | str]:
    """Runs a program once.

    Args:
        program: The name of the program to run.
        order: The matrix order.
        threads: The number of OpenMP threads.
        args: The parsed command line options.

    Returns:
        The CSV fields parsed from the program's output.

    Raises:
        RuntimeError: The program failed or its output could not be parsed.
    """
    command =[str(args.build_dir / "src" / program), "-n", str(order), "-i", str(args.iter),
               "-e", str(args.epsilon), "-s", str(args.seed)]
    env = os.environ | {
        "OMP_NUM_THREADS": str(threads),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    process = subprocess.run(command, env=env, capture_output=True, text=True)
    match = OUTPUT_RE.search(process.stdout)
    if process.returncode != 0 or not match:
        raise RuntimeError(f"{' '.join(command)} failed with exit code {process.returncode}\n"
                           f"{process.stdout}{process.stderr}")
    return {
        "iterations": int(match[2]),
        "converged": match[4],
        "elapsed_s": float(match[1]),
    }


def main() -> int:
    """Runs all configurations, writing each result to the CSV file as it finishes.

    Returns:
        The process exit code.
    """
    args = parse_args()
    started = datetime.datetime.now().astimezone()
    output = args.output or ROOT / "results" / f"benchmark_{started:%Y%m%d_%H%M%S}.csv"

    for program in args.programs:
        path = args.build_dir / "src" / program
        if not path.is_file():
            sys.exit(f"program not found: {path}; build the project first")

    runs = make_runs(args)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", newline="") as file:
        print(f"{len(runs)} runs, writing to {output}", file=sys.stderr)
        writer = csv.DictWriter(file, FIELDS)
        writer.writeheader()
        for index, (program, order, threads, repetition) in enumerate(runs, 1):
            result = run_once(program, order, threads, args)
            writer.writerow({
                "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "host": platform.node(),
                "program": program,
                "order": order,
                "threads": threads,
                "iterations_requested": args.iter,
                "epsilon": args.epsilon,
                "seed": args.seed,
                "repetition": repetition,
                **result,
            })
            file.flush()
            print(f"[{index}/{len(runs)}] {program} n={order} threads={threads} "
                  f"rep={repetition}: {result['elapsed_s']:.4f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileExistsError as error:
        sys.exit(f"output file already exists: {error.filename}")
    except RuntimeError as error:
        sys.exit(f"benchmark failed: {error}")
    except KeyboardInterrupt:
        sys.exit("interrupted; completed runs are saved in the CSV file")
