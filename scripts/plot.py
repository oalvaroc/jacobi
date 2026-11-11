#!/usr/bin/env python3
"""Plot the benchmark results for the report."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
STYLE = {
    "serial": ("#2a78d6", "o"),
    "omp": ("#eb6834", "s"),
    "omp_simd": ("#1baf7a", "^"),
    "cuda": ("#eda100", "D"),
}
INK = "#3d3d3a"
MUTED = "#8a8980"
GRID = "#e4e3dd"
MIN_BLOCK_ORDER = 4096


def parse_args() -> argparse.Namespace:
    """Parses the command line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV file written by benchmark.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures",
                        help="directory for the PNG files (default: docs/figures)")
    return parser.parse_args()


def load_medians(path: Path) -> dict[tuple[str, int, int, int], float]:
    """Reads a benchmark CSV and computes the median time of each configuration.

    The thread count is ignored for the CUDA program and the block size for the others.
    CSV files written before the block_size column existed count as block size 256.

    Args:
        path: The CSV file.

    Returns:
        The median elapsed time in seconds for each (program, order, threads,
        block_size) configuration.
    """
    times = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            program = row["program"]
            cuda = program == "cuda"
            threads = 0 if cuda else int(row["threads"])
            block_size = int(row.get("block_size") or 256) if cuda else 0
            times[(program, int(row["order"]), threads, block_size)].append(float(row["elapsed_s"]))
    return {key: statistics.median(values) for key, values in times.items()}


def style_axes(ax: plt.Axes) -> None:
    """Applies a light grid and recessive axes to a plot.

    Args:
        ax: The axes to style.
    """
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def plot_times(medians: dict, iterations: int, output: Path) -> None:
    """Plots the time of each program against the matrix order.

    The OpenMP programs use their best thread count and CUDA uses block size 256.

    Args:
        medians: The median times from load_medians.
        iterations: The number of iterations of each run.
        output: The PNG file to write.
    """
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for program, (color, marker) in STYLE.items():
        points = defaultdict(list)
        for (prog, order, _, block_size), time in medians.items():
            if prog == program and (program != "cuda" or block_size == 256):
                points[order].append(time)
        if not points:
            continue
        orders = sorted(points)
        ax.plot(orders, [min(points[n]) for n in orders], color=color, marker=marker,
                markersize=6, linewidth=2, label=program)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(orders, [str(n) for n in orders])
    ax.set_xlabel("ordem da matriz (n)", color=INK, fontsize=9)
    ax.set_ylabel(f"tempo de {iterations} iterações (s)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_threads(medians: dict, output: Path) -> None:
    """Plots the OpenMP speedup over one thread against the thread count, one panel per order.

    Args:
        medians: The median times from load_medians.
        output: The PNG file to write.
    """
    orders = sorted({order for (prog, order, _, _) in medians if prog == "omp"})
    fig, axes = plt.subplots(1, len(orders), figsize=(7, 2.6), sharey=True)
    for ax, order in zip(axes, orders):
        for program in ("omp", "omp_simd"):
            color, marker = STYLE[program]
            series = {t: time for (prog, n, t, _), time in medians.items()
                      if prog == program and n == order}
            threads = sorted(series)
            ax.plot(threads, [series[1] / series[t] for t in threads], color=color,
                    marker=marker, markersize=5, linewidth=2, label=program)
        ax.plot(threads, threads, color=MUTED, linestyle="--", linewidth=1, label="ideal")
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads, [str(t) for t in threads])
        ax.set_ylim(0, 8.5)
        ax.set_title(f"n = {order}", color=INK, fontsize=9)
        ax.set_xlabel("threads", color=INK, fontsize=8)
        style_axes(ax)
    axes[0].set_ylabel("speedup sobre 1 thread", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_block_sizes(medians: dict, output: Path) -> bool:
    """Plots the CUDA time against the block size, one line per matrix order.

    Orders below MIN_BLOCK_ORDER are left out because their time is dominated by the
    fixed cost of initialization and transfers, not by the kernel.

    Args:
        medians: The median times from load_medians.
        output: The PNG file to write.

    Returns:
        False if the data has fewer than two block sizes, in which case nothing is written.
    """
    cuda = {(n, b): time for (prog, n, _, b), time in medians.items() if prog == "cuda"}
    block_sizes = sorted({b for _, b in cuda})
    if len(block_sizes) < 2:
        return False

    orders = sorted({n for n, _ in cuda if n >= MIN_BLOCK_ORDER})
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for (color, marker), order in zip(STYLE.values(), orders):
        ax.plot(block_sizes, [cuda[(order, b)] / cuda[(order, 256)] for b in block_sizes],
                color=color, marker=marker, markersize=6, linewidth=2, label=f"n = {order}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(block_sizes, [str(b) for b in block_sizes])
    ax.set_xlabel("threads por bloco", color=INK, fontsize=9)
    ax.set_ylabel("tempo relativo a 256 threads", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    """Writes the report figures."""
    args = parse_args()
    medians = load_medians(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.csv.open(newline="") as file:
        iterations = int(next(csv.DictReader(file))["iterations_requested"])
    plot_times(medians, iterations, args.output_dir / "tempo.png")
    plot_threads(medians, args.output_dir / "omp_threads.png")
    plot_block_sizes(medians, args.output_dir / "cuda_blocos.png")


if __name__ == "__main__":
    main()
