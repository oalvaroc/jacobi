#!/usr/bin/env python3
"""Compare matrices written with --output against a reference matrix, element by element."""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parses the command line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="matrix from the serial program")
    parser.add_argument("matrices", type=Path, nargs="+", help="matrices to check")
    parser.add_argument("--tolerance", type=float, default=1e-5,
                        help="maximum absolute error per element (default: %(default)s)")
    return parser.parse_args()


def compare(expected: np.ndarray, actual: np.ndarray, tolerance: float) -> tuple[float, float, int]:
    """Compares every element of a matrix with the reference matrix.

    The relative error of an element is its absolute error divided by the magnitude of
    the reference value; it is infinite when the reference value is zero and the
    absolute error is not.

    Args:
        expected: The reference matrix.
        actual: The matrix to check, with the same number of elements.
        tolerance: The maximum absolute error per element.

    Returns:
        The largest absolute error, the largest relative error and the number of
        elements whose absolute error is above the tolerance or NaN.
    """
    expected = expected.astype(np.float64)
    abs_error = np.abs(actual.astype(np.float64) - expected)
    magnitude = np.abs(expected)
    rel_error = np.divide(abs_error, magnitude, out=np.where(abs_error == 0, 0.0, np.inf),
                          where=magnitude != 0)
    above = int(np.count_nonzero(~(abs_error <= tolerance)))
    return float(np.max(abs_error)), float(np.max(rel_error)), above


def main() -> int:
    """Compares each matrix with the reference and prints the errors.

    Returns:
        The process exit code: 0 if every matrix is within the tolerance, 1 otherwise.
    """
    args = parse_args()
    expected = np.fromfile(args.reference, dtype=np.float32)
    all_passed = True

    print(f"reference: {args.reference} ({expected.size} elements), tolerance: {args.tolerance}")
    print(f"{'matrix':<40} {'max_abs_error':>14} {'max_rel_error':>14} {'above_tol':>10} result")
    for path in args.matrices:
        actual = np.fromfile(path, dtype=np.float32)
        if actual.size != expected.size:
            sys.exit(f"{path} has {actual.size} elements, the reference has {expected.size}")
        max_abs, max_rel, above = compare(expected, actual, args.tolerance)
        all_passed &= above == 0
        print(f"{str(path):<40} {max_abs:>14.6e} {max_rel:>14.6e} {above:>10} "
              f"{'ok' if above == 0 else 'FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
