"""Shared console logging and number-formatting helpers for the pipeline."""

import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints clean, single-line messages to stdout."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def section(title: str) -> None:
    """Print a visually distinct banner for a new pipeline stage/section."""
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def fmt_int(n: int) -> str:
    """Format an integer with thousands separators, e.g. 12345 -> '12,345'."""
    return f"{n:,}"


def fmt_float(x: float, decimals: int = 4) -> str:
    """Format a float with fixed decimals and thousands separators."""
    return f"{x:,.{decimals}f}"


def print_table(headers: list, rows: list, aligns: list = None) -> None:
    """
    Print a simple, cleanly aligned monospace table.

    headers: column names.
    rows:    list of rows, each a list of pre-formatted strings.
    aligns:  per-column alignment, "l" or "r" (default: "r" for all).
    """
    n_cols = len(headers)
    aligns = aligns or ["r"] * n_cols
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows)) if rows
        else len(str(headers[i]))
        for i in range(n_cols)
    ]

    def _pad(value, width, align):
        return str(value).rjust(width) if align == "r" else str(value).ljust(width)

    header_line = "  ".join(_pad(h, w, a) for h, w, a in zip(headers, widths, aligns))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("  ".join(_pad(v, w, a) for v, w, a in zip(row, widths, aligns)))
