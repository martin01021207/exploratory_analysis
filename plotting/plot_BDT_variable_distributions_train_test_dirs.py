#!/usr/bin/env python3
"""
Plot BDT-variable distributions for real background and proxy CR simulation.

Reads vars_sXX_{bkg,sig}_{train,test}.root (default station: 23), using
vars_bkg / vars_sig trees or automatic tree discovery. Training and testing
samples are concatenated within each population; no event selection is applied.
The output PDF contains exactly two pages: eight waveform variables in a 2x4
layout, then five reconstruction/hit variables in a 2x3 layout.

Histograms retain count-based overlays and a shared range: pooled
0.1/99.9 percentiles with 5% padding. Simulation uses --bins (default: 160);
background uses --bkg-bins (default: twice --bins, hence 320). Both use
unweighted event counts per bin, so narrower bins generally have lower counts.
Values outside the histogram range are not drawn. Nonfinite values are removed independently per variable;
legends count finite values (including values outside the bin range), while
page titles count all input tree entries. No threshold lines are used.

Examples:
    python plot_BDT_variable_distributions_train_test_dirs.py \
        --train-dir /path/to/train --test-dir /path/to/test --station 23 \
        --output BDT_variable_distributions_s23.pdf

    python plot_BDT_variable_distributions_train_test_dirs.py \
        --input-dir /path/to/files --station 23 --linear-y

Explicit --bkg-train, --sig-train, --bkg-test, and --sig-test paths override
the corresponding directory-derived paths. Relative explicit paths are resolved
from the working directory, as in the original script. All 13 exact branch
names below are required; --bkg-tree and --sig-tree select preferred trees.

Dependencies: numpy, matplotlib, uproot (Python 3.10 or newer).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import uproot
except ImportError as exc:
    raise SystemExit(
        "This script requires uproot. Install it with:\n"
        "    pip install uproot awkward\n"
    ) from exc


PAGE_1_VARIABLES = [
    "averageImpulsivity_PA",
    "coherentKurtosis_PA",
    "averageKurtosis_inIce",
    "averageEntropy_inIce",
    "averageImpulsivity_inIce",
    "coherentKurtosis_inIce",
    "coherentEntropy_inIce",
    "coherentImpulsivity_inIce",
]

PAGE_2_VARIABLES = [
    "reco_max_corr",
    "reco_surf_corr_z",
    "reco_surf_corr_zen",
    "passed_hit_filter",
    "nCoincidentPairs_inIce",
]

VARIABLES = PAGE_1_VARIABLES + PAGE_2_VARIABLES

DEFAULT_TREE_NAMES = {
    "bkg": "vars_bkg",
    "sig": "vars_sig",
}


def find_tree_name(file_path: Path, preferred: str | None = None) -> str:
    """Return the tree name to use in a ROOT file.

    If `preferred` is present, it is used. Otherwise, the first TTree-like object
    containing all requested VARIABLES is used.
    """
    with uproot.open(file_path) as root_file:
        keys = [key.split(";")[0] for key in root_file.keys()]

        if preferred is not None and preferred in root_file:
            return preferred

        for key in keys:
            try:
                obj = root_file[key]
                branches = set(obj.keys())
            except Exception:
                continue
            if all(var in branches for var in VARIABLES):
                return key

    raise KeyError(
        f"Could not find a tree containing {VARIABLES} in {file_path}. "
        f"Available keys: {keys}"
    )


def read_variables(
    file_path: Path, tree_name: str | None = None
) -> Tuple[Dict[str, np.ndarray], int]:
    """Read exact requested branches and return finite arrays plus entry count."""
    tree = find_tree_name(file_path, preferred=tree_name)
    with uproot.open(file_path) as root_file:
        branches = set(root_file[tree].keys())
        missing = [var for var in VARIABLES if var not in branches]
        if missing:
            raise KeyError(f"Missing branches in {file_path}:{tree}: {missing}")
        n_events = int(root_file[tree].num_entries)
        arrays = root_file[tree].arrays(VARIABLES, library="np")

    cleaned: Dict[str, np.ndarray] = {}
    for var in VARIABLES:
        arr = np.asarray(arrays[var], dtype=float)
        arr = arr[np.isfinite(arr)]
        cleaned[var] = arr
    return cleaned, n_events


def common_bins(real: np.ndarray, sim: np.ndarray, variable: str, n_bins: int) -> np.ndarray:
    """Create robust common bins for real and simulation arrays."""
    both = np.concatenate([real, sim])
    both = both[np.isfinite(both)]
    if both.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)

    lo, hi = np.nanpercentile(both, [0.1, 99.9])

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.nanmedian(both)) if both.size else 0.0
        lo, hi = center - 0.5, center + 0.5

    pad = 0.05 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def plot_page(
    pdf: PdfPages,
    title: str,
    variables: Iterable[str],
    layout: Tuple[int, int],
    figsize: Tuple[float, float],
    real_data: Dict[str, np.ndarray],
    sim_data: Dict[str, np.ndarray],
    n_real: int,
    n_sim: int,
    n_bins: int,
    log_y: bool,
    bkg_bins: int | None = None,
) -> None:
    """Append one page of combined-event histograms to the PDF."""
    fig, axes = plt.subplots(*layout, figsize=figsize, squeeze=False)
    fig.suptitle(
        f"{title}\nReal background N = {n_real:,}, simulation N = {n_sim:,}",
        fontsize=16,
    )

    variables = list(variables)
    for ax, var in zip(axes.flat, variables):
        real = real_data[var]
        sim = sim_data[var]
        bins = common_bins(real, sim, var, n_bins)
        background_bins = np.linspace(
            bins[0], bins[-1], (bkg_bins if bkg_bins is not None else 2 * n_bins) + 1
        )

        ax.hist(
            real,
            bins=background_bins,
            histtype="stepfilled",
            alpha=0.60,
            label=f"Background real data (N={len(real):,})",
        )
        ax.hist(
            sim,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            label=f"Proxy CR simulation (N={len(sim):,})",
        )

        ax.set_title(var)
        ax.set_xlabel(var)
        ax.set_ylabel("Events")
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in list(axes.flat)[len(variables):]:
        ax.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def combine_dicts(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {var: np.concatenate([a[var], b[var]]) for var in VARIABLES}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 13 BDT variables from combined training + testing ROOT events in exactly two PDF pages."
    )
    parser.add_argument("--input-dir", default=".", help="Fallback directory containing the four ROOT files, or parent directory if --train-dir/--test-dir are not used.")
    parser.add_argument("--train-dir", default=None, help="Directory containing vars_sXX_bkg_train.root and vars_sXX_sig_train.root.")
    parser.add_argument("--test-dir", default=None, help="Directory containing vars_sXX_bkg_test.root and vars_sXX_sig_test.root.")
    parser.add_argument("--station", default="23", help="Station number used for default filenames.")
    parser.add_argument("--bkg-train", default=None, help="Background training ROOT file.")
    parser.add_argument("--sig-train", default=None, help="Signal/simulation training ROOT file.")
    parser.add_argument("--bkg-test", default=None, help="Background testing ROOT file.")
    parser.add_argument("--sig-test", default=None, help="Signal/simulation testing ROOT file.")
    parser.add_argument("--bkg-tree", default=DEFAULT_TREE_NAMES["bkg"], help="Background tree name. Default: vars_bkg")
    parser.add_argument("--sig-tree", default=DEFAULT_TREE_NAMES["sig"], help="Signal/simulation tree name. Default: vars_sig")
    parser.add_argument("--output", default=None, help="Output PDF filename.")
    parser.add_argument("--bins", type=int, default=160, help="Number of simulation histogram bins (default: 160). Background defaults to twice this number.")
    parser.add_argument("--bkg-bins", type=int, default=None, help="Number of background histogram bins (default: twice --bins).")
    parser.add_argument("--linear-y", action="store_true", help="Use linear y-axis instead of log y-axis.")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be a positive integer")
    if args.bkg_bins is not None and args.bkg_bins < 1:
        parser.error("--bkg-bins must be a positive integer")

    input_dir = Path(args.input_dir)
    train_dir = Path(args.train_dir) if args.train_dir else input_dir
    test_dir = Path(args.test_dir) if args.test_dir else input_dir
    station = str(args.station)

    bkg_train = Path(args.bkg_train) if args.bkg_train else train_dir / f"vars_s{station}_bkg_train.root"
    sig_train = Path(args.sig_train) if args.sig_train else train_dir / f"vars_s{station}_sig_train.root"
    bkg_test = Path(args.bkg_test) if args.bkg_test else test_dir / f"vars_s{station}_bkg_test.root"
    sig_test = Path(args.sig_test) if args.sig_test else test_dir / f"vars_s{station}_sig_test.root"

    output = Path(args.output) if args.output else Path(f"BDT_variable_distributions_s{station}.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Using input files:")
    print(f"  bkg_train: {bkg_train}")
    print(f"  sig_train: {sig_train}")
    print(f"  bkg_test : {bkg_test}")
    print(f"  sig_test : {sig_test}")

    for path in [bkg_train, sig_train, bkg_test, sig_test]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    print("Reading ROOT files...")
    bkg_train_data, n_bkg_train = read_variables(bkg_train, args.bkg_tree)
    sig_train_data, n_sig_train = read_variables(sig_train, args.sig_tree)
    bkg_test_data, n_bkg_test = read_variables(bkg_test, args.bkg_tree)
    sig_test_data, n_sig_test = read_variables(sig_test, args.sig_tree)

    bkg_combined = combine_dicts(bkg_train_data, bkg_test_data)
    sig_combined = combine_dicts(sig_train_data, sig_test_data)

    print(f"Writing {output}")
    with PdfPages(output) as pdf:
        for variables, layout, figsize in [
            (PAGE_1_VARIABLES, (2, 4), (24, 10)),
            (PAGE_2_VARIABLES, (2, 3), (18, 10)),
        ]:
            plot_page(
                pdf,
                "Combined training + testing events",
                variables,
                layout,
                figsize,
                bkg_combined,
                sig_combined,
                n_bkg_train + n_bkg_test,
                n_sig_train + n_sig_test,
                args.bins,
                not args.linear_y,
                bkg_bins=args.bkg_bins,
            )

    print("Done.")
    print(f"Output PDF: {output.resolve()}")


if __name__ == "__main__":
    main()
