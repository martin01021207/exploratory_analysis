#!/usr/bin/env python3
"""Plot whole ROOT-file runs and highlight candidate events in a PDF.

The candidate JSON must have this form:

    {"648": [1383], "898": [1739], "963": [1846]}

By default, the output contains one page per candidate event. If a run has
multiple candidates, --combine-events-per-run puts them on a single page.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import uproot

# A non-interactive backend makes the script reliable on batch/cluster nodes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


ANALYSIS_VARIABLES = (
    "averageImpulsivity_PA",
    "coherentKurtosis_PA",
    "averageKurtosis_inIce",
    "averageEntropy_inIce",
    "averageImpulsivity_inIce",
    "coherentKurtosis_inIce",
    "coherentEntropy_inIce",
    "coherentImpulsivity_inIce",
)

IDENTIFIER_BRANCHES = (
    "station_number",
    "run_number",
    "event_number",
    "trigger_time",
)

REQUIRED_BRANCHES = IDENTIFIER_BRANCHES + ANALYSIS_VARIABLES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make a multipage PDF of candidate-event runs. Each page contains "
            "eight scatter panels sharing the trigger_time x-axis."
        )
    )
    parser.add_argument("root_file", type=Path, help="Input ROOT file")
    parser.add_argument("candidate_json", type=Path, help="Candidate-event JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PDF (default: <ROOT stem>_candidate_runs.pdf)",
    )
    parser.add_argument(
        "--tree",
        help=(
            "TTree/RNTuple name. If omitted, the script finds the unique object "
            "that contains all required branches."
        ),
    )
    parser.add_argument(
        "--combine-events-per-run",
        action="store_true",
        help="Use one page per run and mark all candidates in that run.",
    )
    parser.add_argument(
        "--relative-time",
        action="store_true",
        help="Plot trigger_time relative to the first finite trigger in each run.",
    )
    parser.add_argument(
        "--step-size",
        default="100 MB",
        help="ROOT chunk size passed to uproot (default: 100 MB).",
    )
    return parser.parse_args()


def load_candidates(path: Path) -> dict[int, list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Candidate JSON must be an object mapping runs to event lists.")

    candidates: dict[int, list[int]] = {}
    for raw_run, raw_events in raw.items():
        try:
            run = int(raw_run)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid run number in JSON: {raw_run!r}") from exc

        if not isinstance(raw_events, list):
            raise ValueError(f"Events for run {run} must be a JSON list.")

        try:
            events = [int(event) for event in raw_events]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Run {run} contains a non-integer event number.") from exc

        if events:
            candidates[run] = list(dict.fromkeys(events))

    if not candidates:
        raise ValueError("Candidate JSON contains no candidate events.")

    return dict(sorted(candidates.items()))


def branch_names(obj: object) -> set[str]:
    """Return branch/field names without ROOT cycle suffixes."""
    try:
        return {str(name).split(";")[0] for name in obj.keys()}  # type: ignore[attr-defined]
    except Exception:
        return set()


def find_data_object(root: uproot.ReadOnlyDirectory, requested_name: str | None):
    required = set(REQUIRED_BRANCHES)

    if requested_name:
        obj = root[requested_name]
        missing = required - branch_names(obj)
        if missing:
            raise KeyError(
                f"Object {requested_name!r} is missing branches: {sorted(missing)}"
            )
        return requested_name, obj

    matches: list[tuple[str, object]] = []
    for key in root.classnames(recursive=True):
        try:
            obj = root[key]
        except Exception:
            continue
        if required.issubset(branch_names(obj)) and hasattr(obj, "iterate"):
            matches.append((key, obj))

    if not matches:
        inspected = ", ".join(root.classnames(recursive=True).keys()) or "<none>"
        raise KeyError(
            "Could not find a TTree/RNTuple containing all required branches. "
            f"Objects inspected: {inspected}. Use --tree if needed."
        )
    if len(matches) > 1:
        names = ", ".join(name for name, _ in matches)
        raise ValueError(f"More than one matching data object was found: {names}. Use --tree.")

    return matches[0]


def chunk_as_mapping(chunk: object) -> dict[str, np.ndarray]:
    if isinstance(chunk, dict):
        return {str(key): np.asarray(value) for key, value in chunk.items()}
    if isinstance(chunk, np.ndarray) and chunk.dtype.names:
        return {name: np.asarray(chunk[name]) for name in chunk.dtype.names}
    raise TypeError(f"Unexpected array container returned by uproot: {type(chunk).__name__}")


def read_candidate_runs(tree: object, wanted_runs: set[int], step_size: str):
    pieces: dict[int, dict[str, list[np.ndarray]]] = {
        run: defaultdict(list) for run in wanted_runs
    }

    for raw_chunk in tree.iterate(  # type: ignore[attr-defined]
        expressions=list(REQUIRED_BRANCHES),
        library="np",
        step_size=step_size,
    ):
        chunk = chunk_as_mapping(raw_chunk)
        run_numbers = np.asarray(chunk["run_number"])
        if run_numbers.ndim != 1:
            raise ValueError("run_number must be a scalar branch (one value per entry).")

        present_runs = wanted_runs.intersection(int(run) for run in np.unique(run_numbers))
        for run in present_runs:
            mask = run_numbers == run
            for branch in REQUIRED_BRANCHES:
                values = np.asarray(chunk[branch])
                if values.ndim != 1:
                    raise ValueError(
                        f"{branch} must be a scalar branch; got shape {values.shape}."
                    )
                pieces[run][branch].append(values[mask])

    runs: dict[int, dict[str, np.ndarray]] = {}
    for run in sorted(wanted_runs):
        if not pieces[run]["run_number"]:
            continue
        runs[run] = {
            branch: np.concatenate(pieces[run][branch]) for branch in REQUIRED_BRANCHES
        }
    return runs


def validate_candidates(
    candidates: dict[int, list[int]], runs: dict[int, dict[str, np.ndarray]]
) -> None:
    problems: list[str] = []
    for run, candidate_events in candidates.items():
        if run not in runs:
            problems.append(f"run {run} was not found")
            continue
        event_numbers = np.asarray(runs[run]["event_number"], dtype=np.int64)
        for event in candidate_events:
            locations = np.flatnonzero(event_numbers == event)
            if locations.size == 0:
                problems.append(f"event {event} was not found in run {run}")
            elif locations.size > 1:
                warnings.warn(
                    f"Event {event} occurs {locations.size} times in run {run}; "
                    "all matching trigger times will be marked.",
                    stacklevel=2,
                )

    if problems:
        raise ValueError("Candidate validation failed: " + "; ".join(problems))


def station_label(station_values: np.ndarray) -> str:
    finite = np.asarray(station_values)
    unique = np.unique(finite)
    if unique.size == 1:
        value = unique[0]
        try:
            return str(int(value)) if float(value).is_integer() else str(value)
        except (TypeError, ValueError):
            return str(value)
    return "/".join(str(value) for value in unique)


def page_specs(
    candidates: dict[int, list[int]], combine_events_per_run: bool
) -> list[tuple[int, list[int]]]:
    if combine_events_per_run:
        return [(run, events) for run, events in candidates.items()]
    return [(run, [event]) for run, events in candidates.items() for event in events]


def make_page(
    run: int,
    highlighted_events: list[int],
    run_data: dict[str, np.ndarray],
    relative_time: bool,
) -> plt.Figure:
    trigger_time = np.asarray(run_data["trigger_time"], dtype=float)
    event_number = np.asarray(run_data["event_number"], dtype=np.int64)

    finite_time = np.isfinite(trigger_time)
    if not np.any(finite_time):
        raise ValueError(f"Run {run} has no finite trigger_time values.")

    order = np.argsort(trigger_time, kind="stable")
    trigger_time = trigger_time[order]
    event_number = event_number[order]
    sorted_variables = {
        variable: np.asarray(run_data[variable], dtype=float)[order]
        for variable in ANALYSIS_VARIABLES
    }

    offset = float(np.nanmin(trigger_time)) if relative_time else 0.0
    x_values = trigger_time - offset
    x_label = "trigger_time"
    if relative_time:
        x_label = f"trigger_time - {offset:.9g}"

    colors = plt.get_cmap("tab10").colors
    marker_info: list[tuple[int, np.ndarray, tuple[float, float, float]]] = []
    for index, event in enumerate(highlighted_events):
        matches = event_number == event
        marker_info.append((event, matches, colors[index % len(colors)]))

    fig, axes = plt.subplots(
        len(ANALYSIS_VARIABLES),
        1,
        figsize=(11.0, 16.0),
        sharex=True,
    )
    fig.patch.set_facecolor("white")

    for ax, variable in zip(axes, ANALYSIS_VARIABLES):
        y_values = sorted_variables[variable]
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        ax.scatter(
            x_values[valid],
            y_values[valid],
            s=11,
            color="#356FA1",
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )

        for event, matches, color in marker_info:
            for x_at_event in np.unique(x_values[matches & np.isfinite(x_values)]):
                ax.axvline(x_at_event, color=color, linestyle="--", linewidth=1.4, alpha=0.95)
            candidate_points = matches & np.isfinite(x_values) & np.isfinite(y_values)
            ax.scatter(
                x_values[candidate_points],
                y_values[candidate_points],
                s=48,
                marker="*",
                color=color,
                edgecolor="black",
                linewidth=0.35,
                zorder=4,
            )

        ax.set_ylabel(variable, fontsize=9)
        ax.grid(True, color="#D5DCE3", linewidth=0.55, alpha=0.8)
        ax.tick_params(axis="both", labelsize=8)
        ax.margins(x=0.015)

    axes[-1].set_xlabel(x_label, fontsize=10)

    station = station_label(run_data["station_number"])
    event_text = ", ".join(str(event) for event in highlighted_events)
    noun = "event" if len(highlighted_events) == 1 else "events"
    fig.suptitle(
        f"Station {station} - Run {run} - Candidate {noun} {event_text}",
        fontsize=15,
        fontweight="bold",
        y=0.992,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle="--",
            marker="*",
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=f"Candidate event {event}",
        )
        for event, _, color in marker_info
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.974),
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.985,
        0.006,
        f"{len(trigger_time):,} entries in whole run",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.21, right=0.97, bottom=0.045, top=0.948, hspace=0.10)
    return fig


def main() -> int:
    args = parse_args()
    root_path = args.root_file.expanduser().resolve()
    json_path = args.candidate_json.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else root_path.with_name(f"{root_path.stem}_candidate_runs.pdf")
    )

    candidates = load_candidates(json_path)
    with uproot.open(root_path) as root:
        object_name, tree = find_data_object(root, args.tree)
        print(f"Reading {object_name!r} from {root_path.name} ...")
        runs = read_candidate_runs(tree, set(candidates), args.step_size)

    validate_candidates(candidates, runs)
    specs = page_specs(candidates, args.combine_events_per_run)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Candidate-event whole-run diagnostic plots"
        metadata["Subject"] = "Eight analysis variables versus trigger_time"
        metadata["Keywords"] = "ROOT, candidate events, trigger_time"

        for page_number, (run, events) in enumerate(specs, start=1):
            print(
                f"Plotting page {page_number}/{len(specs)}: "
                f"run {run}, event(s) {events}"
            )
            figure = make_page(run, events, runs[run], args.relative_time)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    print(f"Saved {len(specs)} page(s) to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, uproot.exceptions.KeyInFileError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
