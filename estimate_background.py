import argparse
import math
import numpy as np
import json
import os
import re
from collections import defaultdict
from array import array
import ROOT
from ROOT import TFile, TTree, TCanvas, TH1F, TLegend, TLine, TF1, TGraphErrors
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
#from scipy.stats import poisson


def _normalized_branch_name(name):
    """Return a case/underscore-insensitive name for branch matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _resolve_tree_leaf(tree, candidates, value_label):
    """Find a scalar leaf while tolerating common branch naming conventions."""
    available = []
    for leaf in tree.GetListOfLeaves():
        leaf_name = leaf.GetName()
        if leaf_name not in available:
            available.append(leaf_name)

    # Prefer an exact match, then compare names without case or punctuation.
    for candidate in candidates:
        leaf = tree.GetLeaf(candidate)
        if leaf:
            return candidate, leaf

    normalized_available = {
        _normalized_branch_name(name): name for name in available
    }
    for candidate in candidates:
        matched_name = normalized_available.get(
            _normalized_branch_name(candidate)
        )
        if matched_name is not None:
            return matched_name, tree.GetLeaf(matched_name)

    raise RuntimeError(
        f"Cannot find a {value_label} branch in tree '{tree.GetName()}'. "
        f"Tried {candidates}. Available leaves: {available}"
    )


def _integer_leaf_value(leaf):
    """Read an integer leaf without routing 64-bit identifiers through float."""
    if hasattr(leaf, "GetValueLong64"):
        return int(leaf.GetValueLong64())
    return int(leaf.GetValue())


def find_outlier_bins(hist, x_low, x_high, prediction):
    """Find bins whose observed count is above the central fit prediction."""
    outlier_bins = set()
    for bin_index in range(1, hist.GetNbinsX() + 1):
        bin_center = hist.GetXaxis().GetBinCenter(bin_index)
        if bin_center < x_low or bin_center > x_high:
            continue

        observed = float(hist.GetBinContent(bin_index))
        expected = max(0.0, float(prediction(bin_center)))
        if observed > expected:
            outlier_bins.add(bin_index)

    return outlier_bins


def exponential_bin_prediction(
    bin_low,
    bin_high,
    A,
    B,
    covariance,
    x_anchor,
    bin_width,
    scale,
):
    """Return a bin prediction, its gradient, and its fit uncertainty."""
    low_offset = bin_low - x_anchor
    high_offset = bin_high - x_anchor

    if abs(B) < 1.0e-10:
        interval = bin_high - bin_low
        prediction = scale * A * interval / bin_width
        gradient = np.array([
            scale * interval / bin_width,
            -scale * A * (
                high_offset**2 - low_offset**2
            ) / (2.0 * bin_width),
        ])
    else:
        exp_low = np.exp(-B * low_offset)
        exp_high = np.exp(-B * high_offset)
        difference = exp_low - exp_high
        difference_derivative = (
            -low_offset * exp_low + high_offset * exp_high
        )
        prediction = scale * A * difference / (B * bin_width)
        gradient = np.array([
            scale * difference / (B * bin_width),
            scale * A / bin_width * (
                difference_derivative / B - difference / B**2
            ),
        ])

    variance = float(gradient @ covariance @ gradient)
    uncertainty = np.sqrt(max(variance, 0.0))
    return prediction, gradient, uncertainty


def exponential_predictions_for_bins(
    hist,
    bin_indices,
    score_minimum,
    score_maximum,
    A,
    B,
    covariance,
    x_anchor,
    bin_width,
    scale,
):
    """Predict selected-bin yields while retaining fit correlations."""
    predictions = {}
    total_prediction = 0.0
    total_gradient = np.zeros(2)

    for bin_index in sorted(bin_indices):
        bin_low = max(
            score_minimum,
            hist.GetXaxis().GetBinLowEdge(bin_index),
        )
        bin_high = min(
            score_maximum,
            hist.GetXaxis().GetBinUpEdge(bin_index),
        )
        if bin_high <= bin_low:
            continue

        prediction, gradient, uncertainty = (
            exponential_bin_prediction(
                bin_low,
                bin_high,
                A,
                B,
                covariance,
                x_anchor,
                bin_width,
                scale,
            )
        )
        predictions[bin_index] = {
            "yield": prediction,
            "uncertainty": uncertainty,
        }
        total_prediction += prediction
        total_gradient += gradient

    total_variance = float(
        total_gradient @ covariance @ total_gradient
    )
    total_uncertainty = np.sqrt(max(total_variance, 0.0))
    return predictions, total_prediction, total_uncertainty


def gaussian_interval_prediction(
    interval_low,
    interval_high,
    amplitude,
    mean,
    sigma,
    covariance,
    bin_width,
):
    """Return a Gaussian yield and covariance-propagated uncertainty."""
    if interval_high <= interval_low or sigma <= 0.0:
        return 0.0, 0.0

    sqrt_two = np.sqrt(2.0)
    z_low = (interval_low - mean) / (sqrt_two * sigma)
    z_high = (interval_high - mean) / (sqrt_two * sigma)

    # erfc avoids cancellation when both integration limits are far into the
    # same Gaussian tail, as is often the case above the selected BDT cut.
    if z_low > 0.0:
        erf_difference = math.erfc(z_low) - math.erfc(z_high)
    elif z_high < 0.0:
        erf_difference = (
            math.erfc(-z_high) - math.erfc(-z_low)
        )
    else:
        erf_difference = math.erf(z_high) - math.erf(z_low)

    normalization = sigma * np.sqrt(np.pi / 2.0) * erf_difference
    integral = amplitude * normalization

    exp_low = np.exp(-0.5 * ((interval_low - mean) / sigma)**2)
    exp_high = np.exp(-0.5 * ((interval_high - mean) / sigma)**2)
    gradient = np.array([
        normalization,
        amplitude * (exp_low - exp_high),
        integral / sigma - amplitude * (
            ((interval_high - mean) / sigma) * exp_high
            - ((interval_low - mean) / sigma) * exp_low
        ),
    ]) / bin_width

    predicted_yield = integral / bin_width
    variance = float(gradient @ covariance @ gradient)
    uncertainty = np.sqrt(max(variance, 0.0))
    return predicted_yield, uncertainty


def fit_covariance_matrix(fit_result, fit_function, n_parameters):
    """Extract a fit covariance matrix with a diagonal-only fallback."""
    try:
        return np.array([
            [
                float(fit_result.CovMatrix(i, j))
                for j in range(n_parameters)
            ]
            for i in range(n_parameters)
        ])
    except (AttributeError, TypeError):
        return np.diag([
            fit_function.GetParError(i)**2
            for i in range(n_parameters)
        ])


def poisson_deviance(hist, fit_function, first_bin, last_bin):
    """Calculate Poisson deviance for a fitted histogram interval."""
    deviance = 0.0
    for bin_index in range(first_bin, last_bin + 1):
        observed = max(0.0, float(hist.GetBinContent(bin_index)))
        bin_center = hist.GetXaxis().GetBinCenter(bin_index)
        expected = max(1.0e-12, float(fit_function.Eval(bin_center)))
        if observed > 0.0:
            deviance += 2.0 * (
                expected - observed
                + observed * np.log(observed / expected)
            )
        else:
            deviance += 2.0 * expected
    return deviance


def scan_exponential_fit_start(
    hist,
    fit_high,
    prediction_low,
    prediction_high,
    min_bins,
    min_events,
    consecutive_fits,
    stability_sigma,
    max_reduced_deviance,
):
    """Select the earliest stable exponential-fit starting bin."""
    peak_bin = hist.GetMaximumBin()
    last_fit_bin = hist.FindFixBin(np.nextafter(fit_high, -np.inf))
    last_fit_bin = min(last_fit_bin, hist.GetNbinsX())
    last_start_bin = last_fit_bin - min_bins + 1
    bin_width = hist.GetBinWidth(peak_bin)
    scan_results = []

    for first_bin in range(peak_bin + 1, last_start_bin + 1):
        n_fit_bins = last_fit_bin - first_bin + 1
        n_fit_events = float(hist.Integral(first_bin, last_fit_bin))
        if n_fit_bins < min_bins or n_fit_events < min_events:
            continue

        fit_low = hist.GetXaxis().GetBinCenter(first_bin)
        formula = f"[0]*exp(-[1]*(x-{fit_low}))"
        candidate_fit = TF1(
            f"fit_start_candidate_{first_bin}",
            formula,
            fit_low,
            fit_high,
        )

        amplitude_seed = max(float(hist.GetBinContent(first_bin)), 1.0)
        final_positive_bin = None
        for bin_index in range(last_fit_bin, first_bin, -1):
            if hist.GetBinContent(bin_index) > 0.0:
                final_positive_bin = bin_index
                break
        slope_seed = 10.0
        if final_positive_bin is not None:
            final_content = max(
                float(hist.GetBinContent(final_positive_bin)), 1.0
            )
            final_center = hist.GetXaxis().GetBinCenter(
                final_positive_bin
            )
            if final_center > fit_low and amplitude_seed > final_content:
                slope_seed = np.log(
                    amplitude_seed / final_content
                ) / (final_center - fit_low)
        slope_seed = min(max(slope_seed, 1.0e-3), 1.0e3)

        candidate_fit.SetParameters(amplitude_seed, slope_seed)
        candidate_fit.SetParLimits(
            0,
            1.0e-12,
            max(1.0e6, 100.0 * hist.GetMaximum()),
        )
        candidate_fit.SetParLimits(1, 1.0e-6, 1.0e3)
        fit_result = hist.Fit(candidate_fit, "RQLS0N")
        fit_status = int(fit_result)
        if fit_status != 0:
            continue

        parameters = np.array([
            candidate_fit.GetParameter(i) for i in range(2)
        ])
        covariance = fit_covariance_matrix(
            fit_result, candidate_fit, 2
        )
        parameter_errors = np.sqrt(
            np.maximum(np.diag(covariance), 0.0)
        )
        if not (
            np.all(np.isfinite(parameters))
            and np.all(np.isfinite(parameter_errors))
            and parameters[0] > 0.0
            and parameters[1] > 0.0
        ):
            continue

        tail_prediction, _, tail_uncertainty = (
            exponential_bin_prediction(
                prediction_low,
                prediction_high,
                parameters[0],
                parameters[1],
                covariance,
                fit_low,
                bin_width,
                1.0,
            )
        )
        deviance = poisson_deviance(
            hist, candidate_fit, first_bin, last_fit_bin
        )
        ndf = max(n_fit_bins - 2, 1)
        reduced_deviance = deviance / ndf

        scan_results.append({
            "bin": first_bin,
            "x": fit_low,
            "events": n_fit_events,
            "fit_bins": n_fit_bins,
            "amplitude": parameters[0],
            "amplitude_error": parameter_errors[0],
            "slope": parameters[1],
            "slope_error": parameter_errors[1],
            "tail": tail_prediction,
            "tail_error": tail_uncertainty,
            "deviance": deviance,
            "ndf": ndf,
            "reduced_deviance": reduced_deviance,
            "acceptable": reduced_deviance <= max_reduced_deviance,
        })

    if not scan_results:
        raise RuntimeError(
            "No exponential fit-start candidates passed the minimum-bin, "
            "minimum-event, and fit-status requirements."
        )

    def statistically_stable(left, right, value_key, error_key):
        combined_error = np.hypot(
            left[error_key], right[error_key]
        )
        difference = abs(left[value_key] - right[value_key])
        if combined_error <= 0.0:
            return difference <= 1.0e-12
        return difference <= stability_sigma * combined_error

    selected = None
    selected_plateau = []
    selection_reason = None
    for start_index in range(
        len(scan_results) - consecutive_fits + 1
    ):
        window = scan_results[
            start_index:start_index + consecutive_fits
        ]
        if any(not result["acceptable"] for result in window):
            continue
        if any(
            window[offset]["bin"] != window[0]["bin"] + offset
            for offset in range(consecutive_fits)
        ):
            continue

        stable_window = all(
            statistically_stable(left, right, "slope", "slope_error")
            and statistically_stable(
                left, right, "tail", "tail_error"
            )
            for left, right in zip(window[:-1], window[1:])
        )
        if stable_window:
            selected = window[0]
            selected_plateau = window
            selection_reason = (
                f"earliest {consecutive_fits}-bin stable plateau"
            )
            break

    if selected is None:
        acceptable_results = [
            result for result in scan_results if result["acceptable"]
        ]
        fallback_pool = acceptable_results or scan_results
        selected = min(
            fallback_pool,
            key=lambda result: (
                result["reduced_deviance"], result["bin"]
            ),
        )
        selected_plateau = acceptable_results or [selected]
        selection_reason = (
            "fallback to the candidate with the lowest reduced Poisson "
            "deviance because no stable plateau was found"
        )

    selected_plateau_bins = {
        result["bin"] for result in selected_plateau
    }
    for result in scan_results:
        result["in_uncertainty_set"] = (
            result["bin"] in selected_plateau_bins
        )

    return scan_results, selected, selection_reason


def save_fit_start_scan_plot(
    output_path,
    hist,
    fit_high,
    scan_results,
    selected,
    selection_reason,
    target_cut,
    max_reduced_deviance,
    station_number,
):
    """Save the fit-start stability diagnostics as a one-page PDF."""
    x_values = np.array([result["x"] for result in scan_results])
    slopes = np.array([result["slope"] for result in scan_results])
    slope_errors = np.array([
        result["slope_error"] for result in scan_results
    ])
    tails = np.array([result["tail"] for result in scan_results])
    tail_errors = np.array([
        result["tail_error"] for result in scan_results
    ])
    reduced_deviances = np.array([
        result["reduced_deviance"] for result in scan_results
    ])
    acceptable = np.array([
        result["acceptable"] for result in scan_results
    ])
    uncertainty_set = np.array([
        result["in_uncertainty_set"] for result in scan_results
    ])

    figure, axes_grid = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes_grid.ravel()
    figure.suptitle(
        f"Exponential fit-start stability scan (S{station_number})",
        fontsize=15,
    )

    axes[0].errorbar(
        x_values,
        slopes,
        yerr=slope_errors,
        fmt="o",
        markersize=4,
        color="0.55",
        ecolor="0.75",
        capsize=2,
        label="Successful candidate fits",
    )
    axes[0].scatter(
        x_values[acceptable],
        slopes[acceptable],
        s=28,
        color="tab:blue",
        label="Acceptable Poisson deviance",
        zorder=3,
    )
    axes[0].scatter(
        x_values[uncertainty_set],
        slopes[uncertainty_set],
        s=55,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.5,
        label="Fit-range uncertainty set",
        zorder=4,
    )
    axes[0].set_ylabel("Exponential slope B")
    axes[0].legend(fontsize=9)

    axes[1].errorbar(
        x_values,
        tails,
        yerr=tail_errors,
        fmt="o",
        markersize=4,
        color="tab:purple",
        ecolor="plum",
        capsize=2,
    )
    axes[1].scatter(
        x_values[uncertainty_set],
        tails[uncertainty_set],
        s=55,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.5,
        zorder=4,
    )
    axes[1].set_ylabel(
        f"Predicted events\nabove BDT = {target_cut:.3f}"
    )

    axes[2].plot(
        x_values,
        reduced_deviances,
        "o-",
        markersize=4,
        color="tab:green",
    )
    axes[2].axhline(
        max_reduced_deviance,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=(
            f"Acceptance limit = {max_reduced_deviance:.2f}"
        ),
    )
    axes[2].set_ylabel("Poisson deviance / ndf")
    axes[2].set_xlabel("Candidate fit starting score")
    axes[2].legend(fontsize=9)

    for axis in axes[:3]:
        axis.axvline(
            selected["x"],
            color="tab:red",
            linestyle="--",
            linewidth=1.8,
        )
        axis.grid(True, alpha=0.25)

    last_fit_bin = hist.FindFixBin(np.nextafter(fit_high, -np.inf))
    last_fit_bin = min(last_fit_bin, hist.GetNbinsX())
    residual_x = []
    residual_pulls = []
    for bin_index in range(selected["bin"], last_fit_bin + 1):
        bin_center = hist.GetXaxis().GetBinCenter(bin_index)
        observed = float(hist.GetBinContent(bin_index))
        expected = selected["amplitude"] * np.exp(
            -selected["slope"] * (bin_center - selected["x"])
        )
        expected = max(expected, 1.0e-12)
        residual_x.append(bin_center)
        residual_pulls.append((observed - expected) / np.sqrt(expected))

    axes[3].axhspan(-2.0, 2.0, color="0.92", zorder=0)
    axes[3].axhline(0.0, color="black", linewidth=1.0)
    axes[3].plot(
        residual_x,
        residual_pulls,
        "o-",
        markersize=4,
        color="tab:orange",
    )
    axes[3].set_xlabel("BDT score in selected fit range")
    axes[3].set_ylabel("Poisson pull")
    axes[3].set_title("Selected-fit residuals")
    axes[3].grid(True, alpha=0.25)

    axes[0].set_title(
        f"Selected x_min = {selected['x']:.4f}: {selection_reason}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def collect_outlier_events(
    tree, hist, outlier_bins, score_branch, minimum_score
):
    """Collect IDs/scores above the threshold and in selected outlier bins."""
    score_name, score_leaf = _resolve_tree_leaf(
        tree, [score_branch], "BDT-score"
    )
    run_name, run_leaf = _resolve_tree_leaf(
        tree,
        [
            "run_number", "runNumber", "run_num", "runNum",
            "run_id", "runID", "run"
        ],
        "run-number",
    )
    event_name, event_leaf = _resolve_tree_leaf(
        tree,
        [
            "event_number", "eventNumber", "event_num", "eventNum",
            "event_id", "eventID", "event"
        ],
        "event-number",
    )

    grouped_events = defaultdict(set)
    outlier_scores = []
    for entry_index in range(tree.GetEntries()):
        tree.GetEntry(entry_index)
        score = float(score_leaf.GetValue())
        if score <= minimum_score:
            continue

        bin_index = hist.FindFixBin(score)
        if bin_index not in outlier_bins:
            continue

        run_number = _integer_leaf_value(run_leaf)
        event_number = _integer_leaf_value(event_leaf)
        grouped_events[run_number].add(event_number)
        outlier_scores.append(score)

    branch_names = {
        "score": score_name,
        "run": run_name,
        "event": event_name,
    }
    return grouped_events, outlier_scores, branch_names


def save_event_json(output_path, grouped_events):
    """Write exactly {\"run_number\": [event_number]} with stable ordering."""
    serializable = {
        str(run_number): sorted(grouped_events[run_number])
        for run_number in sorted(grouped_events)
    }
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(serializable, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    # This script only writes files, so do not require a display server.
    ROOT.gROOT.SetBatch(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("stationNumber", type=str)
    parser.add_argument('file_in', type=str, help="Path to the input file")
    parser.add_argument('dir_out', type=str, help="Output directory")
    parser.add_argument('--target_cut', type=float, default=0.45, help="Target BDT cut")
    parser.add_argument('--full_data_file', type=str, default=None, help="Path to the input file")
    parser.add_argument(
        '--outlier_min_score',
        type=float,
        default=0.06,
        help=(
            "Only investigate outliers with BDT scores strictly greater "
            "than this value (default: 0.06)"
        ),
    )
    parser.add_argument(
        '--fit_start_consecutive',
        type=int,
        default=4,
        help=(
            "Number of consecutive stable starting bins required for the "
            "automatic exponential fit-start selection (default: 4)"
        ),
    )
    parser.add_argument(
        '--fit_start_stability_sigma',
        type=float,
        default=1.0,
        help=(
            "Maximum adjacent-fit change in combined standard deviations "
            "for the slope and tail prediction (default: 1.0)"
        ),
    )
    parser.add_argument(
        '--fit_start_max_reduced_deviance',
        type=float,
        default=2.0,
        help=(
            "Maximum Poisson deviance/ndf for an acceptable candidate fit "
            "(default: 2.0)"
        ),
    )
    parser.add_argument(
        '--fit_start_min_bins',
        type=int,
        default=8,
        help="Minimum number of bins in each candidate fit (default: 8)",
    )
    parser.add_argument(
        '--fit_start_min_events',
        type=float,
        default=50.0,
        help=(
            "Minimum burn-sample events in each candidate fit range "
            "(default: 50)"
        ),
    )
    args = parser.parse_args()

    stationNumber = args.stationNumber
    path_to_file_in = args.file_in
    dir_out = args.dir_out
    if not dir_out.endswith("/"):
        dir_out += "/"

    cut = args.target_cut
    full_data_file = args.full_data_file
    outlier_min_score = args.outlier_min_score
    fit_start_consecutive = args.fit_start_consecutive
    fit_start_stability_sigma = args.fit_start_stability_sigma
    fit_start_max_reduced_deviance = (
        args.fit_start_max_reduced_deviance
    )
    fit_start_min_bins = args.fit_start_min_bins
    fit_start_min_events = args.fit_start_min_events

    if fit_start_consecutive < 2:
        raise ValueError("--fit_start_consecutive must be at least 2")
    if fit_start_stability_sigma <= 0.0:
        raise ValueError("--fit_start_stability_sigma must be positive")
    if fit_start_max_reduced_deviance <= 0.0:
        raise ValueError(
            "--fit_start_max_reduced_deviance must be positive"
        )
    if fit_start_min_bins < 3:
        raise ValueError("--fit_start_min_bins must be at least 3")
    if fit_start_min_events <= 0.0:
        raise ValueError("--fit_start_min_events must be positive")

    method = "BDTD"

    whole_test_factor = 0.93
    burn_test_factor = 0.03
    scale_factor = whole_test_factor / burn_test_factor

    filename_in = path_to_file_in.split(".root")[0].split("testTree_")[1]
    graphFilename_expMC = f"expMC_{filename_in}.pdf"
    graphFilename = f"fittedTestedResults_{filename_in}.pdf"
    graphFilename_fit_start = f"fitStartScan_{filename_in}.pdf"

    input = TFile.Open(path_to_file_in)
    tree_S = input.Get("TestTree_S")
    tree_B = input.Get("TestTree_B")
    nEvents_S = tree_S.GetEntries()
    nEvents_B = tree_B.GetEntries()

    if full_data_file:
        input_full = TFile.Open(full_data_file)
        if not input_full or input_full.IsZombie():
            raise RuntimeError(f"Cannot open full-data file: {full_data_file}")
        tree_B_full = input_full.Get("TestTree_B")
        if not tree_B_full:
            raise RuntimeError(
                f"Cannot find TestTree_B in full-data file: {full_data_file}"
            )
        nEvents_B_full = tree_B_full.GetEntries()

    histTitle = f"TMVA response for classifier: {method} (S{stationNumber})"
    canvas = TCanvas("c1", histTitle, 10, 10, 850, 500)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gPad.SetLogy(1)

    nbin = 100
    xMin = -0.45
    xMax = 1.0

    if not xMin <= outlier_min_score < xMax:
        raise ValueError(
            f"--outlier_min_score must be in [{xMin}, {xMax}); "
            f"received {outlier_min_score}"
        )

    hist_S = TH1F("hist_S", histTitle, nbin, xMin, xMax)
    hist_S.GetXaxis().SetTitle(f"{method} response")
    hist_S.GetYaxis().SetTitle("Events")
    hist_S.SetLineColorAlpha(ROOT.kAzure+2, 0.5)
    hist_S.SetLineWidth(3)
    hist_S.SetFillColorAlpha(ROOT.kAzure-7, 0.2)
    hist_B = TH1F("hist_B", histTitle, nbin, xMin, xMax)
    hist_B.GetXaxis().SetTitle(f"{method} response")
    hist_B.GetYaxis().SetTitle("Events")
    hist_B.SetLineColor(ROOT.kRed+1)
    hist_B.SetLineWidth(3)
    hist_B.SetFillColor(ROOT.kRed+1)
    hist_B.SetFillStyle(3354)
    hist_B_full = TH1F("hist_B_full", histTitle + "[full data]", nbin, xMin, xMax)
    hist_B_full.SetLineColor(ROOT.kRed+1)
    hist_B_full.SetLineWidth(3)
    hist_B_full.SetFillColor(ROOT.kRed+1)
    hist_B_full.SetFillStyle(3354)

    for i_event in range(nEvents_S):
        tree_S.GetEntry(i_event)
        MVA = tree_S.BDTD
        hist_S.Fill(MVA)

    for i_event in range(nEvents_B):
        tree_B.GetEntry(i_event)
        MVA = tree_B.BDTD
        hist_B.Fill(MVA)

    if full_data_file:
        for i_event in range(nEvents_B_full):
            tree_B_full.GetEntry(i_event)
            MVA = tree_B_full.BDTD
            hist_B_full.Fill(MVA)

    canvas.cd()
    hist_B.Draw()
    hist_S.Draw("same")

    cutLine = TLine(cut, 0, cut, hist_B.GetMaximum()*1.5)
    cutLine.SetLineStyle(2)
    cutLine.SetLineWidth(2)
    cutLine.Draw("same")

    x2 = 0.35
    (
        fit_start_scan_results,
        selected_fit_start,
        fit_start_selection_reason,
    ) = scan_exponential_fit_start(
        hist_B,
        x2,
        cut,
        xMax,
        fit_start_min_bins,
        fit_start_min_events,
        fit_start_consecutive,
        fit_start_stability_sigma,
        fit_start_max_reduced_deviance,
    )
    bin1 = selected_fit_start["bin"]
    x1 = hist_B.GetXaxis().GetBinCenter(bin1)
    bin_width = hist_B.GetBinWidth(hist_B.FindFixBin(x1))

    fit_start_scan_path = os.path.join(
        dir_out, graphFilename_fit_start
    )
    save_fit_start_scan_plot(
        fit_start_scan_path,
        hist_B,
        x2,
        fit_start_scan_results,
        selected_fit_start,
        fit_start_selection_reason,
        cut,
        fit_start_max_reduced_deviance,
        stationNumber,
    )

    print("\nAutomatic exponential fit-start scan:")
    print(f"Successful candidates: {len(fit_start_scan_results)}")
    print(
        f"Requirements: at least {fit_start_min_bins} bins, "
        f"{fit_start_min_events:g} events, "
        f"deviance/ndf <= {fit_start_max_reduced_deviance:g}, and "
        f"{fit_start_consecutive} consecutive fits stable within "
        f"{fit_start_stability_sigma:g} sigma"
    )
    print(
        f"Selected starting bin: {bin1} "
        f"(x_min = {x1:.6f})"
    )
    print(f"Selection reason: {fit_start_selection_reason}")
    if fit_start_selection_reason.startswith("fallback"):
        print(
            "WARNING: no stable fit-start plateau satisfied all default "
            "criteria. Inspect the diagnostic PDF before using this fit."
        )
    print(
        f"Selected candidate deviance/ndf: "
        f"{selected_fit_start['reduced_deviance']:.3f}"
    )
    print(f"Saved fit-start diagnostics: {fit_start_scan_path}")

    formula = f"[0]*exp(-[1]*(x-{x1}))"
    fitLine = TF1("fitLine", formula, x1, x2)
    fitLine.SetParameters(
        selected_fit_start["amplitude"],
        selected_fit_start["slope"],
    )
    fitLine.SetParLimits(
        0,
        1.0e-12,
        max(1.0e6, 100.0 * hist_B.GetMaximum()),
    )
    fitLine.SetParLimits(1, 1.0e-6, 1.0e3)
    fit = hist_B.Fit(fitLine, "RQLS")
    fit_status = int(fit)
    if fit_status != 0:
        raise RuntimeError(
            f"Final exponential fit failed with ROOT status {fit_status}"
        )
    fitLine.SetLineStyle(9)
    fitLine.SetLineWidth(3)
    fitLine.SetLineColor(4)

    popt = np.array([
        fitLine.GetParameter(parameter_index)
        for parameter_index in range(2)
    ])
    pcov = fit_covariance_matrix(fit, fitLine, 2)
    nEvents_tail, _, tail_error = exponential_bin_prediction(
        cut,
        xMax,
        popt[0],
        popt[1],
        pcov,
        x1,
        bin_width,
        1.0,
    )
    selected_plateau_tails = [
        result["tail"]
        for result in fit_start_scan_results
        if result["in_uncertainty_set"]
    ]
    fit_start_tail_systematic = max(
        (
            abs(candidate_tail - nEvents_tail)
            for candidate_tail in selected_plateau_tails
        ),
        default=0.0,
    )
    print(f"Number of Events in Tail: {nEvents_tail} +/- {tail_error}")
    print(
        f"Fit-start range systematic: +/- "
        f"{fit_start_tail_systematic:.3f}"
    )
    print(
        f"Initial Fit: A = {popt[0]:.2f}, B = {popt[1]:.2f}, "
        f"x_min = {x1}"
    )
    print("Covariance matrix:")
    print(pcov)

    A_burn, B_burn = popt
    fitLine_full = TF1(
        "fitLine_full",
        formula,
        x1,
        xMax
    )

    # Scale only the normalization. Do not refit.
    fitLine_full.SetParameter(0, scale_factor * A_burn)
    fitLine_full.SetParameter(1, B_burn)
    fitLine_full.SetLineStyle(9)
    fitLine_full.SetLineWidth(3)
    fitLine_full.SetLineColor(ROOT.kBlue)


    x_data = np.arange(cut, 1, bin_width)

    def exp_func(x, A, B):
        return A * np.exp(-B*(x-x1))

    def exp_uncertainty(x, A, B, cov):
        exp_term = np.exp(-B * (x - x1))
        dfdA = exp_term
        dfdB = -A * (x - x1) * exp_term
        J = np.array([dfdA, dfdB])
        sigma2 = J @ cov @ J
        return np.sqrt(max(sigma2, 0.0))

    # A bin is an outlier bin when its observed background count is above the
    # central exponential prediction at the bin center. Only the right tail
    # above outlier_min_score is investigated. The fit parameters always come
    # from the burn sample; the full-data prediction changes only by the
    # already-defined exposure scale factor.
    outlier_search_min = max(x1, outlier_min_score)
    burn_outlier_bins = find_outlier_bins(
        hist_B,
        outlier_search_min,
        xMax,
        lambda score: exp_func(score, *popt),
    )
    burn_outlier_events, burn_outlier_scores, burn_branches = (
        collect_outlier_events(
            tree_B,
            hist_B,
            burn_outlier_bins,
            method,
            outlier_min_score,
        )
    )
    burn_json_path = os.path.join(
        dir_out, f"burn_outlier_events_{filename_in}.json"
    )
    save_event_json(burn_json_path, burn_outlier_events)
    print("\nBurn-sample outliers:")
    print(f"Selection: {method} > {outlier_min_score:.3f}")
    print(f"Outlier bins: {len(burn_outlier_bins)}")
    print(f"Outlier events: {len(burn_outlier_scores)}")
    print(f"Branches used: {burn_branches}")
    print(f"Saved event IDs: {burn_json_path}")

    full_outlier_bins = set()
    full_outlier_events = defaultdict(set)
    full_outlier_scores = []
    full_expected_by_bin = {}
    full_expected_outliers = 0.0
    full_expected_outliers_error = 0.0
    full_excess_outliers = 0.0
    full_excess_outliers_error = 0.0
    if full_data_file:
        full_outlier_bins = find_outlier_bins(
            hist_B_full,
            outlier_search_min,
            xMax,
            lambda score: scale_factor * exp_func(score, *popt),
        )
        full_outlier_events, full_outlier_scores, full_branches = (
            collect_outlier_events(
                tree_B_full,
                hist_B_full,
                full_outlier_bins,
                method,
                outlier_min_score,
            )
        )
        (
            full_expected_by_bin,
            full_expected_outliers,
            full_expected_outliers_error,
        ) = exponential_predictions_for_bins(
            hist_B_full,
            full_outlier_bins,
            outlier_min_score,
            xMax,
            popt[0],
            popt[1],
            pcov,
            x1,
            bin_width,
            scale_factor,
        )
        full_observed_outliers = len(full_outlier_scores)
        full_excess_outliers = (
            full_observed_outliers - full_expected_outliers
        )
        # Combine observed Poisson uncertainty with the correlated fit error.
        full_excess_outliers_error = np.sqrt(
            full_observed_outliers
            + full_expected_outliers_error**2
        )
        full_json_path = os.path.join(
            dir_out, f"full_data_outlier_events_{filename_in}.json"
        )
        save_event_json(full_json_path, full_outlier_events)
        print("\nFull-data outliers:")
        print(f"Selection: {method} > {outlier_min_score:.3f}")
        print(f"Outlier bins: {len(full_outlier_bins)}")
        print(f"Observed events in outlier bins: {full_observed_outliers}")
        print(
            f"Fit-estimated events in outlier bins: "
            f"{full_expected_outliers:.3f} +/- "
            f"{full_expected_outliers_error:.3f}"
        )
        print(
            f"Background-subtracted excess: "
            f"{full_excess_outliers:.3f} +/- "
            f"{full_excess_outliers_error:.3f}"
        )
        print(f"Branches used: {full_branches}")
        print(f"Saved event IDs: {full_json_path}")

    x_fit = np.linspace(x1, x2, 300)
    y_fit = exp_func(x_fit, *popt)
    y_err = np.array([
        exp_uncertainty(x, popt[0], popt[1], pcov)
        for x in x_fit
    ])
    y_upper = y_fit + y_err
    y_lower = np.maximum(0, y_fit - y_err)
    graph = TGraphErrors(len(x_fit))
    for i, (x, y, err) in enumerate(zip(x_fit, y_fit, y_err)):
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, 0, err)

    graph.SetFillColorAlpha(ROOT.kViolet, 0.5)
    graph.SetLineColor(ROOT.kViolet + 2)
    graph.Draw("3 same")
    fitLine.Draw("same")

    n_experiments = 100000
    integral_estimates = []

    for i in range(n_experiments):
        A_s, B_s = np.random.multivariate_normal(popt, pcov)
        #y_fake = exp_func(x_data, A_s, B_s) + np.random.normal(0, 0.5, size=len(x_data))
        try:
            #popt_fake, _ = curve_fit(exp_func, x_data, y_fake, p0=popt)
            #A_fake, B_fake = popt_fake
            #integral, error = quad(exp_func, cut, 1, args=(A_fake, B_fake))
            integral, error = quad(exp_func, cut, 1, args=(A_s, B_s))
            if integral >= 0:
                integral_estimates.append(integral * scale_factor / bin_width)
            else:
                continue
        except RuntimeError:
            continue

    mean = np.mean(integral_estimates)
    std = np.std(integral_estimates, ddof=1)  # ddof=1 for unbiased estimate
    #upper_limit_95 = np.percentile(integral_estimates, 95)
    #upper_limit_95 = poisson.ppf(0.95, mu=mean)
    n_experiments = 500
    counts = np.random.poisson(mean, size=n_experiments)
    upper_limit_95 = np.percentile(counts, 95)
    print("Background Estimation:")
    print(f"Mean background estimate: {mean:.6f}")
    print(f"Uncertainty (1σ): {std:.6f}")
    print(f"95% CI upper limit: {upper_limit_95:.6f}")

    textstr = '\n'.join((
        f'N = {len(integral_estimates)}',
        f'Mean = {mean:.6f}',
        f'Std Dev = {std:.6f}',
        f'95% CI upper limit = {upper_limit_95:.6f}'
    ))

    plt.figure(figsize=(10, 5))
    plt.hist(integral_estimates, bins=40, alpha=0.7, color='royalblue', edgecolor='black')
    plt.axvline(mean-std, color='orange', linestyle='--', linewidth=1.0)
    plt.axvline(mean, color='red', linestyle='--')
    plt.axvline(mean+std, color='orange', linestyle='--', linewidth=1.0)
    #plt.axvline(upper_limit_95, color='green', linestyle='--', linewidth=1.0)
    plt.xlabel(f"Estimated Background Events ({int(whole_test_factor*100)}% Full Sample)", fontsize = 12.0)
    plt.ylabel("Number of MC Pseudo-Experiments", fontsize = 12.0)
    plt.title(f"Distribution of nEvents_bkg from Monte Carlo Pseudo-Experiments (S{stationNumber})")
    plt.text(
        0.95, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.grid(True)
    plt.savefig(dir_out+graphFilename_expMC, bbox_inches='tight')
    plt.close()



    leg_xMin = 0.7
    leg_xMax = 0.9
    leg_yMin = 0.65
    leg_yMax = 0.9

    leg_hist = TLegend(leg_xMin, leg_yMin, leg_xMax, leg_yMax)
    leg_hist.AddEntry(hist_S, "Signal", "f")
    leg_hist.AddEntry(hist_B, "Background", "f")
    leg_hist.AddEntry(fitLine, "Exponential Fit", "l")
    leg_hist.AddEntry(graph, "Uncertainty", "f")
    leg_hist.Draw()

    if full_data_file:
        canvas.Print(dir_out+graphFilename + "(", "pdf")
    else:
        canvas.Print(dir_out+graphFilename, "pdf")
    canvas.Clear("D")


    x_fit_full = np.linspace(x1, xMax, 500)
    y_fit_full = scale_factor * exp_func(x_fit_full, *popt)
    y_err_full = scale_factor * np.array([
        exp_uncertainty(x, popt[0], popt[1], pcov)
        for x in x_fit_full
    ])
    graph_full = TGraphErrors(len(x_fit_full))

    for i, (x, y, err) in enumerate(
        zip(x_fit_full, y_fit_full, y_err_full)
    ):
        graph_full.SetPoint(i, x, y)
        graph_full.SetPointError(i, 0.0, err)

    graph_full.SetFillColorAlpha(ROOT.kViolet, 0.5)
    graph_full.SetLineColor(ROOT.kViolet + 2)

    if full_data_file:
        canvas.Clear()
        canvas.cd()
        canvas.SetLogy(1)

        hist_B_full.SetTitle(
            f"Full-data background compared with scaled burn-sample fit "
            f"(S{stationNumber})"
        )

        hist_B_full.GetXaxis().SetTitle(f"{method} response")
        hist_B_full.GetYaxis().SetTitle("Events")

        # Leave enough vertical space for the histogram, fit, and band.
        plot_max = max(
            hist_B_full.GetMaximum(),
            fitLine_full.GetMaximum(x1, xMax)
        )
        hist_B_full.SetMaximum(2.0 * plot_max)

        # A positive minimum is required for a logarithmic y-axis.
        hist_B_full.SetMinimum(0.1)

        hist_B_full.Draw("hist")

        # Draw the uncertainty band first, then the central fit on top.
        graph_full.Draw("3 same")
        fitLine_full.Draw("same")

        cutLine_full = TLine(
            cut,
            hist_B_full.GetMinimum(),
            cut,
            hist_B_full.GetMaximum()
        )
        cutLine_full.SetLineStyle(2)
        cutLine_full.SetLineWidth(2)
        cutLine_full.Draw("same")

        leg_full = TLegend(0.565, 0.58, 0.9, 0.9)
        leg_full.AddEntry(
            hist_B_full,
            f"{int(whole_test_factor*100)}% full sample",
            "f"
        )
        leg_full.AddEntry(
            fitLine_full,
            f"Burn-sample fit #times {scale_factor:.0f}",
            "l"
        )
        leg_full.AddEntry(
            graph_full,
            "Scaled burn-fit uncertainty",
            "f"
        )
        leg_full.AddEntry(
            cutLine_full,
            f"BDT cut = {cut:.3f}",
            "l"
        )
        leg_full.Draw()

        # Keep the multipage PDF open; page 4 closes it below.
        canvas.Print(dir_out + graphFilename, "pdf")

        cut_bin_full = hist_B_full.FindFixBin(cut)

        n_observed_full = hist_B_full.Integral(
            cut_bin_full,
            hist_B_full.GetNbinsX()
        )

        n_predicted_full = (
            fitLine_full.Integral(cut, xMax) / bin_width
        )

        n_predicted_full_error = scale_factor * tail_error
        n_predicted_full_fit_start_systematic = (
            scale_factor * fit_start_tail_systematic
        )
        n_predicted_full_total_error = np.hypot(
            n_predicted_full_error,
            n_predicted_full_fit_start_systematic,
        )

        print("\nFull-data validation:")
        print(f"Scale factor: {scale_factor:.0f}")
        print(f"Observed events above cut: {n_observed_full:.0f}")
        print(
            f"Predicted events above cut: "
            f"{n_predicted_full:.3f} +/- "
            f"{n_predicted_full_error:.3f} (fit statistical) +/- "
            f"{n_predicted_full_fit_start_systematic:.3f} "
            f"(fit-start range)"
        )
        print(
            f"Total prediction uncertainty: "
            f"{n_predicted_full_total_error:.3f}"
        )

        # Page 3: subtract the fixed, scaled burn-sample prediction from each
        # selected full-data outlier bin. The Gaussian is fitted to these excess
        # bin contents; the exponential parameters are never refitted.
        canvas.Clear()
        canvas.cd()
        canvas.SetLogy(0)

        outlier_title = (
            f"Background-subtracted right-tail excess: "
            f"{method} > {outlier_min_score:.3f} (S{stationNumber})"
        )
        hist_outliers_observed = TH1F(
            "hist_outliers_observed",
            "Observed events in selected outlier bins",
            nbin,
            xMin,
            xMax,
        )
        for score in full_outlier_scores:
            hist_outliers_observed.Fill(score)

        hist_outliers_full = TH1F(
            "hist_outliers_full",
            outlier_title,
            nbin,
            xMin,
            xMax,
        )
        hist_outliers_full.GetXaxis().SetTitle(f"{method} response")
        hist_outliers_full.GetYaxis().SetTitle(
            "Excess events after fit subtraction"
        )
        hist_outliers_full.GetXaxis().SetRangeUser(
            outlier_min_score, cut
        )
        hist_outliers_full.SetLineColor(ROOT.kRed + 1)
        hist_outliers_full.SetLineWidth(3)
        hist_outliers_full.SetFillColorAlpha(ROOT.kRed - 9, 0.45)

        for bin_index in sorted(full_outlier_bins):
            observed = hist_outliers_observed.GetBinContent(bin_index)
            expected_info = full_expected_by_bin.get(
                bin_index,
                {"yield": 0.0, "uncertainty": 0.0},
            )
            excess = observed - expected_info["yield"]
            excess_error = np.sqrt(
                observed + expected_info["uncertainty"]**2
            )
            hist_outliers_full.SetBinContent(bin_index, excess)
            hist_outliers_full.SetBinError(bin_index, excess_error)

        hist_outliers_full.SetMinimum(0.0)
        maximum_with_error = max(
            (
                hist_outliers_full.GetBinContent(bin_index)
                + hist_outliers_full.GetBinError(bin_index)
                for bin_index in full_outlier_bins
            ),
            default=1.0,
        )
        hist_outliers_full.SetMaximum(
            max(1.0, 1.35 * maximum_with_error)
        )
        hist_outliers_full.Draw("hist")

        gaussian_fit = None
        gaussian_fit_status = None
        gaussian_fit_succeeded = False
        gaussian_fit_covariance = None
        positive_excess_bins = [
            bin_index
            for bin_index in sorted(full_outlier_bins)
            if hist_outliers_full.GetBinContent(bin_index) > 0.0
        ]
        excess_weights = np.array([
            hist_outliers_full.GetBinContent(bin_index)
            for bin_index in positive_excess_bins
        ])
        excess_centers = np.array([
            hist_outliers_full.GetXaxis().GetBinCenter(bin_index)
            for bin_index in positive_excess_bins
        ])

        if len(positive_excess_bins) >= 3 and np.sum(excess_weights) > 0.0:
            weighted_mean = np.average(
                excess_centers, weights=excess_weights
            )
            weighted_variance = np.average(
                (excess_centers - weighted_mean)**2,
                weights=excess_weights,
            )
            weighted_std = np.sqrt(max(weighted_variance, 0.0))
            fit_low = max(
                outlier_min_score,
                hist_outliers_full.GetXaxis().GetBinLowEdge(
                    positive_excess_bins[0]
                ),
            )
            fit_high = min(
                xMax,
                hist_outliers_full.GetXaxis().GetBinUpEdge(
                    positive_excess_bins[-1]
                ),
            )
            gaussian_fit = TF1(
                "gaussian_fit_full_outliers",
                "gaus",
                fit_low,
                fit_high,
            )
            gaussian_fit.SetParameters(
                hist_outliers_full.GetMaximum(),
                weighted_mean,
                max(weighted_std, bin_width),
            )
            gaussian_fit.SetParLimits(1, fit_low, fit_high)
            gaussian_fit.SetParLimits(
                2, 0.1 * bin_width, xMax - outlier_min_score
            )
            fit_result = hist_outliers_full.Fit(
                gaussian_fit, "RQS0"
            )
            gaussian_fit_status = int(fit_result)
            gaussian_fit_succeeded = gaussian_fit_status == 0
            if gaussian_fit_succeeded:
                gaussian_fit_covariance = fit_covariance_matrix(
                    fit_result, gaussian_fit, 3
                )
                gaussian_fit.SetLineColor(ROOT.kBlue)
                gaussian_fit.SetLineWidth(3)
                gaussian_fit.Draw("same")

        outlier_legend = TLegend(0.12, 0.66, 0.49, 0.89)
        outlier_legend.AddEntry(
            hist_outliers_full,
            "Background-subtracted excess",
            "f",
        )
        if gaussian_fit_succeeded:
            outlier_legend.AddEntry(
                gaussian_fit, "Gaussian fit to excess", "l"
            )
        outlier_legend.Draw()

        fit_text = ROOT.TLatex()
        fit_text.SetNDC(True)
        fit_text.SetTextSize(0.027)
        fit_text.DrawLatex(
            0.58,
            0.86,
            f"Selection: {method} > {outlier_min_score:.3f}",
        )
        fit_text.DrawLatex(
            0.58,
            0.81,
            f"Observed in selected bins: {full_observed_outliers}",
        )
        fit_text.DrawLatex(
            0.58,
            0.76,
            f"Exponential estimate: {full_expected_outliers:.2f} #pm "
            f"{full_expected_outliers_error:.2f}",
        )
        fit_text.DrawLatex(
            0.58,
            0.71,
            f"Background-subtracted excess: {full_excess_outliers:.2f} "
            f"#pm {full_excess_outliers_error:.2f}",
        )
        if gaussian_fit_succeeded:
            fit_text.DrawLatex(
                0.58,
                0.66,
                f"Gaussian mean: {gaussian_fit.GetParameter(1):.4f}",
            )
            fit_text.DrawLatex(
                0.58,
                0.61,
                f"Gaussian sigma: {gaussian_fit.GetParameter(2):.4f}",
            )
            fit_text.DrawLatex(
                0.58,
                0.56,
                f"#chi^{{2}} / ndf: {gaussian_fit.GetChisquare():.2f} / "
                f"{gaussian_fit.GetNDF()}",
            )
            fit_text.DrawLatex(
                0.58,
                0.51,
                f"Gaussian-fit p-value: {gaussian_fit.GetProb():.3g}",
            )
        elif gaussian_fit is not None:
            fit_text.DrawLatex(
                0.58,
                0.66,
                f"Gaussian fit failed (ROOT status {gaussian_fit_status})",
            )
        else:
            fit_text.DrawLatex(
                0.58,
                0.66,
                "Gaussian fit unavailable: fewer than 3 excess bins",
            )

        # Keep the multipage PDF open for the combined-model comparison.
        canvas.Print(dir_out + graphFilename, "pdf")

        # Page 4: compare the full-data histogram with the sum of the fixed,
        # scaled burn-sample exponential and the Gaussian excess fitted on
        # page 3. Neither component is refitted here.
        canvas.Clear()
        canvas.cd()
        canvas.SetLogy(1)

        hist_B_full.SetTitle(
            f"Full-data background compared with exponential + Gaussian "
            f"model (S{stationNumber})"
        )
        hist_B_full.GetXaxis().SetTitle(f"{method} response")
        hist_B_full.GetYaxis().SetTitle("Events")
        hist_B_full.SetMinimum(0.1)
        hist_B_full.Draw("hist")

        combined_model_full = None
        combined_uncertainty = graph_full
        combined_line = fitLine_full
        combined_label = "Scaled burn fit (Gaussian unavailable)"
        uncertainty_label = "Scaled burn-fit uncertainty"

        if gaussian_fit_succeeded:
            combined_formula = (
                f"[0]*exp(-[1]*(x-{x1})) + "
                "[2]*exp(-0.5*((x-[3])/[4])^2)"
            )
            combined_model_full = TF1(
                "combined_exponential_gaussian_full",
                combined_formula,
                x1,
                xMax,
            )
            combined_model_full.SetParameter(
                0, fitLine_full.GetParameter(0)
            )
            combined_model_full.SetParameter(
                1, fitLine_full.GetParameter(1)
            )
            combined_model_full.SetParameter(
                2, gaussian_fit.GetParameter(0)
            )
            combined_model_full.SetParameter(
                3, gaussian_fit.GetParameter(1)
            )
            combined_model_full.SetParameter(
                4, gaussian_fit.GetParameter(2)
            )
            combined_model_full.SetLineStyle(1)
            combined_model_full.SetLineWidth(3)
            combined_model_full.SetLineColor(ROOT.kBlue + 1)

            # Shift the page-2 burn-fit uncertainty band by the Gaussian
            # central value. The error bars still represent only the
            # exponential-component uncertainty; the Gaussian is held fixed.
            combined_uncertainty = TGraphErrors(len(x_fit_full))
            for point_index, (x_value, exp_error) in enumerate(
                zip(x_fit_full, y_err_full)
            ):
                combined_y = combined_model_full.Eval(x_value)
                combined_uncertainty.SetPoint(
                    point_index, x_value, combined_y
                )
                combined_uncertainty.SetPointError(
                    point_index, 0.0, exp_error
                )

            combined_uncertainty.SetFillColorAlpha(
                ROOT.kViolet, 0.5
            )
            combined_uncertainty.SetLineColor(ROOT.kViolet + 2)
            combined_line = combined_model_full
            combined_label = "Scaled burn fit + Gaussian excess"
            uncertainty_label = "Exponential-component uncertainty"

            print("\nPage 4 combined model:")
            print(
                "Using fixed scaled-exponential and Gaussian parameters; "
                "no combined refit was performed."
            )

            gaussian_parameters = np.array([
                gaussian_fit.GetParameter(parameter_index)
                for parameter_index in range(3)
            ])
            (
                n_predicted_gaussian_full,
                n_predicted_gaussian_full_error,
            ) = gaussian_interval_prediction(
                cut,
                xMax,
                gaussian_parameters[0],
                gaussian_parameters[1],
                gaussian_parameters[2],
                gaussian_fit_covariance,
                bin_width,
            )
            n_predicted_combined_full = (
                n_predicted_full + n_predicted_gaussian_full
            )
            # The burn-sample exponential fit and the full-data Gaussian fit
            # are treated as independent for this approximate propagation.
            n_predicted_combined_full_statistical_error = np.sqrt(
                n_predicted_full_error**2
                + n_predicted_gaussian_full_error**2
            )
            n_predicted_combined_full_total_error = np.hypot(
                n_predicted_combined_full_statistical_error,
                n_predicted_full_fit_start_systematic,
            )

            print("\nFull-data combined-model estimate:")
            print(f"Observed events above cut: {n_observed_full:.0f}")
            print(
                f"Exponential component above cut: "
                f"{n_predicted_full:.3f} +/- "
                f"{n_predicted_full_error:.3f} (fit statistical) +/- "
                f"{n_predicted_full_fit_start_systematic:.3f} "
                f"(fit-start range)"
            )
            print(
                f"Gaussian component above cut: "
                f"{n_predicted_gaussian_full:.3f} +/- "
                f"{n_predicted_gaussian_full_error:.3f}"
            )
            print(
                f"Combined predicted events above cut: "
                f"{n_predicted_combined_full:.3f} +/- "
                f"{n_predicted_combined_full_statistical_error:.3f} "
                f"(fit statistical) +/- "
                f"{n_predicted_full_fit_start_systematic:.3f} "
                f"(fit-start range)"
            )
            print(
                f"Combined total uncertainty: "
                f"{n_predicted_combined_full_total_error:.3f}"
            )
            print(
                "The Gaussian component is fitted to the full-data excess; "
                "this combined value is a data-calibrated estimate, not an "
                "independent full-data prediction."
            )

        combined_uncertainty.Draw("3 same")
        combined_line.Draw("same")
        cutLine_full.Draw("same")

        leg_combined = TLegend(0.50, 0.55, 0.90, 0.90)
        leg_combined.AddEntry(
            hist_B_full,
            f"{int(whole_test_factor*100)}% full sample",
            "f",
        )
        leg_combined.AddEntry(
            combined_line,
            combined_label,
            "l",
        )
        leg_combined.AddEntry(
            combined_uncertainty,
            uncertainty_label,
            "f",
        )
        leg_combined.AddEntry(
            cutLine_full,
            f"BDT cut = {cut:.3f}",
            "l",
        )
        leg_combined.Draw()

        canvas.Print(dir_out + graphFilename + ")", "pdf")
