import argparse
import numpy as np
import ROOT
from ROOT import TFile, TTree, TCanvas, TPad, TH1F, TGraph, TLine, TLegend
from array import array
import json
from simulation_weighting import get_sim_event_weight

parser = argparse.ArgumentParser(description='test BDT')
parser.add_argument('station', type=str, help="Station number")
parser.add_argument('file_in', type=str, help="Path to the input file")
parser.add_argument('sim_file_in', type=str, help="Path to the input simulation file")
parser.add_argument('dir_trained', type=str, help="Path to the directory of trained BDT weights")
parser.add_argument('dir_out', type=str, help="Output directory")
parser.add_argument('--target_cut', type=float, default=0.45, help="Target BDT cut")
parser.add_argument(
    '--efficiency_half_window',
    type=float,
    default=0.30,
    help=(
        "Half-width of the cut-centered x-axis range on page 5 "
        "(default: 0.30)."
    ),
)
parser.add_argument(
    '--weighting_convention',
    choices=('colleague', 'truth-support', 'truth-extrapolated'),
    default='colleague',
    help=(
        "Weighting definition. 'colleague' reproduces the collaborator "
        "plotting fallback (Y=1, extrapolation included); 'truth-support' "
        "uses per-event inelasticity within digitized support; "
        "'truth-extrapolated' uses per-event inelasticity plus extrapolation."
    ),
)
args = parser.parse_args()

station = args.station
file_in = args.file_in
sim_file_in = args.sim_file_in
dir_trained = args.dir_trained
if not dir_trained.endswith("/"):
    dir_trained += "/"
dir_out = args.dir_out
if not dir_out.endswith("/"):
    dir_out += "/"

station_str = f"s{station}"

# Target signal efficiency
targetCut = args.target_cut
efficiency_half_window = args.efficiency_half_window
weighting_convention = args.weighting_convention

if efficiency_half_window <= 0:
    raise ValueError("--efficiency_half_window must be positive.")

use_unity_inelasticity = weighting_convention == 'colleague'
include_extrapolated_weights = weighting_convention in (
    'colleague',
    'truth-extrapolated',
)
if weighting_convention == 'colleague':
    weighting_curve_label = 'Signal weighted (Y=1 fallback)'
elif weighting_convention == 'truth-support':
    weighting_curve_label = 'Signal weighted (truth Y, support only)'
else:
    weighting_curve_label = 'Signal weighted (truth Y, extrapolated)'

def weight_status_is_selected(status):
    """Return whether a Coleman weight belongs in the selected analysis."""
    return status == "in_support" or (
        include_extrapolated_weights and status == "extrapolated"
    )

# Coleman shower-to-cosmic-ray energy conversion factors.
F_LOW = 0.1
F_CENTRAL = 0.3
F_HIGH = 0.5

# Method
method = "BDTD"

PRIMARY_BDT_VARIABLES = (
    "averageImpulsivity_PA",
    "coherentKurtosis_PA",
    "averageKurtosis_inIce",
    "averageEntropy_inIce",
    "averageImpulsivity_inIce",
    "coherentKurtosis_inIce",
    "coherentEntropy_inIce",
    "coherentImpulsivity_inIce",
)

SECONDARY_BDT_VARIABLES = (
    "reco_max_corr",
    "reco_surf_corr_z",
    "reco_surf_corr_zen",
    "passed_hit_filter",
    "nCoincidentPairs_inIce",
)

ALL_BDT_VARIABLES = PRIMARY_BDT_VARIABLES + SECONDARY_BDT_VARIABLES
DISCRETE_BDT_VARIABLES = {
    "passed_hit_filter",
    "nCoincidentPairs_inIce",
}


def required_variable_leaves(tree):
    """Return the 13 required input-variable leaves or fail clearly."""
    leaves = {}
    missing = []
    for variable_name in ALL_BDT_VARIABLES:
        leaf = tree.GetLeaf(variable_name)
        if leaf:
            leaves[variable_name] = leaf
        else:
            missing.append(variable_name)

    if missing:
        raise RuntimeError(
            f"Tree '{tree.GetName()}' is missing BDT variables: {missing}"
        )
    return leaves


def finite_values(values):
    """Return a one-dimensional array containing finite values only."""
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def variable_axis_range(signal_values, background_values, is_discrete):
    """Choose a shared axis range for signal and background."""
    signal_values = finite_values(signal_values)
    background_values = finite_values(background_values)
    combined = np.concatenate((signal_values, background_values))
    if combined.size == 0:
        return -0.5, 0.5

    value_min = float(np.min(combined))
    value_max = float(np.max(combined))
    if is_discrete:
        return np.floor(value_min) - 0.5, np.ceil(value_max) + 0.5

    if value_max <= value_min:
        padding = max(0.5, 0.05 * abs(value_min))
    else:
        padding = 0.04 * (value_max - value_min)
    return value_min - padding, value_max + padding


def configure_variable_pad(pad):
    """Apply consistent margins and typography space to a subplot pad."""
    pad.SetLeftMargin(0.15)
    pad.SetRightMargin(0.04)
    pad.SetBottomMargin(0.17)
    pad.SetTopMargin(0.11)
    pad.SetTicks(1, 1)


def make_variable_histogram_page(
    canvas_name,
    page_title,
    variable_names,
    signal_values_by_name,
    background_values_by_name,
):
    """Create normalized 1-D overlays with background drawn last."""
    columns = 4 if len(variable_names) == 8 else 3
    canvas = TCanvas(canvas_name, page_title, 10, 10, 1800, 900)
    canvas.Divide(columns, 2, 0.002, 0.002)
    objects = []

    for panel_index, variable_name in enumerate(variable_names, start=1):
        pad = canvas.cd(panel_index)
        configure_variable_pad(pad)
        pad.SetLogy(False)

        is_discrete = variable_name in DISCRETE_BDT_VARIABLES
        x_low, x_high = variable_axis_range(
            signal_values_by_name[variable_name],
            background_values_by_name[variable_name],
            is_discrete,
        )
        if is_discrete:
            n_bins = max(1, min(60, int(round(x_high - x_low))))
        else:
            n_bins = 50

        signal_hist = TH1F(
            f"{canvas_name}_signal_{panel_index}",
            f"{variable_name};{variable_name};Normalized entries",
            n_bins,
            x_low,
            x_high,
        )
        background_hist = TH1F(
            f"{canvas_name}_background_{panel_index}",
            f"{variable_name};{variable_name};Normalized entries",
            n_bins,
            x_low,
            x_high,
        )
        signal_hist.SetDirectory(0)
        background_hist.SetDirectory(0)

        for value in finite_values(signal_values_by_name[variable_name]):
            signal_hist.Fill(value)
        for value in finite_values(background_values_by_name[variable_name]):
            background_hist.Fill(value)

        if signal_hist.Integral() > 0:
            signal_hist.Scale(1.0 / signal_hist.Integral())
        if background_hist.Integral() > 0:
            background_hist.Scale(1.0 / background_hist.Integral())

        signal_hist.SetLineColor(ROOT.kAzure + 2)
        signal_hist.SetLineWidth(2)
        signal_hist.SetFillColorAlpha(ROOT.kAzure - 7, 0.30)
        background_hist.SetLineColor(ROOT.kRed + 1)
        background_hist.SetLineWidth(3)
        background_hist.SetFillColorAlpha(ROOT.kRed - 9, 0.18)

        maximum = max(signal_hist.GetMaximum(), background_hist.GetMaximum())
        signal_hist.SetMaximum(max(1.0e-6, 1.25 * maximum))
        signal_hist.SetMinimum(0.0)
        signal_hist.GetXaxis().SetTitleSize(0.045)
        signal_hist.GetYaxis().SetTitleSize(0.045)
        signal_hist.GetXaxis().SetLabelSize(0.036)
        signal_hist.GetYaxis().SetLabelSize(0.036)
        signal_hist.GetYaxis().SetTitleOffset(1.45)

        # Draw signal first and background last so background remains visible.
        signal_hist.Draw("hist")
        background_hist.Draw("hist same")

        objects.extend((signal_hist, background_hist))
        if panel_index == 1:
            legend = TLegend(0.54, 0.72, 0.95, 0.91)
            legend.SetTextSize(0.036)
            legend.AddEntry(signal_hist, "Signal", "f")
            legend.AddEntry(background_hist, "Background", "f")
            legend.Draw()
            objects.append(legend)

    canvas.Modified()
    canvas.Update()
    return canvas, objects


def make_variable_scatter_page(
    canvas_name,
    page_title,
    variable_names,
    signal_scores,
    background_scores,
    signal_values_by_name,
    background_values_by_name,
    score_minimum,
    score_maximum,
    selected_cut,
):
    """Create BDT-score scatter panels with background drawn last."""
    columns = 4 if len(variable_names) == 8 else 3
    canvas = TCanvas(canvas_name, page_title, 10, 10, 1800, 900)
    canvas.Divide(columns, 2, 0.002, 0.002)
    objects = []
    signal_scores = np.asarray(signal_scores, dtype=float)
    background_scores = np.asarray(background_scores, dtype=float)

    for panel_index, variable_name in enumerate(variable_names, start=1):
        pad = canvas.cd(panel_index)
        configure_variable_pad(pad)
        pad.SetLogy(False)

        signal_values = np.asarray(
            signal_values_by_name[variable_name], dtype=float
        )
        background_values = np.asarray(
            background_values_by_name[variable_name], dtype=float
        )
        signal_mask = np.isfinite(signal_scores) & np.isfinite(signal_values)
        background_mask = (
            np.isfinite(background_scores) & np.isfinite(background_values)
        )
        x_signal = signal_scores[signal_mask]
        y_signal = signal_values[signal_mask]
        x_background = background_scores[background_mask]
        y_background = background_values[background_mask]

        if x_signal.size == 0 or x_background.size == 0:
            raise RuntimeError(
                f"No finite signal/background points for {variable_name}"
            )

        is_discrete = variable_name in DISCRETE_BDT_VARIABLES
        y_low, y_high = variable_axis_range(
            y_signal,
            y_background,
            is_discrete,
        )

        signal_graph = TGraph(len(x_signal))
        background_graph = TGraph(len(x_background))
        for point_index, (score, value) in enumerate(
            zip(x_signal, y_signal)
        ):
            signal_graph.SetPoint(point_index, score, value)
        for point_index, (score, value) in enumerate(
            zip(x_background, y_background)
        ):
            background_graph.SetPoint(point_index, score, value)

        signal_graph.SetTitle(
            f"{variable_name} vs {method};{method} response;{variable_name}"
        )
        signal_graph.SetMarkerStyle(20)
        signal_graph.SetMarkerSize(0.85)
        signal_graph.SetMarkerColorAlpha(ROOT.kAzure + 2, 0.28)
        background_graph.SetMarkerStyle(20)
        background_graph.SetMarkerSize(1.05)
        background_graph.SetMarkerColorAlpha(ROOT.kRed + 1, 0.62)
        signal_graph.SetMinimum(y_low)
        signal_graph.SetMaximum(y_high)

        signal_graph.Draw("AP")
        signal_graph.GetXaxis().SetLimits(score_minimum, score_maximum)
        signal_graph.GetXaxis().SetTitleSize(0.045)
        signal_graph.GetYaxis().SetTitleSize(0.045)
        signal_graph.GetXaxis().SetLabelSize(0.036)
        signal_graph.GetYaxis().SetLabelSize(0.036)
        signal_graph.GetYaxis().SetTitleOffset(1.45)

        # Draw background after signal, then place the cut line above both.
        background_graph.Draw("P same")
        cut_line = TLine(selected_cut, y_low, selected_cut, y_high)
        cut_line.SetLineColor(ROOT.kBlack)
        cut_line.SetLineStyle(2)
        cut_line.SetLineWidth(2)
        cut_line.Draw("same")

        objects.extend((signal_graph, background_graph, cut_line))
        if panel_index == 1:
            legend = TLegend(0.50, 0.68, 0.95, 0.91)
            legend.SetTextSize(0.033)
            legend.AddEntry(signal_graph, "Signal", "p")
            legend.AddEntry(background_graph, "Background", "p")
            legend.AddEntry(
                cut_line, f"BDT cut = {selected_cut:.3f}", "l"
            )
            legend.Draw()
            objects.append(legend)

    canvas.Modified()
    canvas.Update()
    return canvas, objects

jsonFileName = "falsePositiveEvents_vars_" + station_str + f"_{method}.json"
targetFileName = "testTree_vars_" + station_str + f"_{method}.root"
graphFileName = "testedResults_vars_" + station_str + f"_{method}.pdf"
variableGraphFileName = (
    "variableDistributions_vars_" + station_str + f"_{method}.pdf"
)

TMVA = ROOT.TMVA

TMVA.Tools.Instance()
#TMVA.PyMethodBase.PyInitialize()

print("==> Start BDT testing")


station_number_float = array("f", [0.])
run_number_float = array("f", [0.])
event_number_float = array("f", [0.])

interaction_type_float = array("f", [0.])

trigger_time_float = array("f", [0.])

true_source_theta_float = array("f", [0.])
true_source_phi_float = array("f", [0.])

passed_hit_filter_float = array("f", [0.])
nCoincidentPairs_PA_float = array("f", [0.])
nHighHits_PA_float = array("f", [0.])
nCoincidentPairs_inIce_float = array("f", [0.])
nHighHits_inIce_float = array("f", [0.])




station_number = array("i", [0])
run_number = array("i", [0])
event_number = array("i", [0])

sim_energy = array("f", [0.])
shower_energy = array("f", [0.])
inelasticity = array("f", [0.])
interaction_type = array("i", [0])

trigger_time = array("d", [0.])

true_radius = array("f", [0.])
true_theta = array("f", [0.])
true_phi = array("f", [0.])
true_source_theta = array("i", [0])
true_source_phi = array("i", [0])

reco_max_corr = array("f", [np.nan])
reco_surf_corr_z = array("f", [np.nan])
reco_surf_corr_zen = array("f", [np.nan])
reco_rho = array("f", [np.nan])
reco_phi = array("f", [np.nan])
reco_z = array("f", [np.nan])

passed_hit_filter = array("i", [0])
nCoincidentPairs_PA = array("i", [0])
nHighHits_PA = array("i", [0])
nCoincidentPairs_inIce = array("i", [0])
nHighHits_inIce = array("i", [0])

averageSNR_PA = array("f", [0.])
averageKurtosis_PA = array("f", [0.])
averageEntropy_PA = array("f", [0.])
averageImpulsivity_PA = array("f", [0.])
coherentSNR_PA = array("f", [0.])
coherentKurtosis_PA = array("f", [0.])
coherentEntropy_PA = array("f", [0.])
coherentImpulsivity_PA = array("f", [0.])

averageSNR_inIce = array("f", [0.])
averageKurtosis_inIce = array("f", [0.])
averageEntropy_inIce = array("f", [0.])
averageImpulsivity_inIce = array("f", [0.])
coherentSNR_inIce = array("f", [0.])
coherentKurtosis_inIce = array("f", [0.])
coherentEntropy_inIce = array("f", [0.])
coherentImpulsivity_inIce = array("f", [0.])

weight = array("f", [0.])


reader = TMVA.Reader( "!Color:!Silent" )
reader.AddVariable( "passed_hit_filter", passed_hit_filter_float )
#reader.AddVariable( "nCoincidentPairs_PA", nCoincidentPairs_PA_float )
#reader.AddVariable( "nHighHits_PA", nHighHits_PA_float )
reader.AddVariable( "nCoincidentPairs_inIce", nCoincidentPairs_inIce_float )
#reader.AddVariable( "nHighHits_inIce", nHighHits_inIce_float )

reader.AddVariable( "reco_max_corr", reco_max_corr )
reader.AddVariable( "reco_surf_corr_z", reco_surf_corr_z )
reader.AddVariable( "reco_surf_corr_zen", reco_surf_corr_zen )

#reader.AddVariable( "averageSNR_PA", averageSNR_PA )
#reader.AddVariable( "averageKurtosis_PA", averageKurtosis_PA )
#reader.AddVariable( "averageEntropy_PA", averageEntropy_PA )
reader.AddVariable( "averageImpulsivity_PA", averageImpulsivity_PA )
#reader.AddVariable( "coherentSNR_PA", coherentSNR_PA )
reader.AddVariable( "coherentKurtosis_PA", coherentKurtosis_PA )
#reader.AddVariable( "coherentEntropy_PA", coherentEntropy_PA )
#reader.AddVariable( "coherentImpulsivity_PA", coherentImpulsivity_PA )

#reader.AddVariable( "averageSNR_inIce", averageSNR_inIce )
reader.AddVariable( "averageKurtosis_inIce", averageKurtosis_inIce )
reader.AddVariable( "averageEntropy_inIce", averageEntropy_inIce )
reader.AddVariable( "averageImpulsivity_inIce", averageImpulsivity_inIce )
#reader.AddVariable( "coherentSNR_inIce", coherentSNR_inIce )
reader.AddVariable( "coherentKurtosis_inIce", coherentKurtosis_inIce )
reader.AddVariable( "coherentEntropy_inIce", coherentEntropy_inIce )
reader.AddVariable( "coherentImpulsivity_inIce", coherentImpulsivity_inIce )

reader.AddSpectator( "station_number", station_number_float )
reader.AddSpectator( "run_number", run_number_float )
reader.AddSpectator( "event_number", event_number_float )

reader.AddSpectator( "sim_energy", sim_energy )
reader.AddSpectator( "shower_energy", shower_energy )
reader.AddSpectator( "inelasticity", inelasticity )
reader.AddSpectator( "interaction_type", interaction_type_float )

reader.AddSpectator( "trigger_time", trigger_time_float )

reader.AddSpectator( "true_radius", true_radius )
reader.AddSpectator( "true_theta", true_theta )
reader.AddSpectator( "true_phi", true_phi )
reader.AddSpectator( "true_source_theta", true_source_theta_float )
reader.AddSpectator( "true_source_phi", true_source_phi_float )

reader.AddSpectator( "reco_rho", reco_rho )
reader.AddSpectator( "reco_phi", reco_phi )
reader.AddSpectator( "reco_z", reco_z )

prefix = "TMVA_Classification"
methodName = f"{method} method"
weightfile = dir_trained + prefix + "_" + method + ".weights.xml"
reader.BookMVA( methodName, weightfile )

input_sig = TFile.Open(sim_file_in)
tree_S = input_sig.Get(f"vars_sig")

tree_S.SetBranchAddress( "station_number", station_number )
tree_S.SetBranchAddress( "run_number", run_number )
tree_S.SetBranchAddress( "event_number", event_number )

tree_S.SetBranchAddress( "sim_energy", sim_energy )
tree_S.SetBranchAddress( "shower_energy", shower_energy )
tree_S.SetBranchAddress( "inelasticity", inelasticity )
tree_S.SetBranchAddress( "interaction_type", interaction_type )

tree_S.SetBranchAddress( "trigger_time", trigger_time )

tree_S.SetBranchAddress( "true_radius", true_radius )
tree_S.SetBranchAddress( "true_theta", true_theta )
tree_S.SetBranchAddress( "true_phi", true_phi )
tree_S.SetBranchAddress( "true_source_theta", true_source_theta )
tree_S.SetBranchAddress( "true_source_phi", true_source_phi )

tree_S.SetBranchAddress( "reco_max_corr", reco_max_corr )
tree_S.SetBranchAddress( "reco_surf_corr_z", reco_surf_corr_z )
tree_S.SetBranchAddress( "reco_surf_corr_zen", reco_surf_corr_zen )
tree_S.SetBranchAddress( "reco_rho", reco_rho )
tree_S.SetBranchAddress( "reco_phi", reco_phi )
tree_S.SetBranchAddress( "reco_z", reco_z )

tree_S.SetBranchAddress( "passed_hit_filter", passed_hit_filter )
#tree_S.SetBranchAddress( "nCoincidentPairs_PA", nCoincidentPairs_PA )
#tree_S.SetBranchAddress( "nHighHits_PA", nHighHits_PA )
tree_S.SetBranchAddress( "nCoincidentPairs_inIce", nCoincidentPairs_inIce )
#tree_S.SetBranchAddress( "nHighHits_inIce", nHighHits_inIce )

#tree_S.SetBranchAddress( "averageSNR_PA", averageSNR_PA )
#tree_S.SetBranchAddress( "averageKurtosis_PA", averageKurtosis_PA )
#tree_S.SetBranchAddress( "averageEntropy_PA", averageEntropy_PA )
tree_S.SetBranchAddress( "averageImpulsivity_PA", averageImpulsivity_PA )
#tree_S.SetBranchAddress( "coherentSNR_PA", coherentSNR_PA )
tree_S.SetBranchAddress( "coherentKurtosis_PA", coherentKurtosis_PA )
#tree_S.SetBranchAddress( "coherentEntropy_PA", coherentEntropy_PA )
#tree_S.SetBranchAddress( "coherentImpulsivity_PA", coherentImpulsivity_PA )

#tree_S.SetBranchAddress( "averageSNR_inIce", averageSNR_inIce )
tree_S.SetBranchAddress( "averageKurtosis_inIce", averageKurtosis_inIce )
tree_S.SetBranchAddress( "averageEntropy_inIce", averageEntropy_inIce )
tree_S.SetBranchAddress( "averageImpulsivity_inIce", averageImpulsivity_inIce )
#tree_S.SetBranchAddress( "coherentSNR_inIce", coherentSNR_inIce )
tree_S.SetBranchAddress( "coherentKurtosis_inIce", coherentKurtosis_inIce )
tree_S.SetBranchAddress( "coherentEntropy_inIce", coherentEntropy_inIce )
tree_S.SetBranchAddress( "coherentImpulsivity_inIce", coherentImpulsivity_inIce )

nEvents_S = tree_S.GetEntries()


input_bkg = TFile.Open(file_in)
tree_B = input_bkg.Get(f"vars_bkg")

tree_B.SetBranchAddress( "station_number", station_number )
tree_B.SetBranchAddress( "run_number", run_number )
tree_B.SetBranchAddress( "event_number", event_number )

tree_B.SetBranchAddress( "sim_energy", sim_energy )
tree_B.SetBranchAddress( "shower_energy", shower_energy )
tree_B.SetBranchAddress( "inelasticity", inelasticity )
tree_B.SetBranchAddress( "interaction_type", interaction_type )

tree_B.SetBranchAddress( "trigger_time", trigger_time )

tree_B.SetBranchAddress( "true_radius", true_radius )
tree_B.SetBranchAddress( "true_theta", true_theta )
tree_B.SetBranchAddress( "true_phi", true_phi )
tree_B.SetBranchAddress( "true_source_theta", true_source_theta )
tree_B.SetBranchAddress( "true_source_phi", true_source_phi )

tree_B.SetBranchAddress( "reco_max_corr", reco_max_corr )
tree_B.SetBranchAddress( "reco_surf_corr_z", reco_surf_corr_z )
tree_B.SetBranchAddress( "reco_surf_corr_zen", reco_surf_corr_zen )
tree_B.SetBranchAddress( "reco_rho", reco_rho )
tree_B.SetBranchAddress( "reco_phi", reco_phi )
tree_B.SetBranchAddress( "reco_z", reco_z )

tree_B.SetBranchAddress( "passed_hit_filter", passed_hit_filter )
#tree_B.SetBranchAddress( "nCoincidentPairs_PA", nCoincidentPairs_PA )
#tree_B.SetBranchAddress( "nHighHits_PA", nHighHits_PA )
tree_B.SetBranchAddress( "nCoincidentPairs_inIce", nCoincidentPairs_inIce )
#tree_B.SetBranchAddress( "nHighHits_inIce", nHighHits_inIce )

#tree_B.SetBranchAddress( "averageSNR_PA", averageSNR_PA )
#tree_B.SetBranchAddress( "averageKurtosis_PA", averageKurtosis_PA )
#tree_B.SetBranchAddress( "averageEntropy_PA", averageEntropy_PA )
tree_B.SetBranchAddress( "averageImpulsivity_PA", averageImpulsivity_PA )
#tree_B.SetBranchAddress( "coherentSNR_PA", coherentSNR_PA )
tree_B.SetBranchAddress( "coherentKurtosis_PA", coherentKurtosis_PA )
#tree_B.SetBranchAddress( "coherentEntropy_PA", coherentEntropy_PA )
#tree_B.SetBranchAddress( "coherentImpulsivity_PA", coherentImpulsivity_PA )

#tree_B.SetBranchAddress( "averageSNR_inIce", averageSNR_inIce )
tree_B.SetBranchAddress( "averageKurtosis_inIce", averageKurtosis_inIce )
tree_B.SetBranchAddress( "averageEntropy_inIce", averageEntropy_inIce )
tree_B.SetBranchAddress( "averageImpulsivity_inIce", averageImpulsivity_inIce )
#tree_B.SetBranchAddress( "coherentSNR_inIce", coherentSNR_inIce )
tree_B.SetBranchAddress( "coherentKurtosis_inIce", coherentKurtosis_inIce )
tree_B.SetBranchAddress( "coherentEntropy_inIce", coherentEntropy_inIce )
tree_B.SetBranchAddress( "coherentImpulsivity_inIce", coherentImpulsivity_inIce )

nEvents_B = tree_B.GetEntries()

signal_variable_leaves = required_variable_leaves(tree_S)
background_variable_leaves = required_variable_leaves(tree_B)
signal_variable_values = {
    variable_name: [] for variable_name in ALL_BDT_VARIABLES
}
background_variable_values = {
    variable_name: [] for variable_name in ALL_BDT_VARIABLES
}
background_scores_all = []

output = TFile( dir_out+targetFileName, "RECREATE" )
output.cd()

nbin = 100
if method == "BDTD":
    xMin = -0.45
    xMax = 1.0
else:
    xMin = -0.1
    xMax = 1.1

histTitle = f"TMVA response for classifier: {method} (S{station})"
hist_S = TH1F("hist_S", "", nbin, xMin, xMax)
hist_S.GetXaxis().SetTitle(f"{method} response")
hist_S.GetYaxis().SetTitle("Events")
hist_S.SetLineColorAlpha(ROOT.kAzure+2, 0.5)
hist_S.SetLineWidth(3)
hist_S.SetFillColorAlpha(ROOT.kAzure-7, 0.2)

hist_S_weighted_f01 = TH1F(
    "hist_S_weighted_f01",
    "Coleman-weighted signal TMVA response (f=0.1)",
    nbin,
    xMin,
    xMax,
)
hist_S_weighted_f03 = TH1F(
    "hist_S_weighted_f03",
    "Coleman-weighted signal TMVA response (f=0.3)",
    nbin,
    xMin,
    xMax,
)
hist_S_weighted_f05 = TH1F(
    "hist_S_weighted_f05",
    "Coleman-weighted signal TMVA response (f=0.5)",
    nbin,
    xMin,
    xMax,
)
for weighted_hist in (
    hist_S_weighted_f01,
    hist_S_weighted_f03,
    hist_S_weighted_f05,
):
    weighted_hist.Sumw2()
hist_S_weighted_f03.SetLineColor(ROOT.kBlue+2)
hist_S_weighted_f03.SetLineWidth(3)

hist_S_true_y_f01 = TH1F(
    "hist_S_true_y_f01",
    "True-Y Coleman-weighted signal TMVA response (f=0.1)",
    nbin,
    xMin,
    xMax,
)
hist_S_true_y_f03 = TH1F(
    "hist_S_true_y_f03",
    "True-Y Coleman-weighted signal TMVA response (f=0.3)",
    nbin,
    xMin,
    xMax,
)
hist_S_true_y_f05 = TH1F(
    "hist_S_true_y_f05",
    "True-Y Coleman-weighted signal TMVA response (f=0.5)",
    nbin,
    xMin,
    xMax,
)
for true_y_hist in (
    hist_S_true_y_f01,
    hist_S_true_y_f03,
    hist_S_true_y_f05,
):
    true_y_hist.Sumw2()
hist_S_true_y_f03.SetLineColor(ROOT.kGreen+2)
hist_S_true_y_f03.SetLineWidth(3)

hist_B = TH1F("hist_B", histTitle, nbin, xMin, xMax)
hist_B.GetXaxis().SetTitle(f"{method} response")
hist_B.GetYaxis().SetTitle("Events")
hist_B.SetLineColor(ROOT.kRed+1)
hist_B.SetLineWidth(3)
hist_B.SetFillColor(ROOT.kRed+1)
hist_B.SetFillStyle(3354)

EvaluateMVA = array("f", [0.])

testTree_S = TTree("TestTree_S", "TestTree_S")
testTree_S.SetDirectory(output)

testTree_S.Branch( "weight", weight, "weight/F" )

testTree_S.Branch( method, EvaluateMVA, method+"/F" )

testTree_S.Branch( "station_number", station_number, "station_number/I" )
testTree_S.Branch( "run_number", run_number, "run_number/I" )
testTree_S.Branch( "event_number", event_number, "event_number/I" )

testTree_S.Branch( "sim_energy", sim_energy, "sim_energy/F" )
testTree_S.Branch( "shower_energy", shower_energy, "shower_energy/F" )
testTree_S.Branch( "inelasticity", inelasticity, "inelasticity/F" )
testTree_S.Branch( "interaction_type", interaction_type, "interaction_type/I" )

testTree_S.Branch( "trigger_time", trigger_time, "trigger_time/D" )

testTree_S.Branch( "true_radius", true_radius, "true_radius/F" )
testTree_S.Branch( "true_theta", true_theta, "true_theta/F" )
testTree_S.Branch( "true_phi", true_phi, "true_phi/F" )
testTree_S.Branch( "true_source_theta", true_source_theta, "true_source_theta/I" )
testTree_S.Branch( "true_source_phi", true_source_phi, "true_source_phi/I" )

testTree_S.Branch( "reco_rho", reco_rho, "reco_rho/F" )
testTree_S.Branch( "reco_phi", reco_phi, "reco_phi/F" )
testTree_S.Branch( "reco_z", reco_z, "reco_z/F" )

testTree_B = TTree("TestTree_B", "TestTree_B")
testTree_B.SetDirectory(output)

testTree_B.Branch( method, EvaluateMVA, method+"/F" )

testTree_B.Branch( "station_number", station_number, "station_number/I" )
testTree_B.Branch( "run_number", run_number, "run_number/I" )
testTree_B.Branch( "event_number", event_number, "event_number/I" )

testTree_B.Branch( "sim_energy", sim_energy, "sim_energy/F" )
testTree_B.Branch( "shower_energy", shower_energy, "shower_energy/F" )
testTree_B.Branch( "inelasticity", inelasticity, "inelasticity/F" )
testTree_B.Branch( "interaction_type", interaction_type, "interaction_type/I" )

testTree_B.Branch( "trigger_time", trigger_time, "trigger_time/D" )

testTree_B.Branch( "true_radius", true_radius, "true_radius/F" )
testTree_B.Branch( "true_theta", true_theta, "true_theta/F" )
testTree_B.Branch( "true_phi", true_phi, "true_phi/F" )
testTree_B.Branch( "true_source_theta", true_source_theta, "true_source_theta/I" )
testTree_B.Branch( "true_source_phi", true_source_phi, "true_source_phi/I" )

testTree_B.Branch( "reco_rho", reco_rho, "reco_rho/F" )
testTree_B.Branch( "reco_phi", reco_phi, "reco_phi/F" )
testTree_B.Branch( "reco_z", reco_z, "reco_z/F" )


print(f"--- TMVA Classification App    : Using input sim file: {input_sig.GetName()}")

signal_scores_all = []
signal_scores_weighted = []
signal_weights_f01 = []
signal_weights_f03 = []
signal_weights_f05 = []
signal_support_statuses_f01 = []
signal_support_statuses_f03 = []
signal_support_statuses_f05 = []
signal_log10_deposited_energy = []
signal_scores_true_y = []
signal_true_y_weights_f01 = []
signal_true_y_weights_f03 = []
signal_true_y_weights_f05 = []
signal_true_y_statuses_f01 = []
signal_true_y_statuses_f03 = []
signal_true_y_statuses_f05 = []
n_invalid_weights = 0
n_out_of_range = 0

for i_event in range(nEvents_S):
    tree_S.GetEntry(i_event)

    passed_hit_filter_float[0] = passed_hit_filter[0]
    nCoincidentPairs_PA_float[0] = nCoincidentPairs_PA[0]
    nHighHits_PA_float[0] = nHighHits_PA[0]
    nCoincidentPairs_inIce_float[0] = nCoincidentPairs_inIce[0]
    nHighHits_inIce_float[0] = nHighHits_inIce[0]

    station_number_float[0] = station_number[0]
    run_number_float[0] = run_number[0]
    event_number_float[0] = event_number[0]

    interaction_type_float[0] = interaction_type[0]

    trigger_time_float[0] = trigger_time[0]

    true_source_theta_float[0] = true_source_theta[0]
    true_source_phi_float[0] = true_source_phi[0]

    primary_energy_eV = float(sim_energy[0])
    inelasticity_value = float(inelasticity[0])
    shower_energy_eV = float(shower_energy[0])
    cos_true_theta = np.cos(np.deg2rad(float(true_source_theta[0])))

    # The collaborator plotting script substitutes Y=1 when inelasticity is
    # unavailable. In colleague-compatible mode, reproduce that convention
    # deliberately. Passing no shower energy makes the weighting helper use
    # E_shower = Y * E_primary, so its consistency check remains meaningful.
    if use_unity_inelasticity:
        weighting_inelasticity = 1.0
        weighting_shower_energy_eV = None
    else:
        weighting_inelasticity = inelasticity_value
        weighting_shower_energy_eV = shower_energy_eV

    mva_score = reader.EvaluateMVA(methodName)
    EvaluateMVA[0] = mva_score

    signal_scores_all.append(mva_score)
    for variable_name in ALL_BDT_VARIABLES:
        signal_variable_values[variable_name].append(
            float(signal_variable_leaves[variable_name].GetValue())
        )

    weight[0] = np.nan

    try:
        weighting_results = {}
        for f_label, f_factor in (
            ("f01", F_LOW),
            ("f03", F_CENTRAL),
            ("f05", F_HIGH),
        ):
            event_weight, support_status, lgE_dep = get_sim_event_weight(
                primary_energy_eV=primary_energy_eV,
                inelasticity=weighting_inelasticity,
                shower_energy_eV=weighting_shower_energy_eV,
                cos_theta=cos_true_theta,
                f_factor=f_factor,
            )
            if not np.isfinite(event_weight) or event_weight < 0:
                raise ValueError(
                    f"Invalid event weight for f={f_factor}: {event_weight}"
                )
            weighting_results[f_label] = (
                float(event_weight),
                str(support_status),
                float(lgE_dep),
            )

        true_y_weighting_results = {}
        for f_label, f_factor in (
            ("f01", F_LOW),
            ("f03", F_CENTRAL),
            ("f05", F_HIGH),
        ):
            true_y_weight, true_y_status, true_y_lgE_dep = (
                get_sim_event_weight(
                    primary_energy_eV=primary_energy_eV,
                    inelasticity=inelasticity_value,
                    shower_energy_eV=shower_energy_eV,
                    cos_theta=cos_true_theta,
                    f_factor=f_factor,
                )
            )
            if not np.isfinite(true_y_weight) or true_y_weight < 0:
                raise ValueError(
                    "Invalid true-Y event weight for "
                    f"f={f_factor}: {true_y_weight}"
                )
            true_y_weighting_results[f_label] = (
                float(true_y_weight),
                str(true_y_status),
                float(true_y_lgE_dep),
            )

        weight_f01, support_f01, _ = weighting_results["f01"]
        weight_f03, support_f03, lgE_dep_f03 = weighting_results["f03"]
        weight_f05, support_f05, _ = weighting_results["f05"]
        true_y_weight_f01, true_y_status_f01, _ = (
            true_y_weighting_results["f01"]
        )
        true_y_weight_f03, true_y_status_f03, _ = (
            true_y_weighting_results["f03"]
        )
        true_y_weight_f05, true_y_status_f05, _ = (
            true_y_weighting_results["f05"]
        )

        weight[0] = weight_f03

        signal_scores_weighted.append(mva_score)
        signal_weights_f01.append(weight_f01)
        signal_weights_f03.append(weight_f03)
        signal_weights_f05.append(weight_f05)
        signal_support_statuses_f01.append(support_f01)
        signal_support_statuses_f03.append(support_f03)
        signal_support_statuses_f05.append(support_f05)
        signal_log10_deposited_energy.append(lgE_dep_f03)
        signal_scores_true_y.append(mva_score)
        signal_true_y_weights_f01.append(true_y_weight_f01)
        signal_true_y_weights_f03.append(true_y_weight_f03)
        signal_true_y_weights_f05.append(true_y_weight_f05)
        signal_true_y_statuses_f01.append(true_y_status_f01)
        signal_true_y_statuses_f03.append(true_y_status_f03)
        signal_true_y_statuses_f05.append(true_y_status_f05)

        if support_f03 == "out_of_range":
            n_out_of_range += 1

        if weight_status_is_selected(support_f01):
            hist_S_weighted_f01.Fill(mva_score, weight_f01)
        if weight_status_is_selected(support_f03):
            hist_S_weighted_f03.Fill(mva_score, weight_f03)
        if weight_status_is_selected(support_f05):
            hist_S_weighted_f05.Fill(mva_score, weight_f05)

        if true_y_status_f01 == "in_support":
            hist_S_true_y_f01.Fill(mva_score, true_y_weight_f01)
        if true_y_status_f03 == "in_support":
            hist_S_true_y_f03.Fill(mva_score, true_y_weight_f03)
        if true_y_status_f05 == "in_support":
            hist_S_true_y_f05.Fill(mva_score, true_y_weight_f05)

    except ValueError as error:
        n_invalid_weights += 1
        print(
            f"Signal event {i_event} has no valid weight: {error}"
        )

    # Preserve the original, unweighted TMVA response and retain every event
    # in TestTree_S regardless of whether its simulation weight is valid.
    hist_S.Fill(mva_score)
    testTree_S.Fill()
print(f"--- SIGNAL: {testTree_S.GetEntries()} events")
print("--- End of event loop (SIGNAL)")

signal_scores_all = np.asarray(
    signal_scores_all,
    dtype=float,
)
signal_scores_weighted = np.asarray(
    signal_scores_weighted,
    dtype=float,
)
signal_weights_f01 = np.asarray(signal_weights_f01, dtype=float)
signal_weights_f03 = np.asarray(signal_weights_f03, dtype=float)
signal_weights_f05 = np.asarray(signal_weights_f05, dtype=float)
signal_support_statuses_f01 = np.asarray(
    signal_support_statuses_f01, dtype=str
)
signal_support_statuses_f03 = np.asarray(
    signal_support_statuses_f03, dtype=str
)
signal_support_statuses_f05 = np.asarray(
    signal_support_statuses_f05, dtype=str
)
signal_log10_deposited_energy = np.asarray(
    signal_log10_deposited_energy,
    dtype=float,
)
signal_scores_true_y = np.asarray(signal_scores_true_y, dtype=float)
signal_true_y_weights_f01 = np.asarray(
    signal_true_y_weights_f01, dtype=float
)
signal_true_y_weights_f03 = np.asarray(
    signal_true_y_weights_f03, dtype=float
)
signal_true_y_weights_f05 = np.asarray(
    signal_true_y_weights_f05, dtype=float
)
signal_true_y_statuses_f01 = np.asarray(
    signal_true_y_statuses_f01, dtype=str
)
signal_true_y_statuses_f03 = np.asarray(
    signal_true_y_statuses_f03, dtype=str
)
signal_true_y_statuses_f05 = np.asarray(
    signal_true_y_statuses_f05, dtype=str
)

in_support_f01 = signal_support_statuses_f01 == "in_support"
in_support_f03 = signal_support_statuses_f03 == "in_support"
in_support_f05 = signal_support_statuses_f05 == "in_support"

if include_extrapolated_weights:
    analysis_mask_f01 = np.isin(
        signal_support_statuses_f01, ("in_support", "extrapolated")
    )
    analysis_mask_f03 = np.isin(
        signal_support_statuses_f03, ("in_support", "extrapolated")
    )
    analysis_mask_f05 = np.isin(
        signal_support_statuses_f05, ("in_support", "extrapolated")
    )
else:
    analysis_mask_f01 = in_support_f01
    analysis_mask_f03 = in_support_f03
    analysis_mask_f05 = in_support_f05

total_weight_f01 = signal_weights_f01[analysis_mask_f01].sum()
total_weight_f03 = signal_weights_f03[analysis_mask_f03].sum()
total_weight_f05 = signal_weights_f05[analysis_mask_f05].sum()

true_y_analysis_mask_f01 = signal_true_y_statuses_f01 == "in_support"
true_y_analysis_mask_f03 = signal_true_y_statuses_f03 == "in_support"
true_y_analysis_mask_f05 = signal_true_y_statuses_f05 == "in_support"
true_y_support_mask_f03 = true_y_analysis_mask_f03
true_y_total_weight_f01 = signal_true_y_weights_f01[
    true_y_analysis_mask_f01
].sum()
true_y_total_weight_f03 = signal_true_y_weights_f03[
    true_y_analysis_mask_f03
].sum()
true_y_total_weight_f05 = signal_true_y_weights_f05[
    true_y_analysis_mask_f05
].sum()
true_y_support_total_weight_f03 = signal_true_y_weights_f03[
    true_y_support_mask_f03
].sum()

if min(total_weight_f01, total_weight_f03, total_weight_f05) <= 0:
    raise RuntimeError(
        "At least one f-factor variation has zero total weight under "
        f"the '{weighting_convention}' weighting convention."
    )
if min(
    true_y_total_weight_f01,
    true_y_total_weight_f03,
    true_y_total_weight_f05,
) <= 0:
    raise RuntimeError(
        "At least one true-Y f-factor variation has zero total weight."
    )
if true_y_support_total_weight_f03 <= 0:
    raise RuntimeError(
        "True-Y f=0.3 has zero total weight inside digitized support."
    )

print(f"--- Weighting convention: {weighting_convention}")
if weighting_convention == "colleague":
    print(
        "--- Colleague-compatible convention: Y=1 fallback and "
        "Coleman extrapolation included"
    )
else:
    print(
        "--- Truth convention: per-event inelasticity, "
        + (
            "Coleman extrapolation included"
            if include_extrapolated_weights
            else "digitized support only"
        )
    )
print(f"Valid weighted signal events: {len(signal_scores_weighted)}")
print(f"Invalid-weight events: {n_invalid_weights}")
print(f"Out-of-range events at f={F_CENTRAL}: {n_out_of_range}")
print(
    "--- True-Y digitized-support-only totals: "
    f"f=0.1 {true_y_total_weight_f01:.6g}, "
    f"f=0.3 {true_y_total_weight_f03:.6g}, "
    f"f=0.5 {true_y_total_weight_f05:.6g}"
)
print(
    "--- True-Y digitized-support-only total at f=0.3: "
    f"{true_y_support_total_weight_f03:.6g}"
)

print("--- Coleman weighting diagnostics")
for f_factor, weights_for_f, statuses_for_f, selected_mask, total_weight in (
    (
        F_LOW,
        signal_weights_f01,
        signal_support_statuses_f01,
        analysis_mask_f01,
        total_weight_f01,
    ),
    (
        F_CENTRAL,
        signal_weights_f03,
        signal_support_statuses_f03,
        analysis_mask_f03,
        total_weight_f03,
    ),
    (
        F_HIGH,
        signal_weights_f05,
        signal_support_statuses_f05,
        analysis_mask_f05,
        total_weight_f05,
    ),
):
    print(
        f"    f={f_factor}: selected={np.count_nonzero(selected_mask)}, "
        f"in_support={np.count_nonzero(statuses_for_f == 'in_support')}, "
        f"extrapolated={np.count_nonzero(statuses_for_f == 'extrapolated')}, "
        f"out_of_range={np.count_nonzero(statuses_for_f == 'out_of_range')}, "
        f"selected_sum_w={total_weight:.6g}"
    )

central_analysis_weights = signal_weights_f03[analysis_mask_f03]
sum_signal_weight_squared = np.square(central_analysis_weights).sum()
effective_signal_events = (
    total_weight_f03**2 / sum_signal_weight_squared
    if sum_signal_weight_squared > 0
    else 0.0
)
print(
    "    effective weighted event count at f=0.3: "
    f"{effective_signal_events:.6g}"
)
print(
    "    selected f=0.3 weight quantiles [0, 50%, 90%, 99%, 100%]:",
    np.quantile(
        central_analysis_weights,
        [0.0, 0.5, 0.9, 0.99, 1.0],
    ),
)

largest_weight_indices = np.argsort(signal_weights_f03)[-10:][::-1]
print("--- Ten largest f=0.3 signal weights")
for index in largest_weight_indices:
    print(
        f"    score={signal_scores_weighted[index]:.8g}, "
        f"weight={signal_weights_f03[index]:.8g}, "
        f"lgE_dep={signal_log10_deposited_energy[index]:.8g}, "
        f"status={signal_support_statuses_f03[index]}"
    )




print(f"--- TMVA Classification App    : Using input file: {input_bkg.GetName()}")
for i_event in range(nEvents_B):
    tree_B.GetEntry(i_event)

    passed_hit_filter_float[0] = passed_hit_filter[0]
    nCoincidentPairs_PA_float[0] = nCoincidentPairs_PA[0]
    nHighHits_PA_float[0] = nHighHits_PA[0]
    nCoincidentPairs_inIce_float[0] = nCoincidentPairs_inIce[0]
    nHighHits_inIce_float[0] = nHighHits_inIce[0]

    station_number_float[0] = station_number[0]
    run_number_float[0] = run_number[0]
    event_number_float[0] = event_number[0]

    interaction_type_float[0] = interaction_type[0]

    trigger_time_float[0] = trigger_time[0]

    true_source_theta_float[0] = true_source_theta[0]
    true_source_phi_float[0] = true_source_phi[0]

    EvaluateMVA[0] = reader.EvaluateMVA(methodName)
    background_scores_all.append(float(EvaluateMVA[0]))
    for variable_name in ALL_BDT_VARIABLES:
        background_variable_values[variable_name].append(
            float(background_variable_leaves[variable_name].GetValue())
        )

    testTree_B.Fill()
    hist_B.Fill(EvaluateMVA[0])
print(f"--- BACKGROUND: {testTree_B.GetEntries()} events")
print("--- End of event loop (BACKGROUND)")

graph_roc_original = TGraph()
graph_roc_weighted = TGraph()
graph_roc_true_y = TGraph()
graph_roc_weighted_band = ROOT.TGraphAsymmErrors()

cutValues = np.array([])
cut = -1.0
while cut <= 0.95:
    cutValues = np.append(cutValues, cut)
    cut += 0.01
while cut > 0.95 and cut <= 1.0:
    cutValues = np.append(cutValues, cut)
    cut += 0.001

nCounts_S = testTree_S.GetEntries()
nCounts_B = testTree_B.GetEntries()

if nCounts_S <= 0 or nCounts_B <= 0:
    raise RuntimeError("Signal and background test trees must both be non-empty.")

selected_index = int(
    np.argmin(np.abs(cutValues - targetCut))
)

efficiency_original_values = []
efficiency_f01_values = []
efficiency_f03_values = []
efficiency_f05_values = []
efficiency_true_y_f01_values = []
efficiency_true_y_f03_values = []
efficiency_true_y_f05_values = []
efficiency_true_y_support_f03_values = []
background_rejection_values = []
background_passing_counts = []

for i_cut, cut in enumerate(cutValues):
    threshold = f"{method} > {cut}"

    # Keep the original calculation exactly as it was before weighting.
    count_S = testTree_S.GetEntries(threshold)
    count_B = testTree_B.GetEntries(threshold)
    eff_original = count_S / nCounts_S
    rej = 1 - count_B / nCounts_B

    passed_mask = signal_scores_weighted > cut
    eff_f01 = signal_weights_f01[
        analysis_mask_f01 & passed_mask
    ].sum() / total_weight_f01
    eff_f03 = signal_weights_f03[
        analysis_mask_f03 & passed_mask
    ].sum() / total_weight_f03
    eff_f05 = signal_weights_f05[
        analysis_mask_f05 & passed_mask
    ].sum() / total_weight_f05

    true_y_passed_mask = signal_scores_true_y > cut
    eff_true_y_f01 = signal_true_y_weights_f01[
        true_y_analysis_mask_f01 & true_y_passed_mask
    ].sum() / true_y_total_weight_f01
    eff_true_y_f03 = signal_true_y_weights_f03[
        true_y_analysis_mask_f03 & true_y_passed_mask
    ].sum() / true_y_total_weight_f03
    eff_true_y_f05 = signal_true_y_weights_f05[
        true_y_analysis_mask_f05 & true_y_passed_mask
    ].sum() / true_y_total_weight_f05
    eff_true_y_support_f03 = signal_true_y_weights_f03[
        true_y_support_mask_f03 & true_y_passed_mask
    ].sum() / true_y_support_total_weight_f03

    efficiency_original_values.append(eff_original)
    efficiency_f01_values.append(eff_f01)
    efficiency_f03_values.append(eff_f03)
    efficiency_f05_values.append(eff_f05)
    efficiency_true_y_f01_values.append(eff_true_y_f01)
    efficiency_true_y_f03_values.append(eff_true_y_f03)
    efficiency_true_y_f05_values.append(eff_true_y_f05)
    efficiency_true_y_support_f03_values.append(
        eff_true_y_support_f03
    )
    background_rejection_values.append(rej)
    background_passing_counts.append(count_B)

    graph_roc_original.SetPoint(
        graph_roc_original.GetN(),
        eff_original,
        rej,
    )
    graph_roc_weighted.SetPoint(
        graph_roc_weighted.GetN(),
        eff_f03,
        rej,
    )
    graph_roc_true_y.SetPoint(
        graph_roc_true_y.GetN(),
        eff_true_y_support_f03,
        rej,
    )

    eff_band_low = min(eff_f01, eff_f03, eff_f05)
    eff_band_high = max(eff_f01, eff_f03, eff_f05)
    graph_roc_weighted_band.SetPoint(
        graph_roc_weighted_band.GetN(),
        eff_f03,
        rej,
    )
    graph_roc_weighted_band.SetPointError(
        graph_roc_weighted_band.GetN() - 1,
        eff_f03 - eff_band_low,
        eff_band_high - eff_f03,
        0.0,
        0.0,
    )

    if i_cut == selected_index:
        cut_selected = float(cut)
        eff_selected = eff_original
        eff_selected_f01 = eff_f01
        eff_selected_f03 = eff_f03
        eff_selected_f05 = eff_f05
        eff_selected_true_y_f01 = eff_true_y_f01
        eff_selected_true_y_f03 = eff_true_y_f03
        eff_selected_true_y_f05 = eff_true_y_f05
        eff_selected_true_y_support_f03 = eff_true_y_support_f03
        rej_selected = rej
        count_S_selected = count_S
        count_B_selected = count_B

efficiency_original_values = np.asarray(
    efficiency_original_values, dtype=float
)
efficiency_f01_values = np.asarray(efficiency_f01_values, dtype=float)
efficiency_f03_values = np.asarray(efficiency_f03_values, dtype=float)
efficiency_f05_values = np.asarray(efficiency_f05_values, dtype=float)
efficiency_true_y_f01_values = np.asarray(
    efficiency_true_y_f01_values, dtype=float
)
efficiency_true_y_f03_values = np.asarray(
    efficiency_true_y_f03_values, dtype=float
)
efficiency_true_y_f05_values = np.asarray(
    efficiency_true_y_f05_values, dtype=float
)
efficiency_true_y_support_f03_values = np.asarray(
    efficiency_true_y_support_f03_values, dtype=float
)
background_rejection_values = np.asarray(
    background_rejection_values, dtype=float
)
background_passing_counts = np.asarray(
    background_passing_counts, dtype=int
)


def compute_roc_auc(signal_efficiency, background_rejection):
    """Integrate background rejection versus signal efficiency."""
    x_values = np.asarray(signal_efficiency, dtype=float)
    y_values = np.asarray(background_rejection, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)

    if np.count_nonzero(finite) < 2:
        return float("nan")

    x_values = x_values[finite]
    y_values = y_values[finite]
    order = np.argsort(x_values, kind="mergesort")

    # The plotted axes are sensitivity versus specificity. The area under
    # this curve is the conventional ROC AUC when the full endpoints are
    # included.
    return float(
        np.clip(
            np.trapezoid(y_values[order], x_values[order]),
            0.0,
            1.0,
        )
    )


auc_original = compute_roc_auc(
    efficiency_original_values,
    background_rejection_values,
)
auc_weighted = compute_roc_auc(
    efficiency_f03_values,
    background_rejection_values,
)
auc_true_y_support = compute_roc_auc(
    efficiency_true_y_support_f03_values,
    background_rejection_values,
)

print(f"*** ROC AUC (original): {auc_original:.6f}")
print(f"*** ROC AUC (Y=1 fallback, f=0.3): {auc_weighted:.6f}")
print(
    "*** ROC AUC (true Y, digitized support only, f=0.3): "
    f"{auc_true_y_support:.6f}"
)

# A filled polygon is used for the ROC systematic band because the
# f-factor uncertainty is horizontal (signal-efficiency direction).
roc_efficiency_low = np.minimum.reduce(
    (
        efficiency_f01_values,
        efficiency_f03_values,
        efficiency_f05_values,
    )
)
roc_efficiency_high = np.maximum.reduce(
    (
        efficiency_f01_values,
        efficiency_f03_values,
        efficiency_f05_values,
    )
)
graph_roc_weighted_band_polygon = TGraph(2 * len(cutValues))
for index in range(len(cutValues)):
    graph_roc_weighted_band_polygon.SetPoint(
        index,
        roc_efficiency_low[index],
        background_rejection_values[index],
    )
for reverse_index in range(len(cutValues)):
    source_index = len(cutValues) - 1 - reverse_index
    graph_roc_weighted_band_polygon.SetPoint(
        len(cutValues) + reverse_index,
        roc_efficiency_high[source_index],
        background_rejection_values[source_index],
    )

count_from_array = np.count_nonzero(signal_scores_all > cut_selected)
if count_from_array != count_S_selected:
    raise RuntimeError(
        "Signal-score consistency check failed: "
        f"tree count={count_S_selected}, array count={count_from_array}."
    )

print(f"*** Requested target cut: {targetCut}")
print(f"*** Actual selected cut: {cut_selected}")
print(f"*** Signal Efficiency (original): {eff_selected}")
print(
    f"*** Signal Efficiency ({weighting_curve_label}, f=0.3): "
    f"{eff_selected_f03}"
)
print(
    "*** Signal Efficiency f-band [f=0.1, f=0.5]: "
    f"[{min(eff_selected_f01, eff_selected_f03, eff_selected_f05)}, "
    f"{max(eff_selected_f01, eff_selected_f03, eff_selected_f05)}]"
)
print(
    "*** Signal Efficiency (true Y, digitized support only, f=0.3): "
    f"{eff_selected_true_y_f03}"
)
print(
    "*** True-Y support-only f-band [f=0.1, f=0.5]: "
    f"[{min(eff_selected_true_y_f01, eff_selected_true_y_f03, eff_selected_true_y_f05)}, "
    f"{max(eff_selected_true_y_f01, eff_selected_true_y_f03, eff_selected_true_y_f05)}]"
)
print(f"*** Background Rejection: {rej_selected}")
nEvents_FP = 0
i = 0
info_FP = {}
bkgRuns = []
bkgEvents = []
threshold = f"{method} > {cut_selected}"
while nEvents_FP < count_B_selected:
    testTree_B.GetEntry(i)
    if EvaluateMVA[0] > cut_selected:
        run_bkg = int(testTree_B.run_number)
        event_bkg = int(testTree_B.event_number)
        bkgRuns.append(run_bkg)
        bkgEvents.append(event_bkg)
        info_FP[str(run_bkg)] = []
        nEvents_FP += 1
    i += 1
print(f"*** Number of False Positive Events: {nEvents_FP}")

for i, run in enumerate(bkgRuns):
    info_FP[str(run)].append(bkgEvents[i])
with open(dir_out+jsonFileName, "w") as file:
    json.dump(info_FP, file)

canvas = TCanvas("c1", histTitle, 10, 10, 1150, 600)
ROOT.gStyle.SetOptStat(0)

# Normalize each Coleman-weighted shape selected by the active convention to
# the number of signal events. Page 2 therefore remains in
# event-count-equivalent units; it is a shape comparison, not an absolute
# event-rate prediction.
weighted_count_hists = []
for source_hist, output_name in (
    (hist_S_weighted_f01, "hist_S_weighted_f01_event_counts"),
    (hist_S_weighted_f03, "hist_S_weighted_f03_event_counts"),
    (hist_S_weighted_f05, "hist_S_weighted_f05_event_counts"),
):
    count_hist = source_hist.Clone(output_name)
    count_integral = count_hist.Integral(1, count_hist.GetNbinsX())
    if count_integral <= 0:
        raise RuntimeError(
            f"Cannot normalize {output_name}: histogram integral is zero."
        )
    count_hist.Scale(nCounts_S / count_integral)
    weighted_count_hists.append(count_hist)

(
    hist_S_weighted_f01_counts,
    hist_S_weighted_f03_counts,
    hist_S_weighted_f05_counts,
) = weighted_count_hists
hist_S_weighted_f03_counts.SetLineColor(ROOT.kBlue+2)
hist_S_weighted_f03_counts.SetLineWidth(3)
hist_S_weighted_f03_counts.SetFillStyle(0)

true_y_weighted_count_hists = []
for source_hist, output_name in (
    (hist_S_true_y_f01, "hist_S_true_y_f01_event_counts"),
    (hist_S_true_y_f03, "hist_S_true_y_f03_event_counts"),
    (hist_S_true_y_f05, "hist_S_true_y_f05_event_counts"),
):
    count_hist = source_hist.Clone(output_name)
    count_integral = count_hist.Integral(1, count_hist.GetNbinsX())
    if count_integral <= 0:
        raise RuntimeError(
            f"Cannot normalize {output_name}: histogram integral is zero."
        )
    count_hist.Scale(nCounts_S / count_integral)
    true_y_weighted_count_hists.append(count_hist)

(
    hist_S_true_y_f01_counts,
    hist_S_true_y_f03_counts,
    hist_S_true_y_f05_counts,
) = true_y_weighted_count_hists
hist_S_true_y_f03_counts.SetLineColor(ROOT.kGreen+2)
hist_S_true_y_f03_counts.SetLineWidth(3)
hist_S_true_y_f03_counts.SetFillStyle(0)

graph_distribution_band = ROOT.TGraphAsymmErrors(nbin)
distribution_band_maximum = 0.0
for bin_index in range(1, nbin + 1):
    x_value = hist_S_weighted_f03_counts.GetBinCenter(bin_index)
    half_width = 0.5 * hist_S_weighted_f03_counts.GetBinWidth(bin_index)
    central_value = hist_S_weighted_f03_counts.GetBinContent(bin_index)
    varied_values = (
        hist_S_weighted_f01_counts.GetBinContent(bin_index),
        central_value,
        hist_S_weighted_f05_counts.GetBinContent(bin_index),
    )
    band_low = min(varied_values)
    band_high = max(varied_values)
    distribution_band_maximum = max(distribution_band_maximum, band_high)
    graph_distribution_band.SetPoint(
        bin_index - 1, x_value, central_value
    )
    graph_distribution_band.SetPointError(
        bin_index - 1,
        half_width,
        half_width,
        central_value - band_low,
        band_high - central_value,
    )
graph_distribution_band.SetFillColorAlpha(ROOT.kMagenta-9, 0.40)
graph_distribution_band.SetLineColor(ROOT.kMagenta-7)

# Page 1: original unweighted distributions.
canvas.cd()
canvas.SetRightMargin(0.30)
canvas.SetLogy(True)
hist_B.SetMinimum(0.5)
hist_B.SetMaximum(2.0 * max(hist_B.GetMaximum(), hist_S.GetMaximum()))
hist_B.Draw("hist")
hist_S.Draw("hist same")

legend_page1 = TLegend(0.71, 0.64, 0.88, 0.90)
legend_page1.SetTextSize(0.028)
legend_page1.SetHeader("Event Class", "")
legend_page1.AddEntry(hist_S, "Signal (unweighted)", "f")
legend_page1.AddEntry(hist_B, "Background", "f")
legend_page1.Draw()
canvas.Print(dir_out + graphFileName + "(", "pdf")
canvas.Clear("D")

# Page 2: original distributions plus the central weighted signal, all in
# event-count-equivalent normalization.
canvas.cd()
canvas.SetRightMargin(0.30)
canvas.SetLogy(True)
page2_maximum = 2.0 * max(
    hist_B.GetMaximum(),
    hist_S.GetMaximum(),
    hist_S_weighted_f03_counts.GetMaximum(),
)
hist_B.SetMaximum(page2_maximum)
hist_B.Draw("hist")
hist_S.Draw("hist same")
hist_S_weighted_f03_counts.Draw("hist same")

cut_line_page2 = TLine(cut_selected, 0.5, cut_selected, page2_maximum)
cut_line_page2.SetLineStyle(2)
cut_line_page2.SetLineWidth(2)
cut_line_page2.Draw("same")

legend_page2 = TLegend(0.71, 0.54, 0.93, 0.90)
legend_page2.SetTextSize(0.024)
legend_page2.SetHeader("Event Class", "")
legend_page2.AddEntry(hist_S, "Signal (unweighted)", "f")
legend_page2.AddEntry(
    hist_S_weighted_f03_counts,
    weighting_curve_label,
    "l",
)
legend_page2.AddEntry(hist_B, "Background", "f")
legend_page2.AddEntry(cut_line_page2, f"BDT cut = {cut_selected:.3f}", "l")
legend_page2.Draw()
canvas.Print(dir_out + graphFileName, "pdf")
canvas.Clear("D")

# Page 3: original distributions plus the true-inelasticity,
# digitized-support-only weighted signal.
canvas.cd()
canvas.SetRightMargin(0.30)
canvas.SetLogy(True)
page3_maximum = 2.0 * max(
    hist_B.GetMaximum(),
    hist_S.GetMaximum(),
    hist_S_true_y_f03_counts.GetMaximum(),
)
hist_B.SetMaximum(page3_maximum)
hist_B.Draw("hist")
hist_S.Draw("hist same")
hist_S_true_y_f03_counts.Draw("hist same")

cut_line_page3 = TLine(cut_selected, 0.5, cut_selected, page3_maximum)
cut_line_page3.SetLineStyle(2)
cut_line_page3.SetLineWidth(2)
cut_line_page3.Draw("same")

legend_page3 = TLegend(0.71, 0.54, 0.97, 0.90)
legend_page3.SetTextSize(0.024)
legend_page3.SetHeader("Event Class", "")
legend_page3.AddEntry(hist_S, "Signal (unweighted)", "f")
legend_page3.AddEntry(
    hist_S_true_y_f03_counts,
    "Signal weighted (true Y, support only)",
    "l",
)
legend_page3.AddEntry(hist_B, "Background", "f")
legend_page3.AddEntry(cut_line_page3, f"BDT cut = {cut_selected:.3f}", "l")
legend_page3.Draw()
canvas.Print(dir_out + graphFileName, "pdf")
canvas.Clear("D")

graph_roc_original.SetLineColor(ROOT.kAzure+2)
graph_roc_original.SetLineWidth(3)
graph_roc_weighted.SetLineColor(ROOT.kRed+1)
graph_roc_weighted.SetLineWidth(3)
graph_roc_true_y.SetLineColor(ROOT.kGreen+2)
graph_roc_true_y.SetLineWidth(3)
graph_roc_weighted_band.SetFillColorAlpha(ROOT.kMagenta-9, 0.40)
graph_roc_weighted_band.SetLineColor(ROOT.kMagenta-7)
graph_roc_weighted_band_polygon.SetFillColorAlpha(
    ROOT.kMagenta-9, 0.40
)
graph_roc_weighted_band_polygon.SetLineColor(ROOT.kMagenta-7)

graph_roc_original.SetTitle(
    f"ROC comparison: {method} (S{station})"
)
graph_roc_original.GetXaxis().SetTitle("Signal efficiency (Sensitivity)")
graph_roc_original.GetYaxis().SetTitle("Background rejection (Specificity)")

roc_selected_original = TGraph(1)
roc_selected_original.SetPoint(0, eff_selected, rej_selected)
roc_selected_original.SetMarkerStyle(20)
roc_selected_original.SetMarkerColor(ROOT.kAzure+2)
roc_selected_original.SetMarkerSize(1.2)

roc_selected_weighted = TGraph(1)
roc_selected_weighted.SetPoint(0, eff_selected_f03, rej_selected)
roc_selected_weighted.SetMarkerStyle(21)
roc_selected_weighted.SetMarkerColor(ROOT.kRed+1)
roc_selected_weighted.SetMarkerSize(1.2)

roc_selected_true_y = TGraph(1)
roc_selected_true_y.SetPoint(
    0,
    eff_selected_true_y_support_f03,
    rej_selected,
)
roc_selected_true_y.SetMarkerStyle(22)
roc_selected_true_y.SetMarkerColor(ROOT.kGreen+2)
roc_selected_true_y.SetMarkerSize(1.2)

def draw_roc_page(true_y_support_only=False, x_min=0.0, x_max=1.01, y_min=0.0, y_max=1.01):

    canvas.cd()
    canvas.SetRightMargin(0.08)
    canvas.SetLogy(False)
    canvas.SetGrid()
    graph_roc_original.Draw("AL")
    graph_roc_original.GetXaxis().SetLimits(x_min, x_max)
    graph_roc_original.GetYaxis().SetRangeUser(y_min, y_max)
    graph_roc_original.Draw("L same")
    roc_selected_original.Draw("P same")

    roc_legend = TLegend(0.12, 0.14, 0.55, 0.40)
    roc_legend.SetTextSize(0.024)
    original_legend_label = (
        f"Original (signal unweighted), AUC = {auc_original:.4f}"
    )
    roc_legend.AddEntry(
        graph_roc_original,
        original_legend_label,
        "l",
    )

    if true_y_support_only:
        graph_roc_true_y.Draw("L same")
        roc_selected_true_y.Draw("P same")
        true_y_legend_label = (
            "Signal weighted (true Y, support only), "
            f"AUC = {auc_true_y_support:.4f}"
        )
        roc_legend.AddEntry(
            graph_roc_true_y,
            true_y_legend_label,
            "l",
        )
    else:
        graph_roc_weighted.Draw("L same")
        roc_selected_weighted.Draw("P same")
        weighted_legend_label = (
            f"{weighting_curve_label}, AUC = {auc_weighted:.4f}"
        )
        roc_legend.AddEntry(
            graph_roc_weighted,
            weighted_legend_label,
            "l",
        )

    roc_legend.Draw()
    canvas.Print(dir_out + graphFileName, "pdf")
    canvas.Clear("D")

# Page 4: ROC comparison: original vs. Y=1 fallback.
draw_roc_page()

# Page 5: ROC comparison: original vs. true Y support only.
draw_roc_page(True)

# Page 6: signal efficiency as a function of BDT cut.
graph_efficiency_original = TGraph()
graph_efficiency_weighted = TGraph()
graph_efficiency_band = ROOT.TGraphAsymmErrors()
graph_efficiency_true_y = TGraph()
graph_efficiency_true_y_band = ROOT.TGraphAsymmErrors()
for index, cut_value in enumerate(cutValues):
    eff_original_value = efficiency_original_values[index]
    eff_f01_value = efficiency_f01_values[index]
    eff_f03_value = efficiency_f03_values[index]
    eff_f05_value = efficiency_f05_values[index]
    eff_low = min(eff_f01_value, eff_f03_value, eff_f05_value)
    eff_high = max(eff_f01_value, eff_f03_value, eff_f05_value)

    graph_efficiency_original.SetPoint(
        index, cut_value, eff_original_value
    )
    graph_efficiency_weighted.SetPoint(
        index, cut_value, eff_f03_value
    )
    graph_efficiency_band.SetPoint(index, cut_value, eff_f03_value)
    graph_efficiency_band.SetPointError(
        index,
        0.0,
        0.0,
        eff_f03_value - eff_low,
        eff_high - eff_f03_value,
    )

    eff_true_y_f01_value = efficiency_true_y_f01_values[index]
    eff_true_y_f03_value = efficiency_true_y_f03_values[index]
    eff_true_y_f05_value = efficiency_true_y_f05_values[index]
    eff_true_y_low = min(
        eff_true_y_f01_value,
        eff_true_y_f03_value,
        eff_true_y_f05_value,
    )
    eff_true_y_high = max(
        eff_true_y_f01_value,
        eff_true_y_f03_value,
        eff_true_y_f05_value,
    )
    graph_efficiency_true_y.SetPoint(
        index,
        cut_value,
        eff_true_y_f03_value,
    )
    graph_efficiency_true_y_band.SetPoint(
        index,
        cut_value,
        eff_true_y_f03_value,
    )
    graph_efficiency_true_y_band.SetPointError(
        index,
        0.0,
        0.0,
        eff_true_y_f03_value - eff_true_y_low,
        eff_true_y_high - eff_true_y_f03_value,
    )

graph_efficiency_original.SetLineColor(ROOT.kAzure+2)
graph_efficiency_original.SetLineWidth(3)
graph_efficiency_weighted.SetLineColor(ROOT.kRed+1)
graph_efficiency_weighted.SetLineWidth(3)
graph_efficiency_band.SetFillColorAlpha(ROOT.kRed-9, 0.30)
graph_efficiency_band.SetLineColor(ROOT.kRed-7)
graph_efficiency_true_y.SetLineColor(ROOT.kGreen+2)
graph_efficiency_true_y.SetLineWidth(3)
graph_efficiency_true_y_band.SetFillColorAlpha(ROOT.kGreen-9, 0.30)
graph_efficiency_true_y_band.SetLineColor(ROOT.kGreen-7)

# The colleague plot's 68% interval belongs to its expected-background
# estimate. Here the available quantity is the finite TMVA background test
# sample, so use an exact 68% binomial interval for the number of test events
# passing each cut. The counts are mapped logarithmically onto the left-pad
# coordinates and labelled with a separate right-hand axis.
background_confidence_level = 0.682689492137
background_count_lower = []
background_count_upper = []
for count_background in background_passing_counts:
    lower_fraction = ROOT.TEfficiency.ClopperPearson(
        nCounts_B,
        int(count_background),
        background_confidence_level,
        False,
    )
    upper_fraction = ROOT.TEfficiency.ClopperPearson(
        nCounts_B,
        int(count_background),
        background_confidence_level,
        True,
    )
    background_count_lower.append(nCounts_B * lower_fraction)
    background_count_upper.append(nCounts_B * upper_fraction)

background_count_lower = np.asarray(background_count_lower, dtype=float)
background_count_upper = np.asarray(background_count_upper, dtype=float)
background_plot_floor = 0.5
background_plot_ceiling = max(
    1.0,
    1.15 * float(np.max(background_count_upper)),
)
background_log_floor = np.log10(background_plot_floor)
background_log_ceiling = np.log10(background_plot_ceiling)
efficiency_x_min = cut_selected - efficiency_half_window
efficiency_x_max = cut_selected + efficiency_half_window
original_efficiency_y_max = 1.01
weighted_efficiency_y_max = 0.12
true_y_window_mask = (
    (cutValues >= efficiency_x_min)
    & (cutValues <= efficiency_x_max)
)
true_y_window_maximum = np.max(
    np.concatenate(
        (
            efficiency_true_y_f01_values[true_y_window_mask],
            efficiency_true_y_f03_values[true_y_window_mask],
            efficiency_true_y_f05_values[true_y_window_mask],
        )
    )
)
true_y_efficiency_y_max = min(
    1.01,
    max(1.0e-12, 1.15 * float(true_y_window_maximum)),
)

def background_count_to_pad_y(values, pad_y_max):
    values = np.asarray(values, dtype=float)
    clipped = np.clip(
        values,
        background_plot_floor,
        background_plot_ceiling,
    )
    normalized_log_y = (
        (np.log10(clipped) - background_log_floor)
        / (background_log_ceiling - background_log_floor)
    )
    return pad_y_max * normalized_log_y

def build_background_graphs(pad_y_max):
    background_count_pad_y = background_count_to_pad_y(
        background_passing_counts,
        pad_y_max,
    )
    background_lower_pad_y = background_count_to_pad_y(
        background_count_lower,
        pad_y_max,
    )
    background_upper_pad_y = background_count_to_pad_y(
        background_count_upper,
        pad_y_max,
    )

    graph_passing = TGraph()
    graph_confidence = ROOT.TGraphAsymmErrors()
    for index, cut_value in enumerate(cutValues):
        central_y = background_count_pad_y[index]
        graph_passing.SetPoint(index, cut_value, central_y)
        graph_confidence.SetPoint(index, cut_value, central_y)
        graph_confidence.SetPointError(
            index,
            0.0,
            0.0,
            central_y - background_lower_pad_y[index],
            background_upper_pad_y[index] - central_y,
        )

    graph_passing.SetLineColor(ROOT.kViolet+2)
    graph_passing.SetLineWidth(3)
    graph_confidence.SetFillColorAlpha(ROOT.kViolet-9, 0.35)
    graph_confidence.SetLineColor(ROOT.kViolet-7)
    return graph_passing, graph_confidence

(
    graph_background_passing_original,
    graph_background_confidence_original,
) = build_background_graphs(original_efficiency_y_max)
(
    graph_background_passing_weighted,
    graph_background_confidence_weighted,
) = build_background_graphs(weighted_efficiency_y_max)
(
    graph_background_passing_true_y,
    graph_background_confidence_true_y,
) = build_background_graphs(true_y_efficiency_y_max)

# Match the collaborator-style two-panel layout: unweighted on the left,
# Coleman weighted with the f-factor band on the right. Use a new canvas
# rather than re-dividing the ROC canvas; ROOT can otherwise retain old pad
# primitives even after Clear("D").
efficiency_canvas = TCanvas(
    "c_efficiency",
    "Signal efficiency comparison",
    10,
    10,
    1600,
    650,
)
efficiency_canvas.Divide(2, 1)

efficiency_pad_original = efficiency_canvas.cd(1)
efficiency_pad_original.SetGrid()
efficiency_pad_original.SetLeftMargin(0.14)
efficiency_pad_original.SetRightMargin(0.16)
graph_efficiency_original.SetTitle(
    "Unweighted Signal Efficiency"
)
graph_efficiency_original.GetXaxis().SetTitle("BDT cut value")
graph_efficiency_original.GetYaxis().SetTitle("Signal efficiency")
graph_efficiency_original.Draw("AL")
graph_efficiency_original.GetXaxis().SetLimits(
    efficiency_x_min,
    efficiency_x_max,
)
graph_efficiency_original.GetYaxis().SetRangeUser(
    0.0,
    original_efficiency_y_max,
)
graph_background_confidence_original.Draw("3 same")
graph_background_passing_original.Draw("L same")
graph_efficiency_original.Draw("L same")

cut_line_efficiency_original = TLine(
    cut_selected, 0.0, cut_selected, original_efficiency_y_max
)
cut_line_efficiency_original.SetLineStyle(2)
cut_line_efficiency_original.SetLineWidth(2)
cut_line_efficiency_original.Draw("same")

legend_efficiency_original = TLegend(0.38, 0.64, 0.82, 0.88)
legend_efficiency_original.SetTextSize(0.030)
legend_efficiency_original.AddEntry(
    graph_efficiency_original,
    "Signal efficiency",
    "l",
)
legend_efficiency_original.AddEntry(
    cut_line_efficiency_original,
    f"Selected cut = {cut_selected:.3f}",
    "l",
)
legend_efficiency_original.AddEntry(
    graph_background_passing_original,
    "Background passing events",
    "l",
)
legend_efficiency_original.AddEntry(
    graph_background_confidence_original,
    "68% binomial CI",
    "f",
)
legend_efficiency_original.Draw()

background_axis_original = ROOT.TGaxis(
    efficiency_x_max,
    0.0,
    efficiency_x_max,
    original_efficiency_y_max,
    background_plot_floor,
    background_plot_ceiling,
    510,
    "+LG",
)
background_axis_original.SetTitle("Background passing events")
background_axis_original.SetTitleColor(ROOT.kViolet+2)
background_axis_original.SetLabelColor(ROOT.kViolet+2)
background_axis_original.SetLineColor(ROOT.kViolet+2)
background_axis_original.SetTitleOffset(1.25)
background_axis_original.Draw()

efficiency_pad_weighted = efficiency_canvas.cd(2)
efficiency_pad_weighted.SetGrid()
efficiency_pad_weighted.SetLeftMargin(0.14)
efficiency_pad_weighted.SetRightMargin(0.16)
graph_efficiency_weighted.SetTitle(
    f"{weighting_curve_label} (f=0.3, band [0.1, 0.5])"
)
graph_efficiency_weighted.GetXaxis().SetTitle("BDT cut value")
graph_efficiency_weighted.GetYaxis().SetTitle("Signal efficiency")
graph_efficiency_weighted.Draw("AL")
graph_efficiency_weighted.GetXaxis().SetLimits(
    efficiency_x_min,
    efficiency_x_max,
)
graph_efficiency_weighted.GetYaxis().SetRangeUser(
    0.0,
    weighted_efficiency_y_max,
)
graph_background_confidence_weighted.Draw("3 same")
graph_background_passing_weighted.Draw("L same")
graph_efficiency_band.Draw("3 same")
graph_efficiency_weighted.Draw("L same")

cut_line_efficiency_weighted = TLine(
    cut_selected, 0.0, cut_selected, weighted_efficiency_y_max
)
cut_line_efficiency_weighted.SetLineStyle(2)
cut_line_efficiency_weighted.SetLineWidth(2)
cut_line_efficiency_weighted.Draw("same")

legend_efficiency_weighted = TLegend(0.4, 0.57, 0.82, 0.88)
legend_efficiency_weighted.SetTextSize(0.028)
legend_efficiency_weighted.AddEntry(
    graph_efficiency_weighted,
    "Signal efficiency (f=0.3)",
    "l",
)
legend_efficiency_weighted.AddEntry(
    graph_efficiency_band,
    "f-factor band [0.1, 0.5]",
    "f",
)
legend_efficiency_weighted.AddEntry(
    cut_line_efficiency_weighted,
    f"Selected cut = {cut_selected:.3f}",
    "l",
)
legend_efficiency_weighted.AddEntry(
    graph_background_passing_weighted,
    "Background passing events",
    "l",
)
legend_efficiency_weighted.AddEntry(
    graph_background_confidence_weighted,
    "68% binomial CI",
    "f",
)
legend_efficiency_weighted.Draw()

background_axis_weighted = ROOT.TGaxis(
    efficiency_x_max,
    0.0,
    efficiency_x_max,
    weighted_efficiency_y_max,
    background_plot_floor,
    background_plot_ceiling,
    510,
    "+LG",
)
background_axis_weighted.SetTitle("Background passing events")
background_axis_weighted.SetTitleColor(ROOT.kViolet+2)
background_axis_weighted.SetLabelColor(ROOT.kViolet+2)
background_axis_weighted.SetLineColor(ROOT.kViolet+2)
background_axis_weighted.SetTitleOffset(1.25)
background_axis_weighted.Draw()
efficiency_canvas.Modified()
efficiency_canvas.Update()
efficiency_canvas.Print(dir_out + graphFileName, "pdf")
efficiency_canvas.Close()

# Page 7: unweighted versus true-inelasticity Coleman weighting.
true_y_efficiency_canvas = TCanvas(
    "c_efficiency_true_y",
    "True-Y signal efficiency comparison",
    10,
    10,
    1600,
    650,
)
true_y_efficiency_canvas.Divide(2, 1)

true_y_pad_original = true_y_efficiency_canvas.cd(1)
true_y_pad_original.SetGrid()
true_y_pad_original.SetLeftMargin(0.14)
true_y_pad_original.SetRightMargin(0.16)
graph_efficiency_original.SetTitle("Unweighted Signal Efficiency")
graph_efficiency_original.Draw("AL")
graph_efficiency_original.GetXaxis().SetLimits(
    efficiency_x_min,
    efficiency_x_max,
)
graph_efficiency_original.GetYaxis().SetRangeUser(
    0.0,
    original_efficiency_y_max,
)
graph_background_confidence_original.Draw("3 same")
graph_background_passing_original.Draw("L same")
graph_efficiency_original.Draw("L same")

cut_line_true_y_original = TLine(
    cut_selected,
    0.0,
    cut_selected,
    original_efficiency_y_max,
)
cut_line_true_y_original.SetLineStyle(2)
cut_line_true_y_original.SetLineWidth(2)
cut_line_true_y_original.Draw("same")

legend_true_y_original = TLegend(0.38, 0.64, 0.82, 0.88)
legend_true_y_original.SetTextSize(0.030)
legend_true_y_original.AddEntry(
    graph_efficiency_original,
    "Signal efficiency",
    "l",
)
legend_true_y_original.AddEntry(
    cut_line_true_y_original,
    f"Selected cut = {cut_selected:.3f}",
    "l",
)
legend_true_y_original.AddEntry(
    graph_background_passing_original,
    "Background passing events",
    "l",
)
legend_true_y_original.AddEntry(
    graph_background_confidence_original,
    "68% binomial CI",
    "f",
)
legend_true_y_original.Draw()

background_axis_true_y_original = ROOT.TGaxis(
    efficiency_x_max,
    0.0,
    efficiency_x_max,
    original_efficiency_y_max,
    background_plot_floor,
    background_plot_ceiling,
    510,
    "+LG",
)
background_axis_true_y_original.SetTitle("Background passing events")
background_axis_true_y_original.SetTitleColor(ROOT.kViolet+2)
background_axis_true_y_original.SetLabelColor(ROOT.kViolet+2)
background_axis_true_y_original.SetLineColor(ROOT.kViolet+2)
background_axis_true_y_original.SetTitleOffset(1.25)
background_axis_true_y_original.Draw()

true_y_pad_weighted = true_y_efficiency_canvas.cd(2)
true_y_pad_weighted.SetGrid()
true_y_pad_weighted.SetLeftMargin(0.14)
true_y_pad_weighted.SetRightMargin(0.16)
graph_efficiency_true_y.SetTitle(
    "Signal weighted (true Y, support only)"
)
graph_efficiency_true_y.GetXaxis().SetTitle("BDT cut value")
graph_efficiency_true_y.GetYaxis().SetTitle("Signal efficiency")
graph_efficiency_true_y.Draw("AL")
graph_efficiency_true_y.GetXaxis().SetLimits(
    efficiency_x_min,
    efficiency_x_max,
)
graph_efficiency_true_y.GetYaxis().SetRangeUser(
    0.0,
    true_y_efficiency_y_max,
)
graph_background_confidence_true_y.Draw("3 same")
graph_background_passing_true_y.Draw("L same")
graph_efficiency_true_y_band.Draw("3 same")
graph_efficiency_true_y.Draw("L same")

cut_line_efficiency_true_y = TLine(
    cut_selected,
    0.0,
    cut_selected,
    true_y_efficiency_y_max,
)
cut_line_efficiency_true_y.SetLineStyle(2)
cut_line_efficiency_true_y.SetLineWidth(2)
cut_line_efficiency_true_y.Draw("same")

legend_efficiency_true_y = TLegend(0.38, 0.57, 0.82, 0.88)
legend_efficiency_true_y.SetTextSize(0.028)
legend_efficiency_true_y.AddEntry(
    graph_efficiency_true_y,
    "Signal efficiency (true Y, f=0.3)",
    "l",
)
legend_efficiency_true_y.AddEntry(
    graph_efficiency_true_y_band,
    "f-factor band [0.1, 0.5]",
    "f",
)
legend_efficiency_true_y.AddEntry(
    cut_line_efficiency_true_y,
    f"Selected cut = {cut_selected:.3f}",
    "l",
)
legend_efficiency_true_y.AddEntry(
    graph_background_passing_true_y,
    "Background passing events",
    "l",
)
legend_efficiency_true_y.AddEntry(
    graph_background_confidence_true_y,
    "68% binomial CI",
    "f",
)
legend_efficiency_true_y.Draw()

background_axis_true_y = ROOT.TGaxis(
    efficiency_x_max,
    0.0,
    efficiency_x_max,
    true_y_efficiency_y_max,
    background_plot_floor,
    background_plot_ceiling,
    510,
    "+LG",
)
background_axis_true_y.SetTitle("Background passing events")
background_axis_true_y.SetTitleColor(ROOT.kViolet+2)
background_axis_true_y.SetLabelColor(ROOT.kViolet+2)
background_axis_true_y.SetLineColor(ROOT.kViolet+2)
background_axis_true_y.SetTitleOffset(1.25)
background_axis_true_y.Draw()

true_y_efficiency_canvas.Modified()
true_y_efficiency_canvas.Update()
true_y_efficiency_canvas.Print(dir_out + graphFileName + ")", "pdf")
true_y_efficiency_canvas.Close()

# Separate four-page PDF: the 13 BDT input-variable distributions. The two
# 1-D pages compare normalized shapes. The two scatter pages use the BDT score
# on the x-axis and draw background after signal so it remains prominent.
variable_pages = []
variable_pages.append(
    make_variable_histogram_page(
        "bdt_variables_primary_1d",
        "Primary BDT input-variable distributions",
        PRIMARY_BDT_VARIABLES,
        signal_variable_values,
        background_variable_values,
    )
)
variable_pages.append(
    make_variable_histogram_page(
        "bdt_variables_secondary_1d",
        "Remaining BDT input-variable distributions",
        SECONDARY_BDT_VARIABLES,
        signal_variable_values,
        background_variable_values,
    )
)
variable_pages.append(
    make_variable_scatter_page(
        "bdt_variables_primary_scatter",
        "Primary BDT variables versus BDT score",
        PRIMARY_BDT_VARIABLES,
        signal_scores_all,
        background_scores_all,
        signal_variable_values,
        background_variable_values,
        xMin,
        xMax,
        cut_selected,
    )
)
variable_pages.append(
    make_variable_scatter_page(
        "bdt_variables_secondary_scatter",
        "Remaining BDT variables versus BDT score",
        SECONDARY_BDT_VARIABLES,
        signal_scores_all,
        background_scores_all,
        signal_variable_values,
        background_variable_values,
        xMin,
        xMax,
        cut_selected,
    )
)

for page_index, (variable_canvas, variable_objects) in enumerate(
    variable_pages
):
    open_pdf = page_index == 0
    close_pdf = page_index == len(variable_pages) - 1
    pdf_delimiter = "(" if open_pdf else ")" if close_pdf else ""
    output_name = dir_out + variableGraphFileName + pdf_delimiter
    variable_canvas.Print(output_name, "pdf")
    variable_canvas.Close()

output.cd()
graph_roc_original.Write("graph_roc_original")
graph_roc_weighted.Write("graph_roc_weighted_f03")
graph_roc_true_y.Write("graph_roc_true_y_support_only_f03")
graph_roc_weighted_band.Write("graph_roc_weighted_f_band")
graph_roc_weighted_band_polygon.Write(
    "graph_roc_weighted_f_band_polygon"
)
graph_efficiency_original.Write("graph_efficiency_original")
graph_efficiency_weighted.Write("graph_efficiency_weighted_f03")
graph_efficiency_band.Write("graph_efficiency_f_band")
graph_efficiency_true_y.Write("graph_efficiency_true_y_f03")
graph_efficiency_true_y_band.Write("graph_efficiency_true_y_f_band")
graph_background_passing_original.Write(
    "graph_background_passing_original_panel"
)
graph_background_confidence_original.Write(
    "graph_background_68pct_binomial_ci_original_panel"
)
graph_background_passing_weighted.Write(
    "graph_background_passing_weighted_panel"
)
graph_background_confidence_weighted.Write(
    "graph_background_68pct_binomial_ci_weighted_panel"
)
graph_background_passing_true_y.Write(
    "graph_background_passing_true_y_panel"
)
graph_background_confidence_true_y.Write(
    "graph_background_68pct_binomial_ci_true_y_panel"
)
graph_distribution_band.Write("graph_distribution_f_band")
testTree_S.Write()
hist_S.Write()
hist_S_weighted_f01.Write()
hist_S_weighted_f03.Write()
hist_S_weighted_f05.Write()
hist_S_weighted_f01_counts.Write()
hist_S_weighted_f03_counts.Write()
hist_S_weighted_f05_counts.Write()
hist_S_true_y_f01.Write()
hist_S_true_y_f03.Write()
hist_S_true_y_f05.Write()
hist_S_true_y_f01_counts.Write()
hist_S_true_y_f03_counts.Write()
hist_S_true_y_f05_counts.Write()
testTree_B.Write()
hist_B.Write()

output.Close()
input_sig.Close()
input_bkg.Close()
del reader

def expected_signal(
    triggered_rate_per_station_year,
    livetime_days,
    weighted_bdt_efficiency,
):
    return (
        triggered_rate_per_station_year
        * livetime_days / 365.25
        * weighted_bdt_efficiency
    )


n_expected = expected_signal(
    triggered_rate_per_station_year=20.0,
    livetime_days=68.4,
    weighted_bdt_efficiency=eff_selected_true_y_support_f03,
)

print("Expected selected signal:", n_expected)

print("==> BDT testing is done!")
