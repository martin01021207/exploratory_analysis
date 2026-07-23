import argparse
import numpy as np
import ROOT
from ROOT import TFile, TTree, TCanvas, TPad, TH1F, TGraph, TLine, TLegend
from array import array
import json

parser = argparse.ArgumentParser(description='test BDT')
parser.add_argument('station', type=str, help="Station number")
parser.add_argument('file_in', type=str, help="Path to the input file")
parser.add_argument('sim_file_in', type=str, help="Path to the input simulation file")
parser.add_argument('dir_trained', type=str, help="Path to the directory of trained BDT weights")
parser.add_argument('dir_out', type=str, help="Output directory")
parser.add_argument('--target_cut', type=float, default=0.1, help="Target BDT cut")
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

# Method
method = "BDTD"

jsonFileName = "falsePositiveEvents_vars_" + station_str + f"_{method}.json"
targetFileName = "testTree_vars_" + station_str + f"_{method}.root"
graphFileName = "testedResults_vars_" + station_str + f"_{method}.pdf"

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


reader = TMVA.Reader( "!Color:!Silent" )
reader.AddVariable( "passed_hit_filter", passed_hit_filter_float )
reader.AddVariable( "nCoincidentPairs_PA", nCoincidentPairs_PA_float )
reader.AddVariable( "nHighHits_PA", nHighHits_PA_float )
reader.AddVariable( "nCoincidentPairs_inIce", nCoincidentPairs_inIce_float )
reader.AddVariable( "nHighHits_inIce", nHighHits_inIce_float )

reader.AddVariable( "reco_max_corr", reco_max_corr )
reader.AddVariable( "reco_surf_corr_z", reco_surf_corr_z )
reader.AddVariable( "reco_surf_corr_zen", reco_surf_corr_zen )

#reader.AddVariable( "averageSNR_PA", averageSNR_PA )
#reader.AddVariable( "averageKurtosis_PA", averageKurtosis_PA )
#reader.AddVariable( "averageEntropy_PA", averageEntropy_PA )
#reader.AddVariable( "averageImpulsivity_PA", averageImpulsivity_PA )
#reader.AddVariable( "coherentSNR_PA", coherentSNR_PA )
#reader.AddVariable( "coherentKurtosis_PA", coherentKurtosis_PA )
#reader.AddVariable( "coherentEntropy_PA", coherentEntropy_PA )
#reader.AddVariable( "coherentImpulsivity_PA", coherentImpulsivity_PA )

reader.AddVariable( "averageSNR_inIce", averageSNR_inIce )
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
tree_S.SetBranchAddress( "nCoincidentPairs_PA", nCoincidentPairs_PA )
tree_S.SetBranchAddress( "nHighHits_PA", nHighHits_PA )
tree_S.SetBranchAddress( "nCoincidentPairs_inIce", nCoincidentPairs_inIce )
tree_S.SetBranchAddress( "nHighHits_inIce", nHighHits_inIce )

#tree_S.SetBranchAddress( "averageSNR_PA", averageSNR_PA )
#tree_S.SetBranchAddress( "averageKurtosis_PA", averageKurtosis_PA )
#tree_S.SetBranchAddress( "averageEntropy_PA", averageEntropy_PA )
#tree_S.SetBranchAddress( "averageImpulsivity_PA", averageImpulsivity_PA )
#tree_S.SetBranchAddress( "coherentSNR_PA", coherentSNR_PA )
#tree_S.SetBranchAddress( "coherentKurtosis_PA", coherentKurtosis_PA )
#tree_S.SetBranchAddress( "coherentEntropy_PA", coherentEntropy_PA )
#tree_S.SetBranchAddress( "coherentImpulsivity_PA", coherentImpulsivity_PA )

tree_S.SetBranchAddress( "averageSNR_inIce", averageSNR_inIce )
tree_S.SetBranchAddress( "averageKurtosis_inIce", averageKurtosis_inIce )
tree_S.SetBranchAddress( "averageEntropy_inIce", averageEntropy_inIce )
tree_S.SetBranchAddress( "averageImpulsivity_inIce", averageImpulsivity_inIce )
tree_S.SetBranchAddress( "coherentSNR_inIce", coherentSNR_inIce )
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
tree_B.SetBranchAddress( "nCoincidentPairs_PA", nCoincidentPairs_PA )
tree_B.SetBranchAddress( "nHighHits_PA", nHighHits_PA )
tree_B.SetBranchAddress( "nCoincidentPairs_inIce", nCoincidentPairs_inIce )
tree_B.SetBranchAddress( "nHighHits_inIce", nHighHits_inIce )

#tree_B.SetBranchAddress( "averageSNR_PA", averageSNR_PA )
#tree_B.SetBranchAddress( "averageKurtosis_PA", averageKurtosis_PA )
#tree_B.SetBranchAddress( "averageEntropy_PA", averageEntropy_PA )
#tree_B.SetBranchAddress( "averageImpulsivity_PA", averageImpulsivity_PA )
#tree_B.SetBranchAddress( "coherentSNR_PA", coherentSNR_PA )
#tree_B.SetBranchAddress( "coherentKurtosis_PA", coherentKurtosis_PA )
#tree_B.SetBranchAddress( "coherentEntropy_PA", coherentEntropy_PA )
#tree_B.SetBranchAddress( "coherentImpulsivity_PA", coherentImpulsivity_PA )

tree_B.SetBranchAddress( "averageSNR_inIce", averageSNR_inIce )
tree_B.SetBranchAddress( "averageKurtosis_inIce", averageKurtosis_inIce )
tree_B.SetBranchAddress( "averageEntropy_inIce", averageEntropy_inIce )
tree_B.SetBranchAddress( "averageImpulsivity_inIce", averageImpulsivity_inIce )
tree_B.SetBranchAddress( "coherentSNR_inIce", coherentSNR_inIce )
tree_B.SetBranchAddress( "coherentKurtosis_inIce", coherentKurtosis_inIce )
tree_B.SetBranchAddress( "coherentEntropy_inIce", coherentEntropy_inIce )
tree_B.SetBranchAddress( "coherentImpulsivity_inIce", coherentImpulsivity_inIce )

nEvents_B = tree_B.GetEntries()

output = TFile( dir_out+targetFileName, "RECREATE" )
output.cd()

nbin = 100
if method == "BDTD":
    xMin = -0.4
    xMax = 1.0
else:
    xMin = -0.1
    xMax = 1.1

histTitle = f"TMVA response for classifier: {method} (S{station})"
hist_S = TH1F("hist_S", "", nbin, xMin, xMax)
hist_S.SetLineColorAlpha(ROOT.kAzure+2, 0.5)
hist_S.SetLineWidth(3)
hist_S.SetFillColorAlpha(ROOT.kAzure-7, 0.2)
hist_B = TH1F("hist_B", histTitle, nbin, xMin, xMax)
hist_B.SetLineColor(ROOT.kRed+1)
hist_B.SetLineWidth(3)
hist_B.SetFillColor(ROOT.kRed+1)
hist_B.SetFillStyle(3354)

EvaluateMVA = array("f", [0.])

testTree_S = TTree("TestTree_S", "TestTree_S")
testTree_S.SetDirectory(output)

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

    trigger_time[0] = trigger_time_float[0]

    true_source_theta_float[0] = true_source_theta[0]
    true_source_phi_float[0] = true_source_phi[0]

    EvaluateMVA[0] = reader.EvaluateMVA(methodName)

    testTree_S.Fill()
    hist_S.Fill(EvaluateMVA[0])
print(f"--- SIGNAL: {testTree_S.GetEntries()} events")
print("--- End of event loop (SIGNAL)")

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

    trigger_time[0] = trigger_time_float[0]

    true_source_theta_float[0] = true_source_theta[0]
    true_source_phi_float[0] = true_source_phi[0]

    EvaluateMVA[0] = reader.EvaluateMVA(methodName)

    testTree_B.Fill()
    hist_B.Fill(EvaluateMVA[0])
print(f"--- BACKGROUND: {testTree_B.GetEntries()} events")
print("--- End of event loop (BACKGROUND)")

graph = TGraph()

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

minDiff = 1.0
for cut in cutValues:
    threshold = f"{method} > {cut}"

    count_S = testTree_S.GetEntries(threshold)
    count_B = testTree_B.GetEntries(threshold)

    eff = count_S / nCounts_S
    rej = 1 - count_B / nCounts_B

    graph.SetPoint(graph.GetN(), eff, rej)

    #print(f"eff: {eff}    rej: {rej}")

    if abs(cut - targetCut) < 1E-5:
        targetEff = eff
        cut_selected = cut
        eff_selected = eff
        rej_selected = rej

print(f"*** Signal Efficiency: {eff_selected}")
print(f"*** Background Rejection: {rej_selected}")
nEvents_FP = 0
i = 0
info_FP = {}
bkgRuns = []
bkgEvents = []
threshold = f"{method} > {cut_selected}"
count_B_selected = testTree_B.GetEntries(threshold)
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

canvas = TCanvas("c1", histTitle, 10, 10, 850, 500)
ROOT.gStyle.SetOptStat(0)
ROOT.gPad.SetLogy(1)

canvas.cd()
hist_B.Draw()
hist_S.Draw("same")

leg_xMin = 0.4
leg_xMax = 0.6

leg_yMin = 0.75
leg_yMax = 0.9

leg_hist = TLegend(leg_xMin, leg_yMin, leg_xMax, leg_yMax)
leg_hist.AddEntry(hist_S, "Signal", "f")
leg_hist.AddEntry(hist_B, "Background", "f")
leg_hist.Draw()

canvas.Print(dir_out+graphFileName+"(", "pdf")
canvas.Clear("D")

hist_B.Draw()
hist_S.Draw("same")

cut_selected = 0.1
yMax_cut= hist_B.GetMaximum()*1.5
vLine_cut = TLine(cut_selected, 0, cut_selected, yMax_cut)
vLine_cut.SetLineStyle(2)
vLine_cut.SetLineWidth(2)
vLine_cut.Draw("same")

leg_hist.AddEntry(hist_S, "Signal", "f")
leg_hist.AddEntry(hist_B, "Background", "f")
leg_hist.Draw()

canvas.Print(dir_out+graphFileName, "pdf")
canvas.Clear("D")

grid = TPad("grid", "", 0, 0, 1, 1)
grid.Draw()
grid.cd()
grid.SetGrid()

ROOT.gPad.SetLeftMargin(0.15)
graphTitle = f"Signal efficiency vs. Background rejection (S{station})"
graph.SetTitle(graphTitle)
graph.GetXaxis().SetTitle("Signal efficiency (Sensitivity)")
graph.GetYaxis().SetTitle("Background rejection (Specificity)")
graph.SetLineWidth(2)
graph.SetLineColor(4)
graph.GetXaxis().SetRangeUser(0, 1.01)
graph.GetYaxis().SetRangeUser(0, 1.01)
graph.Draw("AL")

leg = TLegend(0.2, 0.15, 0.35, 0.3)
leg.SetHeader("MVA Method", "")
leg.AddEntry(graph, method, "l")
leg.Draw()

canvas.Print(dir_out+graphFileName, "pdf")
canvas.Clear("D")

gr_xMin = 0.75
gr_xMax = 0.95
gr_yMin = 0.7
gr_yMax = 1.001

graph.GetXaxis().SetRangeUser(gr_xMin, gr_xMax)
graph.GetYaxis().SetRangeUser(gr_yMin, gr_yMax)
graph.Draw("AL")

leg.Draw()

hLine = TLine(gr_xMin, rej_selected, targetEff, rej_selected)
hLine.SetLineStyle(2)
hLine.SetLineWidth(2)
hLine.SetLineColor(6)
hLine.Draw("same")

vLine = TLine(targetEff, gr_yMin, targetEff, rej_selected)
vLine.SetLineStyle(2)
vLine.SetLineWidth(2)
vLine.SetLineColor(6)
vLine.Draw("same")

canvas.Print(dir_out+graphFileName+")", "pdf")
canvas.Clear("D")

output.cd()
graph.Write()
testTree_S.Write()
hist_S.Write()
testTree_B.Write()
hist_B.Write()

output.Close()
input_sig.Close()
input_bkg.Close()
del reader

print("==> BDT testing is done!")
