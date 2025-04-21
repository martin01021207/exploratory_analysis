import argparse
import numpy as np
import ROOT
from ROOT import TFile, TTree, TCanvas, TPad, TH1F, TGraph, TLine, TLegend
from array import array
import json

parser = argparse.ArgumentParser(description='test_BDT')
parser.add_argument('file_in', type=str, help="Path to the input file")
parser.add_argument('station', type=str, help="Station number")
parser.add_argument('dir_trained', type=str, help="Path to the directory of trained BDT weights")
parser.add_argument('dir_out', type=str, help="Output directory")
parser.add_argument('--target_eff', type=float, default=0.9999, help="Target signal efficiency")
args = parser.parse_args()

file_in = args.file_in
station = args.station
dir_trained = args.dir_trained
if not dir_trained.endswith("/"):
    dir_trained += "/"
dir_out = args.dir_out
if not dir_out.endswith("/"):
    dir_out += "/"

station_str = f"s{station}"

# Target signal efficiency
targetEff = args.target_eff

# Method
method = "BDTD"

jsonFileName = "falsePositiveEvents_vars_" + station_str + ".json"
targetFileName = "testTree_vars_" + station_str + ".root"
graphFileName = "testedResults_vars_" + station_str + ".pdf"

TMVA = ROOT.TMVA

TMVA.Tools.Instance()
#TMVA.PyMethodBase.PyInitialize()

print("==> Start BDT testing")

nCoincidentPairs_PA_float = array("f", [0.])
nHighHits_PA_float = array("f", [0.])
nCoincidentPairs_inIce_float = array("f", [0.])
nHighHits_inIce_float = array("f", [0.])
station_number_float = array("f", [0.])
run_number_float = array("f", [0.])
event_number_float = array("f", [0.])

nCoincidentPairs_PA = array("i", [0])
nHighHits_PA = array("i", [0])
averageSNR_PA = array("f", [0.])
averageKurtosis_PA = array("f", [0.])
averageEntropy_PA = array("f", [0.])
impulsivity_PA = array("f", [0.])
coherentSNR_PA = array("f", [0.])
coherentKurtosis_PA = array("f", [0.])
coherentEntropy_PA = array("f", [0.])
nCoincidentPairs_inIce = array("i", [0])
nHighHits_inIce = array("i", [0])
averageSNR_inIce = array("f", [0.])
averageKurtosis_inIce = array("f", [0.])
averageEntropy_inIce = array("f", [0.])
impulsivity_inIce = array("f", [0.])
coherentSNR_inIce = array("f", [0.])
coherentKurtosis_inIce = array("f", [0.])
coherentEntropy_inIce = array("f", [0.])
station_number = array("i", [0])
run_number = array("i", [0])
event_number = array("i", [0])
sim_energy = array("f", [0.])
#trigger_time_difference = array('f', [0.])

reader = TMVA.Reader( "!Color:!Silent" )
reader.AddVariable( "nCoincidentPairs_PA", nCoincidentPairs_PA_float )
reader.AddVariable( "nHighHits_PA", nHighHits_PA_float )
reader.AddVariable( "averageSNR_PA", averageSNR_PA )
reader.AddVariable( "averageKurtosis_PA", averageKurtosis_PA )
reader.AddVariable( "averageEntropy_PA", averageEntropy_PA )
reader.AddVariable( "impulsivity_PA", impulsivity_PA )
reader.AddVariable( "coherentSNR_PA", coherentSNR_PA )
reader.AddVariable( "coherentKurtosis_PA", coherentKurtosis_PA )
reader.AddVariable( "coherentEntropy_PA", coherentEntropy_PA )
reader.AddVariable( "nCoincidentPairs_inIce", nCoincidentPairs_inIce_float )
reader.AddVariable( "nHighHits_inIce", nHighHits_inIce_float )
reader.AddVariable( "averageSNR_inIce", averageSNR_inIce )
reader.AddVariable( "averageKurtosis_inIce", averageKurtosis_inIce )
reader.AddVariable( "averageEntropy_inIce", averageEntropy_inIce )
reader.AddVariable( "impulsivity_inIce", impulsivity_inIce )
reader.AddVariable( "coherentSNR_inIce", coherentSNR_inIce )
reader.AddVariable( "coherentKurtosis_inIce", coherentKurtosis_inIce )
reader.AddVariable( "coherentEntropy_inIce", coherentEntropy_inIce )
reader.AddSpectator( "station_number", station_number_float )
reader.AddSpectator( "run_number", run_number_float )
reader.AddSpectator( "event_number", event_number_float )
reader.AddSpectator( "sim_energy", sim_energy )
#reader.AddSpectator( "trigger_time_difference", trigger_time_difference )

prefix = "TMVA_Classification"
methodName = f"{method} method"
weightfile = dir_trained + prefix + "_" + method + ".weights.xml"
reader.BookMVA( methodName, weightfile )

input = TFile.Open(file_in)
print(f"--- TMVA Classification App    : Using input file: {input.GetName()}")

tree_S = input.Get(f"vars_sig")
tree_S.SetBranchAddress( "nCoincidentPairs_PA", nCoincidentPairs_PA )
tree_S.SetBranchAddress( "nHighHits_PA", nHighHits_PA )
tree_S.SetBranchAddress( "averageSNR_PA", averageSNR_PA )
tree_S.SetBranchAddress( "averageKurtosis_PA", averageKurtosis_PA )
tree_S.SetBranchAddress( "averageEntropy_PA", averageEntropy_PA )
tree_S.SetBranchAddress( "impulsivity_PA", impulsivity_PA )
tree_S.SetBranchAddress( "coherentSNR_PA", coherentSNR_PA )
tree_S.SetBranchAddress( "coherentKurtosis_PA", coherentKurtosis_PA )
tree_S.SetBranchAddress( "coherentEntropy_PA", coherentEntropy_PA )
tree_S.SetBranchAddress( "nCoincidentPairs_inIce", nCoincidentPairs_inIce )
tree_S.SetBranchAddress( "nHighHits_inIce", nHighHits_inIce )
tree_S.SetBranchAddress( "averageSNR_inIce", averageSNR_inIce )
tree_S.SetBranchAddress( "averageKurtosis_inIce", averageKurtosis_inIce )
tree_S.SetBranchAddress( "averageEntropy_inIce", averageEntropy_inIce )
tree_S.SetBranchAddress( "impulsivity_inIce", impulsivity_inIce )
tree_S.SetBranchAddress( "coherentSNR_inIce", coherentSNR_inIce )
tree_S.SetBranchAddress( "coherentKurtosis_inIce", coherentKurtosis_inIce )
tree_S.SetBranchAddress( "coherentEntropy_inIce", coherentEntropy_inIce )
tree_S.SetBranchAddress( "station_number", station_number )
tree_S.SetBranchAddress( "run_number", run_number )
tree_S.SetBranchAddress( "event_number", event_number )
tree_S.SetBranchAddress( "sim_energy", sim_energy )
#tree_S.SetBranchAddress( "trigger_time_difference", trigger_time_difference )
nEvents_S = tree_S.GetEntries()
print(f"--- SIGNAL: {nEvents_S} events")

tree_B = input.Get(f"vars_bkg")
tree_B.SetBranchAddress( "nCoincidentPairs_PA", nCoincidentPairs_PA )
tree_B.SetBranchAddress( "nHighHits_PA", nHighHits_PA )
tree_B.SetBranchAddress( "averageSNR_PA", averageSNR_PA )
tree_B.SetBranchAddress( "averageKurtosis_PA", averageKurtosis_PA )
tree_B.SetBranchAddress( "averageEntropy_PA", averageEntropy_PA )
tree_B.SetBranchAddress( "impulsivity_PA", impulsivity_PA )
tree_B.SetBranchAddress( "coherentSNR_PA", coherentSNR_PA )
tree_B.SetBranchAddress( "coherentKurtosis_PA", coherentKurtosis_PA )
tree_B.SetBranchAddress( "coherentEntropy_PA", coherentEntropy_PA )
tree_B.SetBranchAddress( "nCoincidentPairs_inIce", nCoincidentPairs_inIce )
tree_B.SetBranchAddress( "nHighHits_inIce", nHighHits_inIce )
tree_B.SetBranchAddress( "averageSNR_inIce", averageSNR_inIce )
tree_B.SetBranchAddress( "averageKurtosis_inIce", averageKurtosis_inIce )
tree_B.SetBranchAddress( "averageEntropy_inIce", averageEntropy_inIce )
tree_B.SetBranchAddress( "impulsivity_inIce", impulsivity_inIce )
tree_B.SetBranchAddress( "coherentSNR_inIce", coherentSNR_inIce )
tree_B.SetBranchAddress( "coherentKurtosis_inIce", coherentKurtosis_inIce )
tree_B.SetBranchAddress( "coherentEntropy_inIce", coherentEntropy_inIce )
tree_B.SetBranchAddress( "station_number", station_number )
tree_B.SetBranchAddress( "run_number", run_number )
tree_B.SetBranchAddress( "event_number", event_number )
tree_B.SetBranchAddress( "sim_energy", sim_energy )
#tree_B.SetBranchAddress( "trigger_time_difference", trigger_time_difference )
nEvents_B = tree_B.GetEntries()
print(f"--- BACKGROUND: {nEvents_B} events")

output = TFile( dir_out+targetFileName, "RECREATE" )
output.cd()

nbin = 100
if method == "BDTD":
    xMin = -0.8
    xMax = 0.8
else:
    xMin = -0.1
    xMax = 1.1

histTitle = f"TMVA response for classifier: {method} (S{station})"
hist_S = TH1F("hist_S", histTitle, nbin, xMin, xMax)
hist_S.SetLineColor(ROOT.kAzure+2)
hist_S.SetLineWidth(3)
hist_S.SetFillColorAlpha(ROOT.kAzure-7, 0.7)
hist_B = TH1F("hist_B", "", nbin, xMin, xMax)
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
#testTree_S.Branch( "trigger_time_difference", trigger_time_difference, "trigger_time_difference/F" )

testTree_B = TTree("TestTree_B", "TestTree_B")
testTree_B.SetDirectory(output)
testTree_B.Branch( method, EvaluateMVA, method+"/F" )
testTree_B.Branch( "station_number", station_number, "station_number/I" )
testTree_B.Branch( "run_number", run_number, "run_number/I" )
testTree_B.Branch( "event_number", event_number, "event_number/I" )
testTree_B.Branch( "sim_energy", sim_energy, "sim_energy/F" )
#testTree_B.Branch( "trigger_time_difference", trigger_time_difference, "trigger_time_difference/F" )

for i_event in range(nEvents_S):
    tree_S.GetEntry(i_event)
    nCoincidentPairs_PA_float[0] = nCoincidentPairs_PA[0]
    nHighHits_PA_float[0] = nHighHits_PA[0]
    nCoincidentPairs_inIce_float[0] = nCoincidentPairs_inIce[0]
    nHighHits_inIce_float[0] = nHighHits_inIce[0]
    station_number_float[0] = station_number[0]
    run_number_float[0] = run_number[0]
    event_number_float[0] = event_number[0]
    EvaluateMVA[0] = reader.EvaluateMVA(methodName)
    testTree_S.Fill()
    hist_S.Fill(EvaluateMVA[0])
print("--- End of event loop (SIGNAL)")

for i_event in range(nEvents_B):
    tree_B.GetEntry(i_event)
    nCoincidentPairs_PA_float[0] = nCoincidentPairs_PA[0]
    nHighHits_PA_float[0] = nHighHits_PA[0]
    nCoincidentPairs_inIce_float[0] = nCoincidentPairs_inIce[0]
    nHighHits_inIce_float[0] = nHighHits_inIce[0]
    station_number_float[0] = station_number[0]
    run_number_float[0] = run_number[0]
    event_number_float[0] = event_number[0]
    EvaluateMVA[0] = reader.EvaluateMVA(methodName)
    testTree_B.Fill()
    hist_B.Fill(EvaluateMVA[0])
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

    effDiff = abs(eff - targetEff)
    if effDiff < minDiff:
        minDiff = effDiff
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
hist_S.Draw()
hist_B.Draw("same")

if nEvents_S > nEvents_B:
    leg_xMin = 0.1
    leg_xMax = 0.3
else:
    leg_xMin = 0.7
    leg_xMax = 0.9

leg_yMin = 0.75
leg_yMax = 0.9

leg_hist = TLegend(leg_xMin, leg_yMin, leg_xMax, leg_yMax)
leg_hist.AddEntry(hist_S, "Signal", "f")
leg_hist.AddEntry(hist_B, "Background", "f")
leg_hist.Draw()

canvas.Print(dir_out+graphFileName+"(", "pdf")
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

gr_xMin = 0.9
gr_xMax = 1.0015
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
input.Close()
del reader

print("==> BDT testing is done!")
