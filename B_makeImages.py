import argparse
import numpy as np
from array import array

import ROOT
from ROOT import TFile, TTree, TCanvas

import MakeImages
from NuRadioReco.utilities import trace_utilities

nChannels = 24

# image size NxN
N = 48
ntot = N * N


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_file_in", type=str, help="Path to the input file")
    parser.add_argument("dir_out", type=str, help="Output directory")
    parser.add_argument('--json_FP', type=str, default=None, help="JSON file of false positive events")
    parser.add_argument("--with_plotting", action="store_true", help="Plot out images in pdf files")
    args = parser.parse_args()

    path_to_file_in = args.path_to_file_in
    dir_out = args.dir_out
    if not dir_out.endswith("/"):
        dir_out += "/"

    json_FP = args.json_FP

    filename_in = path_to_file_in.split("/")[-1]
    filename = filename_in.split("filtered_")[1]
    filename_out = f"images_{filename}"

    if "sim" in filename_in:
        treename_in = "events_sim"
        treename_out = "images_sig"
    else:
        treename_in = "events"
        treename_out = "images_bkg"

    file_in = TFile.Open(path_to_file_in)
    if file_in is None:
        print(f"Can't open file: {path_to_file_in}")
        quit()

    graph_vector = ROOT.std.vector["TGraph"](nChannels)

    tree_in = file_in.Get(treename_in)
    if not tree_in:
        print(f"Tree {treename_in} doesn't exit. Exit now...")
        quit()

    tree_in.SetBranchAddress("waveform_graphs", ROOT.AddressOf(graph_vector))

    nEvents = 0
    nEvents = tree_in.GetEntries()
    print(f"File in: {filename_in}")
    print(f"Number of total events: {nEvents}")
    if not nEvents:
        file_in.Close()
        quit()

    isCollectingFPevents = False
    if json_FP:
        isCollectingFPevents = True
        nEvents_FP = 0
        import json
        tree_in.GetEntry(0)
        run = str(int(tree_in.run_number))
        with open(json_FP, 'r') as json_falsePositiveEvents:
            FP = json.loads(json_falsePositiveEvents.read())
        if run in FP:
            eventList = FP[run]
        else:
            print(f"Run {run} is not in the FP list, exit now.")
            quit()

    if args.with_plotting:
        filename = filename.split(".root")[0]
        graph_file_out = f"images_{filename}.pdf"
        file_type = "pdf"
        ROOT.gROOT.SetBatch(ROOT.kTRUE)
        ROOT.gStyle.SetOptStat(0)
        canvas = TCanvas("c1", "", 1200, 600)
        canvas.Divide(2,1)

    tree_out = ROOT.TTree(treename_out, treename_out)

    file_out = TFile(dir_out+filename_out, "RECREATE")

    image_vector = ROOT.std.vector["float"](ntot)
    station_number = array('i', [0])
    run_number = array('i', [0])
    event_number = array('i', [0])
    sim_energy = array('f', [0.])
    trigger_time_difference = array('f', [0.])

    tree_out.Branch("image", "std::vector<float>", image_vector)
    tree_out.Branch("station_number", station_number, 'station_number/I')
    tree_out.Branch("run_number", run_number, 'run_number/I')
    tree_out.Branch("event_number", event_number, 'event_number/I')
    tree_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_out.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')

    tree_out.SetDirectory(file_out)

    hist = ROOT.TH2D("hist", "hist", N, 0, N, N, 0, N)

    for i_event in range(nEvents):
        tree_in.GetEntry(i_event)

        sim_energy[0] = tree_in.sim_energy
        station_number[0] = tree_in.station_number
        run_number[0] = tree_in.run_number
        event_number[0] = tree_in.event_number
        trigger_time_difference[0] = tree_in.trigger_time_difference

        if isCollectingFPevents:
            if event_number[0] not in eventList:
                continue
            else:
                nEvents_FP += 1

        envelopes = []
        RMS = []
        for i_channel in range(nChannels):
            wf = graph_vector[i_channel]
            x = np.array( wf.GetX() )
            y = np.array( wf.GetY() )

            envelopes.append( trace_utilities.get_hilbert_envelope(y) )
            RMS.append( trace_utilities.get_split_trace_noise_RMS(envelopes[i_channel]) )

            del wf

        bins = MakeImages.imageHistogram(hist, envelopes, RMS)
        MakeImages.imageBinsToVector(hist, image_vector)

        tree_out.Fill()

        if args.with_plotting:
            histTitle = f"Station {station_number[0]}, Run {run_number[0]}, Event {event_number[0]}"
            hist.SetTitle(histTitle)
            canvas.cd(1)
            hist.Draw("colz")
            canvas.cd(2)
            hist.Draw("lego2")
            if i_event == 0:
                canvas.Print(dir_out+graph_file_out+"(", file_type)
            elif i_event == nEvents - 1:
                canvas.Print(dir_out+graph_file_out+")", file_type)
            else:
                canvas.Print(dir_out+graph_file_out, file_type)
            canvas.Clear("D")

        hist.Reset()
        del bins

    tree_out.Write()

    if isCollectingFPevents:
        print(f"Number of FP events: {nEvents_FP}")

    #print("Image data written to the file: ", file_out.GetName())
    #tree_out.Print()

    file_in.Close()
    file_out.Close()
