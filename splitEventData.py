import os
import argparse
import numpy as np
from array import array

import ROOT
from ROOT import TFile, TTree

nChannels = 24


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_file_in", type=str, help="Path to the input file")
    parser.add_argument("dir_out", type=str, help="Output directory")
    args = parser.parse_args()

    path_to_file_in = args.path_to_file_in
    dir_out = args.dir_out
    if not dir_out.endswith("/"):
        dir_out += "/"

    os.mkdir(dir_out+"train")
    os.mkdir(dir_out+"test")

    filename_in = path_to_file_in.split("/")[-1]
    filename_out = filename_in.split(".root")[0]

    if "sim" in filename_in:
        treename = "events_sim"
    else:
        treename = "events"

    graph_vector = ROOT.std.vector["TGraph"](nChannels)

    station_number_train = array('i', [0])
    run_number_train = array('i', [0])
    event_number_train = array('i', [0])
    sim_energy_train = array('f', [0.])

    station_number_test = array('i', [0])
    run_number_test = array('i', [0])
    event_number_test = array('i', [0])
    sim_energy_test = array('f', [0.])

    file_in = TFile.Open(path_to_file_in)
    tree_in = file_in.Get(treename)
    tree_in.SetBranchAddress("waveform_graphs", ROOT.AddressOf(graph_vector))
    tree_in.GetEntry(0)
    station = tree_in.station_number
    print(f"Station {station}")
    nEvents = tree_in.GetEntries()
    print(f"Number of total events: {nEvents}")

    tree_out_train = TTree(treename, treename)
    file_out_train = TFile(dir_out+"train/"+filename_out+"_train.root", "recreate")
    tree_out_train.Branch("waveform_graphs", "std::vector<TGraph>", graph_vector)
    tree_out_train.Branch("station_number", station_number_train, 'station_number/I')
    tree_out_train.Branch("run_number", run_number_train, 'run_number/I')
    tree_out_train.Branch("event_number", event_number_train, 'event_number/I')
    tree_out_train.Branch("sim_energy", sim_energy_train, 'sim_energy/F')
    tree_out_train.SetDirectory(file_out_train)

    tree_out_test = TTree(treename, treename)
    file_out_test = TFile(dir_out+"test/"+filename_out+"_test.root", "recreate")
    tree_out_test.Branch("waveform_graphs", "std::vector<TGraph>", graph_vector)
    tree_out_test.Branch("station_number", station_number_test, 'station_number/I')
    tree_out_test.Branch("run_number", run_number_test, 'run_number/I')
    tree_out_test.Branch("event_number", event_number_test, 'event_number/I')
    tree_out_test.Branch("sim_energy", sim_energy_test, 'sim_energy/F')
    tree_out_test.SetDirectory(file_out_test)

    for i_event in range(nEvents):
        tree_in.GetEntry(i_event)

        if i_event % 2 == 0:
            station_number_train[0] = tree_in.station_number
            run_number_train[0] = tree_in.run_number
            event_number_train[0] = tree_in.event_number
            sim_energy_train[0] = tree_in.sim_energy
            tree_out_train.Fill()
        else:
            station_number_test[0] = tree_in.station_number
            run_number_test[0] = tree_in.run_number
            event_number_test[0] = tree_in.event_number
            sim_energy_test[0] = tree_in.sim_energy
            tree_out_test.Fill()

    print(f"Number of events for training: {tree_out_train.GetEntries()}")
    print(f"Number of events for testing: {tree_out_test.GetEntries()}")

    file_out_train.cd()
    tree_out_train.Write()
    file_out_train.Close()

    file_out_test.cd()
    tree_out_test.Write()
    file_out_test.Close()

    file_in.Close()
