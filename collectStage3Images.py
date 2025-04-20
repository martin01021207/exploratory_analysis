import argparse
import numpy as np
from array import array
import json

import ROOT
from ROOT import TFile, TTree

# image size NxN
N = 32
ntot = N * N


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_in", type=str, help="Input directory")
    parser.add_argument("station", type=str, help="Station number")
    parser.add_argument('json_FP', type=str, help="JSON file of false positive events")
    parser.add_argument("dir_out", type=str, help="Output directory")
    args = parser.parse_args()

    dir_in = args.dir_in
    if not dir_in.endswith("/"):
        dir_in += "/"
    station = args.station

    json_FP = args.json_FP
    with open(json_FP,'r') as json_falsePositiveEvents:
        FP = json.loads(json_falsePositiveEvents.read())

    dir_out = args.dir_out
    if not dir_out.endswith("/"):
        dir_out += "/"

    filename_in = f"images_s{station}_test.root"
    file_in = TFile.Open(dir_in+filename_in)

    image_vector = ROOT.std.vector["float"](ntot)
    station_number = array('i', [0])
    run_number = array('i', [0])
    event_number = array('i', [0])
    sim_energy = array('f', [0.])
    trigger_time_difference = array('f', [0.])

    tree_bkg_in = file_in.Get("images_bkg")
    tree_bkg_in.SetBranchAddress("image", ROOT.AddressOf(image_vector))
    nEvents_bkg = tree_bkg_in.GetEntries()

    tree_sig_in = file_in.Get("images_sig")
    tree_sig_in.SetBranchAddress("image", ROOT.AddressOf(image_vector))
    nEvents_sig = tree_sig_in.GetEntries()

    filename_out = f"images_s{station}_test_stage3.root"

    tree_bkg_out = ROOT.TTree("images_bkg", "images_bkg")
    tree_sig_out = ROOT.TTree("images_sig", "images_sig")

    file_out = TFile(dir_out+filename_out, "RECREATE")

    tree_bkg_out.Branch("image", "std::vector<float>", image_vector)
    tree_bkg_out.Branch("station_number", station_number, 'station_number/I')
    tree_bkg_out.Branch("run_number", run_number, 'run_number/I')
    tree_bkg_out.Branch("event_number", event_number, 'event_number/I')
    tree_bkg_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_bkg_out.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')
    tree_bkg_out.SetDirectory(file_out)

    tree_sig_out.Branch("image", "std::vector<float>", image_vector)
    tree_sig_out.Branch("station_number", station_number, 'station_number/I')
    tree_sig_out.Branch("run_number", run_number, 'run_number/I')
    tree_sig_out.Branch("event_number", event_number, 'event_number/I')
    tree_sig_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_sig_out.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')
    tree_sig_out.SetDirectory(file_out)

    for i_event in range(nEvents_bkg):
        tree_bkg_in.GetEntry(i_event)

        run = str(int(tree_bkg_in.run_number))
        event = tree_bkg_in.event_number

        if run in FP:
            eventList = FP[run]
            if event in eventList:
                station_number[0] = tree_bkg_in.station_number
                run_number[0] = tree_bkg_in.run_number
                event_number[0] = tree_bkg_in.event_number
                sim_energy[0] = tree_bkg_in.sim_energy
                trigger_time_difference[0] = tree_bkg_in.trigger_time_difference
                tree_bkg_out.Fill()

    for i_event in range(nEvents_sig):
        tree_sig_in.GetEntry(i_event)
        station_number[0] = tree_sig_in.station_number
        run_number[0] = tree_sig_in.run_number
        event_number[0] = tree_sig_in.event_number
        sim_energy[0] = tree_sig_in.sim_energy
        trigger_time_difference[0] = tree_bkg_in.trigger_time_difference
        tree_sig_out.Fill()

    tree_bkg_out.Write()
    tree_sig_out.Write()

    print("Image data written to the file ", file_out.GetName())
    tree_bkg_out.Print()
    tree_sig_out.Print()

    file_in.Close()
    file_out.Close()
