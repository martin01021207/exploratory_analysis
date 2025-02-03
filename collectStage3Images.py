import argparse
import numpy as np
from array import array
import json

import ROOT
from ROOT import TFile, TTree

# image size NxN
N = 48
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

    image_vector_bkg = ROOT.std.vector["float"](ntot)
    station_number_bkg = array('i', [0])
    run_number_bkg = array('i', [0])
    event_number_bkg = array('i', [0])
    sim_energy_bkg = array('f', [0.])

    tree_bkg_in = file_in.Get("images_bkg")
    tree_bkg_in.SetBranchAddress("image", ROOT.AddressOf(image_vector_bkg))
    nEvents_bkg = tree_bkg_in.GetEntries()

    image_vector_sig = ROOT.std.vector["float"](ntot)
    station_number_sig = array('i', [0])
    run_number_sig = array('i', [0])
    event_number_sig = array('i', [0])
    sim_energy_sig = array('f', [0.])

    tree_sig_in = file_in.Get("images_sig")
    tree_sig_in.SetBranchAddress("image", ROOT.AddressOf(image_vector_sig))
    nEvents_sig = tree_sig_in.GetEntries()

    filename_out = f"images_s{station}_test_stage3.root"

    tree_bkg_out = ROOT.TTree("images_bkg", "images_bkg")
    tree_sig_out = ROOT.TTree("images_sig", "images_sig")

    file_out = TFile(dir_out+filename_out, "RECREATE")

    tree_bkg_out.Branch("image", "std::vector<float>", image_vector_bkg)
    tree_bkg_out.Branch("station_number", station_number_bkg, 'station_number/I')
    tree_bkg_out.Branch("run_number", run_number_bkg, 'run_number/I')
    tree_bkg_out.Branch("event_number", event_number_bkg, 'event_number/I')
    tree_bkg_out.Branch("sim_energy", sim_energy_bkg, 'sim_energy/F')
    tree_bkg_out.SetDirectory(file_out)

    tree_sig_out.Branch("image", "std::vector<float>", image_vector_sig)
    tree_sig_out.Branch("station_number", station_number_sig, 'station_number/I')
    tree_sig_out.Branch("run_number", run_number_sig, 'run_number/I')
    tree_sig_out.Branch("event_number", event_number_sig, 'event_number/I')
    tree_sig_out.Branch("sim_energy", sim_energy_sig, 'sim_energy/F')
    tree_sig_out.SetDirectory(file_out)

    for i_event in range(nEvents_bkg):
        tree_bkg_in.GetEntry(i_event)

        run = str(int(tree_bkg_in.run_number))
        event = tree_bkg_in.event_number

        if run in FP:
            eventList = FP[run]
            if event in eventList:
                station_number_bkg[0] = tree_bkg_in.station_number
                run_number_bkg[0] = tree_bkg_in.run_number
                event_number_bkg[0] = tree_bkg_in.event_number
                sim_energy_bkg[0] = tree_bkg_in.sim_energy
                tree_bkg_out.Fill()

    for i_event in range(nEvents_sig):
        tree_sig_in.GetEntry(i_event)
        station_number_sig[0] = tree_sig_in.station_number
        run_number_sig[0] = tree_sig_in.run_number
        event_number_sig[0] = tree_sig_in.event_number
        sim_energy_sig[0] = tree_sig_in.sim_energy
        tree_sig_out.Fill()

    tree_bkg_out.Write()
    tree_sig_out.Write()

    print("Image data written to the file ", file_out.GetName())
    tree_bkg_out.Print()
    tree_sig_out.Print()

    file_in.Close()
    file_out.Close()
