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
    parser.add_argument('json_FN', type=str, help="JSON file of false negative events")
    parser.add_argument("dir_out", type=str, help="Output directory")
    parser.add_argument("--clean_mode", action="store_true", help="Remove runs that have more than one event")
    args = parser.parse_args()

    dir_in = args.dir_in
    if not dir_in.endswith("/"):
        dir_in += "/"

    station = args.station

    json_FP = args.json_FP
    with open(json_FP,'r') as json_falsePositiveEvents:
        FP = json.loads(json_falsePositiveEvents.read())

    clean_mode = args.clean_mode
    if clean_mode:
        FP = {k: v for k, v in FP_original.items() if len(v) <= 1}

    json_FN = args.json_FN
    with open(json_FN,'r') as json_falseNegativeEvents:
        FN = json.loads(json_falseNegativeEvents.read())

    dir_out = args.dir_out
    if not dir_out.endswith("/"):
        dir_out += "/"

    filename_bkg_in = f"images_s{station}_bkg_test.root"
    file_bkg_in = TFile.Open(dir_in+filename_bkg_in)
    filename_sig_in = f"images_s{station}_sig_test.root"
    file_sig_in = TFile.Open(dir_in+filename_sig_in)

    image_vector = ROOT.std.vector["float"](ntot)
    station_number = array('i', [0])
    run_number = array('i', [0])
    event_number = array('i', [0])
    sim_energy = array('f', [0.])
    trigger_time_difference = array('f', [0.])
    true_radius = array('f', [0.])
    true_theta = array('f', [0.])
    true_phi = array('f', [0.])
    true_source_theta = array('i', [0])
    true_source_phi = array('i', [0])

    tree_bkg_in = file_bkg_in.Get("images_bkg")
    tree_bkg_in.SetBranchAddress("image", ROOT.AddressOf(image_vector))
    nEvents_bkg = tree_bkg_in.GetEntries()

    filename_bkg_out = f"images_s{station}_3staged_bkg_test.root"

    tree_bkg_out = ROOT.TTree("images_bkg", "images_bkg")

    file_bkg_out = TFile(dir_out+filename_bkg_out, "RECREATE")

    tree_bkg_out.Branch("image", "std::vector<float>", image_vector)
    tree_bkg_out.Branch("station_number", station_number, 'station_number/I')
    tree_bkg_out.Branch("run_number", run_number, 'run_number/I')
    tree_bkg_out.Branch("event_number", event_number, 'event_number/I')
    tree_bkg_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_bkg_out.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')
    tree_bkg_out.Branch("true_radius", true_radius, 'true_radius/F')
    tree_bkg_out.Branch("true_theta", true_theta, 'true_theta/F')
    tree_bkg_out.Branch("true_phi", true_phi, 'true_phi/F')
    tree_bkg_out.Branch("true_source_theta", true_source_theta, 'true_source_theta/I')
    tree_bkg_out.Branch("true_source_phi", true_source_phi, 'true_source_phi/I')
    tree_bkg_out.SetDirectory(file_bkg_out)

    tree_sig_in = file_sig_in.Get("images_sig")
    tree_sig_in.SetBranchAddress("image", ROOT.AddressOf(image_vector))
    nEvents_sig = tree_sig_in.GetEntries()

    filename_sig_out = f"images_s{station}_3staged_sig_test.root"

    tree_sig_out = ROOT.TTree("images_sig", "images_sig")

    file_sig_out = TFile(dir_out+filename_sig_out, "RECREATE")

    tree_sig_out.Branch("image", "std::vector<float>", image_vector)
    tree_sig_out.Branch("station_number", station_number, 'station_number/I')
    tree_sig_out.Branch("run_number", run_number, 'run_number/I')
    tree_sig_out.Branch("event_number", event_number, 'event_number/I')
    tree_sig_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_sig_out.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')
    tree_sig_out.Branch("true_radius", true_radius, 'true_radius/F')
    tree_sig_out.Branch("true_theta", true_theta, 'true_theta/F')
    tree_sig_out.Branch("true_phi", true_phi, 'true_phi/F')
    tree_sig_out.Branch("true_source_theta", true_source_theta, 'true_source_theta/I')
    tree_sig_out.Branch("true_source_phi", true_source_phi, 'true_source_phi/I')
    tree_sig_out.SetDirectory(file_sig_out)

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
                true_radius[0] = tree_bkg_in.true_radius
                true_theta[0] = tree_bkg_in.true_theta
                true_phi[0] = tree_bkg_in.true_phi
                true_source_theta[0] = tree_bkg_in.true_source_theta
                true_source_phi[0] = tree_bkg_in.true_source_phi
                tree_bkg_out.Fill()

    tree_bkg_out.Write()
    print("Image data written to the file ", file_bkg_out.GetName())
    tree_bkg_out.Print()

    for i_event in range(nEvents_sig):
        tree_sig_in.GetEntry(i_event)

        E = str(float(tree_sig_in.run_number))
        run = str(int(tree_sig_in.run_number))
        event = tree_sig_in.event_number

        if E in FN and run in FN[E] and event in FN[E][run]:
            continue
        else:
            station_number[0] = tree_sig_in.station_number
            run_number[0] = tree_sig_in.run_number
            event_number[0] = tree_sig_in.event_number
            sim_energy[0] = tree_sig_in.sim_energy
            trigger_time_difference[0] = tree_sig_in.trigger_time_difference
            true_radius[0] = tree_sig_in.true_radius
            true_theta[0] = tree_sig_in.true_theta
            true_phi[0] = tree_sig_in.true_phi
            true_source_theta[0] = tree_sig_in.true_source_theta
            true_source_phi[0] = tree_sig_in.true_source_phi
            tree_sig_out.Fill()

    tree_sig_out.Write()
    print("Image data written to the file ", file_sig_out.GetName())
    tree_sig_out.Print()

    file_bkg_in.Close()
    file_bkg_out.Close()
    file_sig_in.Close()
    file_sig_out.Close()
