import os
import argparse
import numpy as np
from array import array

import ROOT
from ROOT import TFile, TTree

import MakeVariables
from NuRadioReco.utilities import trace_utilities

nChannels = 24
nChannels_PA = 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_file_in", type=str, help="Path to the input file")
    parser.add_argument("dir_out", type=str, help="Output directory")
    parser.add_argument("--path_to_second_file_in", type=str, default=None, help="Path to the second input file")
    parser.add_argument("--division", type=float, default=None, help="N division")
    parser.add_argument("--count", type=int, default=None, help="N counts")
    args = parser.parse_args()

    path_to_file_in = args.path_to_file_in
    path_to_second_file_in = args.path_to_second_file_in

    dir_out = args.dir_out
    if not dir_out.endswith("/"):
        dir_out += "/"

    division = args.division
    count = args.count
    if division and count:
        print("Need to choose between the division mode or the count mode!")
        quit()
    elif not division and not count:
        print("Need to input an integer of division or an integer of count!")
        quit()

    if not os.path.exists(dir_out+"train"):
        os.mkdir(dir_out+"train")
    if not os.path.exists(dir_out+"test"):
        os.mkdir(dir_out+"test")

    filename_in = path_to_file_in.split("/")[-1]
    filename_out = filename_in.split(".root")[0]

    if "filtered" in filename_in:
        type = "F"
        graph_vector = ROOT.std.vector["TGraph"](nChannels)
    elif "images" in filename_in:
        type = "I"
        N = 32
        ntot = N * N
        image_vector = ROOT.std.vector["float"](ntot)
    elif "vars" in filename_in:
        type = "V"
        nCoincidentPairs_PA = array('i', [0])
        nHighHits_PA = array('i', [0])
        averageRMS_PA = array('f', [0.])
        averageSNR_PA = array('f', [0.])
        averageRPR_PA = array('f', [0.])
        averageKurtosis_PA = array('f', [0.])
        averageEntropy_PA = array('f', [0.])
        impulsivity_PA = array('f', [0.])
        coherentRMS_PA = array('f', [0.])
        coherentSNR_PA = array('f', [0.])
        coherentKurtosis_PA = array('f', [0.])
        coherentEntropy_PA = array('f', [0.])

        nCoincidentPairs_inIce = array('i', [0])
        nHighHits_inIce = array('i', [0])
        averageRMS_inIce = array('f', [0.])
        averageSNR_inIce = array('f', [0.])
        averageRPR_inIce = array('f', [0.])
        averageKurtosis_inIce = array('f', [0.])
        averageEntropy_inIce = array('f', [0.])
        impulsivity_inIce = array('f', [0.])
        coherentRMS_inIce = array('f', [0.])
        coherentSNR_inIce = array('f', [0.])
        coherentKurtosis_inIce = array('f', [0.])
        coherentEntropy_inIce = array('f', [0.])

        averageRMS_surface = array('f', [0.])
        averageSNR_surface = array('f', [0.])
        averageRPR_surface = array('f', [0.])
        averageKurtosis_surface = array('f', [0.])
        averageEntropy_surface = array('f', [0.])
        impulsivity_surface = array('f', [0.])
        coherentRMS_surface = array('f', [0.])
        coherentSNR_surface = array('f', [0.])
        coherentKurtosis_surface = array('f', [0.])
        coherentEntropy_surface = array('f', [0.])
    else:
        print("Unidentifiable data type, quit now.")
        quit()

    isSim = False
    if "sim" in filename_in:
        isSim = True
        if type == "F":
            treename = "events_sim"
        elif type == "I":
            treename = "images_sig"
        elif type == "V":
            treename = "vars_sig"
    else:
        if type == "F":
            treename = "events"
        elif type == "I":
            treename = "images_bkg"
        elif type == "V":
            treename = "vars_bkg"

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

    file_in = TFile.Open(path_to_file_in)
    tree_in = file_in.Get(treename)
    tree_in.GetEntry(0)
    station = tree_in.station_number
    print(f"Station {station}")
    nEvents = tree_in.GetEntries()
    print(f"Number of total events: {nEvents}")

    if type == "F":
        tree_in.SetBranchAddress("waveform_graphs", ROOT.AddressOf(graph_vector))
    elif type == "I":
        tree_in.SetBranchAddress("image", ROOT.AddressOf(image_vector))

    tree_out_train = TTree(treename, treename)
    file_out_train = TFile(dir_out+"train/"+filename_out+"_train.root", "recreate")
    if type == "F":
        tree_out_train.Branch("waveform_graphs", "std::vector<TGraph>", graph_vector)
    elif type == "I":
        tree_out_train.Branch("image", "std::vector<float>", image_vector)
    tree_out_train.Branch("station_number", station_number, 'station_number/I')
    tree_out_train.Branch("run_number", run_number, 'run_number/I')
    tree_out_train.Branch("event_number", event_number, 'event_number/I')
    tree_out_train.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_out_train.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')
    tree_out_train.Branch("true_radius", true_radius, 'true_radius/F')
    tree_out_train.Branch("true_theta", true_theta, 'true_theta/F')
    tree_out_train.Branch("true_phi", true_phi, 'true_phi/F')
    tree_out_train.Branch("true_source_theta", true_source_theta, 'true_source_theta/I')
    tree_out_train.Branch("true_source_phi", true_source_phi, 'true_source_phi/I')
    if type == "V":
        tree_out_train.Branch("nCoincidentPairs_PA", nCoincidentPairs_PA, 'nCoincidentPairs_PA/I')
        tree_out_train.Branch("nHighHits_PA", nHighHits_PA, 'nHighHits_PA/I')
        tree_out_train.Branch("averageRMS_PA", averageRMS_PA, 'averageRMS_PA/F')
        tree_out_train.Branch("averageSNR_PA", averageSNR_PA, 'averageSNR_PA/F')
        tree_out_train.Branch("averageRPR_PA", averageRPR_PA, 'averageRPR_PA/F')
        tree_out_train.Branch("averageKurtosis_PA", averageKurtosis_PA, 'averageKurtosis_PA/F')
        tree_out_train.Branch("averageEntropy_PA", averageEntropy_PA, 'averageEntropy_PA/F')
        tree_out_train.Branch("impulsivity_PA", impulsivity_PA, 'impulsivity_PA/F')
        tree_out_train.Branch("coherentRMS_PA", coherentRMS_PA, 'coherentRMS_PA/F')
        tree_out_train.Branch("coherentSNR_PA", coherentSNR_PA, 'coherentSNR_PA/F')
        tree_out_train.Branch("coherentKurtosis_PA", coherentKurtosis_PA, 'coherentKurtosis_PA/F')
        tree_out_train.Branch("coherentEntropy_PA", coherentEntropy_PA, 'coherentEntropy_PA/F')

        tree_out_train.Branch("nCoincidentPairs_inIce", nCoincidentPairs_inIce, 'nCoincidentPairs_inIce/I')
        tree_out_train.Branch("nHighHits_inIce", nHighHits_inIce, 'nHighHits_inIce/I')
        tree_out_train.Branch("averageRMS_inIce", averageRMS_inIce, 'averageRMS_inIce/F')
        tree_out_train.Branch("averageSNR_inIce", averageSNR_inIce, 'averageSNR_inIce/F')
        tree_out_train.Branch("averageRPR_inIce", averageRPR_inIce, 'averageRPR_inIce/F')
        tree_out_train.Branch("averageKurtosis_inIce", averageKurtosis_inIce, 'averageKurtosis_inIce/F')
        tree_out_train.Branch("averageEntropy_inIce", averageEntropy_inIce, 'averageEntropy_inIce/F')
        tree_out_train.Branch("impulsivity_inIce", impulsivity_inIce, 'impulsivity_inIce/F')
        tree_out_train.Branch("coherentRMS_inIce", coherentRMS_inIce, 'coherentRMS_inIce/F')
        tree_out_train.Branch("coherentSNR_inIce", coherentSNR_inIce, 'coherentSNR_inIce/F')
        tree_out_train.Branch("coherentKurtosis_inIce", coherentKurtosis_inIce, 'coherentKurtosis_inIce/F')
        tree_out_train.Branch("coherentEntropy_inIce", coherentEntropy_inIce, 'coherentEntropy_inIce/F')

        tree_out_train.Branch("averageRMS_surface", averageRMS_surface, 'averageRMS_surface/F')
        tree_out_train.Branch("averageSNR_surface", averageSNR_surface, 'averageSNR_surface/F')
        tree_out_train.Branch("averageRPR_surface", averageRPR_surface, 'averageRPR_surface/F')
        tree_out_train.Branch("averageKurtosis_surface", averageKurtosis_surface, 'averageKurtosis_surface/F')
        tree_out_train.Branch("averageEntropy_surface", averageEntropy_surface, 'averageEntropy_surface/F')
        tree_out_train.Branch("impulsivity_surface", impulsivity_surface, 'impulsivity_surface/F')
        tree_out_train.Branch("coherentRMS_surface", coherentRMS_surface, 'coherentRMS_surface/F')
        tree_out_train.Branch("coherentSNR_surface", coherentSNR_surface, 'coherentSNR_surface/F')
        tree_out_train.Branch("coherentKurtosis_surface", coherentKurtosis_surface, 'coherentKurtosis_surface/F')
        tree_out_train.Branch("coherentEntropy_surface", coherentEntropy_surface, 'coherentEntropy_surface/F')
    tree_out_train.SetDirectory(file_out_train)

    tree_out_test = TTree(treename, treename)
    file_out_test = TFile(dir_out+"test/"+filename_out+"_test.root", "recreate")
    if type == "F":
        tree_out_test.Branch("waveform_graphs", "std::vector<TGraph>", graph_vector)
    elif type == "I":
        tree_out_test.Branch("image", "std::vector<float>", image_vector)
    tree_out_test.Branch("station_number", station_number, 'station_number/I')
    tree_out_test.Branch("run_number", run_number, 'run_number/I')
    tree_out_test.Branch("event_number", event_number, 'event_number/I')
    tree_out_test.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_out_test.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')
    tree_out_test.Branch("true_radius", true_radius, 'true_radius/F')
    tree_out_test.Branch("true_theta", true_theta, 'true_theta/F')
    tree_out_test.Branch("true_phi", true_phi, 'true_phi/F')
    tree_out_test.Branch("true_source_theta", true_source_theta, 'true_source_theta/I')
    tree_out_test.Branch("true_source_phi", true_source_phi, 'true_source_phi/I')
    if type == "V":
        tree_out_test.Branch("nCoincidentPairs_PA", nCoincidentPairs_PA, 'nCoincidentPairs_PA/I')
        tree_out_test.Branch("nHighHits_PA", nHighHits_PA, 'nHighHits_PA/I')
        tree_out_test.Branch("averageRMS_PA", averageRMS_PA, 'averageRMS_PA/F')
        tree_out_test.Branch("averageSNR_PA", averageSNR_PA, 'averageSNR_PA/F')
        tree_out_test.Branch("averageRPR_PA", averageRPR_PA, 'averageRPR_PA/F')
        tree_out_test.Branch("averageKurtosis_PA", averageKurtosis_PA, 'averageKurtosis_PA/F')
        tree_out_test.Branch("averageEntropy_PA", averageEntropy_PA, 'averageEntropy_PA/F')
        tree_out_test.Branch("impulsivity_PA", impulsivity_PA, 'impulsivity_PA/F')
        tree_out_test.Branch("coherentRMS_PA", coherentRMS_PA, 'coherentRMS_PA/F')
        tree_out_test.Branch("coherentSNR_PA", coherentSNR_PA, 'coherentSNR_PA/F')
        tree_out_test.Branch("coherentKurtosis_PA", coherentKurtosis_PA, 'coherentKurtosis_PA/F')
        tree_out_test.Branch("coherentEntropy_PA", coherentEntropy_PA, 'coherentEntropy_PA/F')

        tree_out_test.Branch("nCoincidentPairs_inIce", nCoincidentPairs_inIce, 'nCoincidentPairs_inIce/I')
        tree_out_test.Branch("nHighHits_inIce", nHighHits_inIce, 'nHighHits_inIce/I')
        tree_out_test.Branch("averageRMS_inIce", averageRMS_inIce, 'averageRMS_inIce/F')
        tree_out_test.Branch("averageSNR_inIce", averageSNR_inIce, 'averageSNR_inIce/F')
        tree_out_test.Branch("averageRPR_inIce", averageRPR_inIce, 'averageRPR_inIce/F')
        tree_out_test.Branch("averageKurtosis_inIce", averageKurtosis_inIce, 'averageKurtosis_inIce/F')
        tree_out_test.Branch("averageEntropy_inIce", averageEntropy_inIce, 'averageEntropy_inIce/F')
        tree_out_test.Branch("impulsivity_inIce", impulsivity_inIce, 'impulsivity_inIce/F')
        tree_out_test.Branch("coherentRMS_inIce", coherentRMS_inIce, 'coherentRMS_inIce/F')
        tree_out_test.Branch("coherentSNR_inIce", coherentSNR_inIce, 'coherentSNR_inIce/F')
        tree_out_test.Branch("coherentKurtosis_inIce", coherentKurtosis_inIce, 'coherentKurtosis_inIce/F')
        tree_out_test.Branch("coherentEntropy_inIce", coherentEntropy_inIce, 'coherentEntropy_inIce/F')

        tree_out_test.Branch("averageRMS_surface", averageRMS_surface, 'averageRMS_surface/F')
        tree_out_test.Branch("averageSNR_surface", averageSNR_surface, 'averageSNR_surface/F')
        tree_out_test.Branch("averageRPR_surface", averageRPR_surface, 'averageRPR_surface/F')
        tree_out_test.Branch("averageKurtosis_surface", averageKurtosis_surface, 'averageKurtosis_surface/F')
        tree_out_test.Branch("averageEntropy_surface", averageEntropy_surface, 'averageEntropy_surface/F')
        tree_out_test.Branch("impulsivity_surface", impulsivity_surface, 'impulsivity_surface/F')
        tree_out_test.Branch("coherentRMS_surface", coherentRMS_surface, 'coherentRMS_surface/F')
        tree_out_test.Branch("coherentSNR_surface", coherentSNR_surface, 'coherentSNR_surface/F')
        tree_out_test.Branch("coherentKurtosis_surface", coherentKurtosis_surface, 'coherentKurtosis_surface/F')
        tree_out_test.Branch("coherentEntropy_surface", coherentEntropy_surface, 'coherentEntropy_surface/F')
    tree_out_test.SetDirectory(file_out_test)

    if type == "V" and path_to_second_file_in:
        N = 32
        ntot = N * N
        image_vector = ROOT.std.vector["float"](ntot)

        if isSim:
            treename_2 = "images_sig"
        else:
            treename_2 = "images_bkg"

        file_in_2 = TFile.Open(path_to_second_file_in)
        tree_in_2 = file_in_2.Get(treename_2)
        tree_in_2.SetBranchAddress("image", ROOT.AddressOf(image_vector))

        filename_out_2 = "images" + filename_out.split("vars")[1]

        tree_out_train_2 = TTree(treename_2, treename_2)
        file_out_train_2 = TFile(dir_out+"train/"+filename_out_2+"_train.root", "recreate")
        tree_out_train_2.Branch("image", "std::vector<float>", image_vector)
        tree_out_train_2.Branch("station_number", station_number, 'station_number/I')
        tree_out_train_2.Branch("run_number", run_number, 'run_number/I')
        tree_out_train_2.Branch("event_number", event_number, 'event_number/I')
        tree_out_train_2.Branch("sim_energy", sim_energy, 'sim_energy/F')
        tree_out_train_2.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')
        tree_out_train_2.Branch("true_radius", true_radius, 'true_radius/F')
        tree_out_train_2.Branch("true_theta", true_theta, 'true_theta/F')
        tree_out_train_2.Branch("true_phi", true_phi, 'true_phi/F')
        tree_out_train_2.Branch("true_source_theta", true_source_theta, 'true_source_theta/I')
        tree_out_train_2.Branch("true_source_phi", true_source_phi, 'true_source_phi/I')
        tree_out_train_2.SetDirectory(file_out_train_2)

        tree_out_test_2 = TTree(treename_2, treename_2)
        file_out_test_2 = TFile(dir_out+"test/"+filename_out_2+"_test.root", "recreate")
        tree_out_test_2.Branch("image", "std::vector<float>", image_vector)
        tree_out_test_2.Branch("station_number", station_number, 'station_number/I')
        tree_out_test_2.Branch("run_number", run_number, 'run_number/I')
        tree_out_test_2.Branch("event_number", event_number, 'event_number/I')
        tree_out_test_2.Branch("sim_energy", sim_energy, 'sim_energy/F')
        tree_out_test_2.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')
        tree_out_test_2.Branch("true_radius", true_radius, 'true_radius/F')
        tree_out_test_2.Branch("true_theta", true_theta, 'true_theta/F')
        tree_out_test_2.Branch("true_phi", true_phi, 'true_phi/F')
        tree_out_test_2.Branch("true_source_theta", true_source_theta, 'true_source_theta/I')
        tree_out_test_2.Branch("true_source_phi", true_source_phi, 'true_source_phi/I')
        tree_out_test_2.SetDirectory(file_out_test_2)

    nCounts = 0
    for i_event in range(nEvents):
        tree_in.GetEntry(i_event)
        if path_to_second_file_in:
            tree_in_2.GetEntry(i_event)

        station_number[0] = tree_in.station_number
        run_number[0] = tree_in.run_number
        event_number[0] = tree_in.event_number
        sim_energy[0] = tree_in.sim_energy
        trigger_time_difference[0] = tree_in.trigger_time_difference
        true_radius[0] = tree_in.true_radius
        true_theta[0] = tree_in.true_theta
        true_phi[0] = tree_in.true_phi
        true_source_theta[0] = tree_in.true_source_theta
        true_source_phi[0] = tree_in.true_source_phi

        if type == "V":
            nCoincidentPairs_PA[0] = tree_in.nCoincidentPairs_PA
            nHighHits_PA[0] = tree_in.nHighHits_PA
            averageRMS_PA[0] = tree_in.averageRMS_PA
            averageSNR_PA[0] = tree_in.averageSNR_PA
            averageRPR_PA[0] = tree_in.averageRPR_PA
            averageKurtosis_PA[0] = tree_in.averageKurtosis_PA
            averageEntropy_PA[0] = tree_in.averageEntropy_PA
            impulsivity_PA[0] = tree_in.impulsivity_PA
            coherentRMS_PA[0] = tree_in.coherentRMS_PA
            coherentSNR_PA[0] = tree_in.coherentSNR_PA
            coherentKurtosis_PA[0] = tree_in.coherentKurtosis_PA
            coherentEntropy_PA[0] = tree_in.coherentEntropy_PA

            nCoincidentPairs_inIce[0] = tree_in.nCoincidentPairs_inIce
            nHighHits_inIce[0] = tree_in.nHighHits_inIce
            averageRMS_inIce[0] = tree_in.averageRMS_inIce
            averageSNR_inIce[0] = tree_in.averageSNR_inIce
            averageRPR_inIce[0] = tree_in.averageRPR_inIce
            averageKurtosis_inIce[0] = tree_in.averageKurtosis_inIce
            averageEntropy_inIce[0] = tree_in.averageEntropy_inIce
            impulsivity_inIce[0] = tree_in.impulsivity_inIce
            coherentRMS_inIce[0] = tree_in.coherentRMS_inIce
            coherentSNR_inIce[0] = tree_in.coherentSNR_inIce
            coherentKurtosis_inIce[0] = tree_in.coherentKurtosis_inIce
            coherentEntropy_inIce[0] = tree_in.coherentEntropy_inIce

            averageRMS_surface[0] = tree_in.averageRMS_surface
            averageSNR_surface[0] = tree_in.averageSNR_surface
            averageRPR_surface[0] = tree_in.averageRPR_surface
            averageKurtosis_surface[0] = tree_in.averageKurtosis_surface
            averageEntropy_surface[0] = tree_in.averageEntropy_surface
            impulsivity_surface[0] = tree_in.impulsivity_surface
            coherentRMS_surface[0] = tree_in.coherentRMS_surface
            coherentSNR_surface[0] = tree_in.coherentSNR_surface
            coherentKurtosis_surface[0] = tree_in.coherentKurtosis_surface
            coherentEntropy_surface[0] = tree_in.coherentEntropy_surface

        if isSim or type == "I":
            extraConditions = True
        else:
            extraConditions = False

            if type == "F":
                SNR = np.array([])
                trace_PA = []
                for i_channel in range(nChannels):
                    wf = graph_vector[i_channel]
                    y = np.array( wf.GetY() )
                    RMS = trace_utilities.get_split_trace_noise_RMS(y)
                    SNR = np.append( SNR, trace_utilities.get_signal_to_noise_ratio(y, RMS) )

                    if i_channel < nChannels_PA:
                        trace_PA.append(y)

                trace_PA = np.array(trace_PA)
                refIndex_PA, refIndex_inIce, refIndex_surface = MakeVariables.getReferenceTraceIndices(SNR)

                # Calculations: CSW variables (PA channels)
                csw = trace_utilities.get_coherent_sum( np.delete(trace_PA, refIndex_PA, axis=0), trace_PA[refIndex_PA] )
                impulsivity_PA = trace_utilities.get_impulsivity(csw)
                coherentRMS_PA = trace_utilities.get_split_trace_noise_RMS(csw)
                coherentSNR_PA = trace_utilities.get_signal_to_noise_ratio(csw, coherentRMS_PA)
                coherentKurtosis_PA = trace_utilities.get_kurtosis(csw)
                coherentEntropy_PA = trace_utilities.get_entropy(csw)

                if coherentEntropy_PA > 4.5 and coherentKurtosis_PA < 1.3 and coherentSNR_PA < 3.5 and impulsivity_PA < 0.18:
                    extraConditions = True

            elif type == "V":
                if coherentEntropy_PA[0] > 4.5 and coherentKurtosis_PA[0] < 1.3 and coherentSNR_PA[0] < 3.5 and impulsivity_PA[0] < 0.18:
                    extraConditions = True

        if division:
            if int(i_event % division) == 0 and extraConditions:
                tree_out_train.Fill()
                if path_to_second_file_in:
                    tree_out_train_2.Fill()
            else:
                tree_out_test.Fill()
                if path_to_second_file_in:
                    tree_out_test_2.Fill()
        elif count:
            if nCounts < count and extraConditions:
                tree_out_train.Fill()
                if path_to_second_file_in:
                    tree_out_train_2.Fill()
            else:
                tree_out_test.Fill()
                if path_to_second_file_in:
                    tree_out_test_2.Fill()
            nCounts += 1

    print(f"Number of events for training: {tree_out_train.GetEntries()}")
    print(f"Number of events for testing: {tree_out_test.GetEntries()}")

    file_out_train.cd()
    tree_out_train.Write()
    file_out_train.Close()

    file_out_test.cd()
    tree_out_test.Write()
    file_out_test.Close()

    file_in.Close()

    if path_to_second_file_in:
        file_out_train_2.cd()
        tree_out_train_2.Write()
        file_out_train_2.Close()

        file_out_test_2.cd()
        tree_out_test_2.Write()
        file_out_test_2.Close()

        file_in_2.Close()
