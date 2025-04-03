import argparse
import numpy as np
from array import array

import ROOT
from ROOT import TFile, TTree

import MakeVariables
from NuRadioReco.utilities import trace_utilities
import NuRadioReco.modules.RNO_G.stationHitFilter

nChannels = 24
inIceChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_file_in", type=str, help="Path to the input file")
    parser.add_argument("dir_out", type=str, help="Output directory")
    args = parser.parse_args()

    path_to_file_in = args.path_to_file_in
    dir_out = args.dir_out
    if not dir_out.endswith("/"):
        dir_out += "/"

    filename_in = path_to_file_in.split("/")[-1]
    filename = filename_in.split("filtered_")[1]
    filename_out = f"vars_{filename}"

    if "sim" in filename_in:
        treename_in = "events_sim"
        treename_out = "vars_sig"
    else:
        treename_in = "events"
        treename_out = "vars_bkg"

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

    tree_out = ROOT.TTree(treename_out, treename_out)

    file_out = TFile(dir_out+filename_out, "RECREATE")

    tree_out.SetDirectory(file_out)

    station_number = array('i', [0])
    run_number = array('i', [0])
    event_number = array('i', [0])
    sim_energy = array('f', [0.])
    trigger_time_difference = array('f', [0.])

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

    tree_out.Branch("station_number", station_number, 'station_number/I')
    tree_out.Branch("run_number", run_number, 'run_number/I')
    tree_out.Branch("event_number", event_number, 'event_number/I')
    tree_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_out.Branch("trigger_time_difference", trigger_time_difference, 'trigger_time_difference/F')

    tree_out.Branch("nCoincidentPairs_PA", nCoincidentPairs_PA, 'nCoincidentPairs_PA/I')
    tree_out.Branch("nHighHits_PA", nHighHits_PA, 'nHighHits_PA/I')
    tree_out.Branch("averageRMS_PA", averageRMS_PA, 'averageRMS_PA/F')
    tree_out.Branch("averageSNR_PA", averageSNR_PA, 'averageSNR_PA/F')
    tree_out.Branch("averageRPR_PA", averageRPR_PA, 'averageRPR_PA/F')
    tree_out.Branch("averageKurtosis_PA", averageKurtosis_PA, 'averageKurtosis_PA/F')
    tree_out.Branch("averageEntropy_PA", averageEntropy_PA, 'averageEntropy_PA/F')
    tree_out.Branch("impulsivity_PA", impulsivity_PA, 'impulsivity_PA/F')
    tree_out.Branch("coherentRMS_PA", coherentRMS_PA, 'coherentRMS_PA/F')
    tree_out.Branch("coherentSNR_PA", coherentSNR_PA, 'coherentSNR_PA/F')
    tree_out.Branch("coherentKurtosis_PA", coherentKurtosis_PA, 'coherentKurtosis_PA/F')
    tree_out.Branch("coherentEntropy_PA", coherentEntropy_PA, 'coherentEntropy_PA/F')

    tree_out.Branch("nCoincidentPairs_inIce", nCoincidentPairs_inIce, 'nCoincidentPairs_inIce/I')
    tree_out.Branch("nHighHits_inIce", nHighHits_inIce, 'nHighHits_inIce/I')
    tree_out.Branch("averageRMS_inIce", averageRMS_inIce, 'averageRMS_inIce/F')
    tree_out.Branch("averageSNR_inIce", averageSNR_inIce, 'averageSNR_inIce/F')
    tree_out.Branch("averageRPR_inIce", averageRPR_inIce, 'averageRPR_inIce/F')
    tree_out.Branch("averageKurtosis_inIce", averageKurtosis_inIce, 'averageKurtosis_inIce/F')
    tree_out.Branch("averageEntropy_inIce", averageEntropy_inIce, 'averageEntropy_inIce/F')
    tree_out.Branch("impulsivity_inIce", impulsivity_inIce, 'impulsivity_inIce/F')
    tree_out.Branch("coherentRMS_inIce", coherentRMS_inIce, 'coherentRMS_inIce/F')
    tree_out.Branch("coherentSNR_inIce", coherentSNR_inIce, 'coherentSNR_inIce/F')
    tree_out.Branch("coherentKurtosis_inIce", coherentKurtosis_inIce, 'coherentKurtosis_inIce/F')
    tree_out.Branch("coherentEntropy_inIce", coherentEntropy_inIce, 'coherentEntropy_inIce/F')

    tree_out.Branch("averageRMS_surface", averageRMS_surface, 'averageRMS_surface/F')
    tree_out.Branch("averageSNR_surface", averageSNR_surface, 'averageSNR_surface/F')
    tree_out.Branch("averageRPR_surface", averageRPR_surface, 'averageRPR_surface/F')
    tree_out.Branch("averageKurtosis_surface", averageKurtosis_surface, 'averageKurtosis_surface/F')
    tree_out.Branch("averageEntropy_surface", averageEntropy_surface, 'averageEntropy_surface/F')
    tree_out.Branch("impulsivity_surface", impulsivity_surface, 'impulsivity_surface/F')
    tree_out.Branch("coherentRMS_surface", coherentRMS_surface, 'coherentRMS_surface/F')
    tree_out.Branch("coherentSNR_surface", coherentSNR_surface, 'coherentSNR_surface/F')
    tree_out.Branch("coherentKurtosis_surface", coherentKurtosis_surface, 'coherentKurtosis_surface/F')
    tree_out.Branch("coherentEntropy_surface", coherentEntropy_surface, 'coherentEntropy_surface/F')

    # Initialize Hit Filter
    HF = NuRadioReco.modules.RNO_G.stationHitFilter.stationHitFilter(complete_time_check=True, complete_hit_check=True)
    HF.begin()

    for i_event in range(nEvents):
        tree_in.GetEntry(i_event)

        station_number[0] =  tree_in.station_number
        run_number[0] = tree_in.run_number
        event_number[0] = tree_in.event_number
        sim_energy[0] = tree_in.sim_energy
        trigger_time_difference[0] = tree_in.trigger_time_difference

        trace_PA = []
        trace_surface = []
        trace_inIce = []
        trace_HF = []
        times_HF = []
        RMS_HF = np.array([])
        RMS = np.array([])
        SNR = np.array([])
        RPR = np.array([])
        kurtosis = np.array([])
        entropy = np.array([])

        sumRMS_PA = 0
        sumSNR_PA = 0
        sumRPR_PA = 0
        sumKurtosis_PA = 0
        sumEntropy_PA = 0

        sumRMS_inIce = 0
        sumSNR_inIce = 0
        sumRPR_inIce = 0
        sumKurtosis_inIce = 0
        sumEntropy_inIce = 0

        sumRMS_surface = 0
        sumSNR_surface = 0
        sumRPR_surface = 0
        sumKurtosis_surface = 0
        sumEntropy_surface = 0

        for i_channel in range(nChannels):
            wf = graph_vector[i_channel]
            y = np.array( wf.GetY() )
            x = np.array( wf.GetX() )

            # Calculations: RMS, SNR, Kurtosis, Entropy
            RMS = np.append( RMS, trace_utilities.get_split_trace_noise_RMS(y) )
            SNR = np.append( SNR, trace_utilities.get_signal_to_noise_ratio(y, RMS[i_channel]) )
            RPR = np.append( RPR, trace_utilities.get_root_power_ratio(y, x, RMS[i_channel]) )
            kurtosis = np.append( kurtosis, trace_utilities.get_kurtosis(y) )
            entropy = np.append( entropy, trace_utilities.get_entropy(y) )

            if i_channel < 4:
                trace_PA.append(y)
                sumRMS_PA += RMS[i_channel]
                sumSNR_PA += SNR[i_channel]
                sumRPR_PA += RPR[i_channel]
                sumKurtosis_PA += kurtosis[i_channel]
                sumEntropy_PA += entropy[i_channel]

            if i_channel in inIceChannels:
                trace_inIce.append(y)
                trace_HF.append(y)
                times_HF.append(x)
                RMS_HF = np.append(RMS_HF, RMS[i_channel])
                sumRMS_inIce += RMS[i_channel]
                sumSNR_inIce += SNR[i_channel]
                sumRPR_inIce += RPR[i_channel]
                sumKurtosis_inIce += kurtosis[i_channel]
                sumEntropy_inIce += entropy[i_channel]
            else:
                trace_surface.append(y)
                sumRMS_surface += RMS[i_channel]
                sumSNR_surface += SNR[i_channel]
                sumRPR_surface += RPR[i_channel]
                sumKurtosis_surface += kurtosis[i_channel]
                sumEntropy_surface += entropy[i_channel]

            del wf

        trace_PA = np.array(trace_PA)
        trace_surface = np.array(trace_surface)
        trace_inIce = np.array(trace_inIce)
        trace_HF = np.array(trace_HF)
        times_HF = np.array(times_HF)

        ### Hit Filter ###
        HF.set_up(trace_HF, times_HF, RMS_HF)
        isPassedHF = HF.apply_hit_filter()
        inTimeWindow = HF.is_in_time_window()
        overHitThreshold = HF.is_over_hit_threshold()

        nCoincidentPairs_PA[0] = sum(inTimeWindow[0])
        nCoincidentPairs_inIce[0] = nCoincidentPairs_PA[0]
        nHighHits_PA[0] = 0
        for i_channel in range(15):
            if i_channel < 4:
                nHighHits_PA[0] += overHitThreshold[i_channel]
                if i_channel > 0:
                    nCoincidentPairs_inIce[0] += inTimeWindow[i_channel][0]

            nHighHits_inIce[0] += overHitThreshold[i_channel]

        ### Variables ###
        averageRMS_PA[0] = sumRMS_PA / 4
        averageSNR_PA[0] = sumSNR_PA / 4
        averageRPR_PA[0] = sumRPR_PA / 4
        averageKurtosis_PA[0] = sumKurtosis_PA / 4
        averageEntropy_PA[0] = sumEntropy_PA / 4

        averageRMS_inIce[0] = sumRMS_inIce / 15
        averageSNR_inIce[0] = sumSNR_inIce / 15
        averageRPR_inIce[0] = sumRPR_inIce / 15
        averageKurtosis_inIce[0] = sumKurtosis_inIce / 15
        averageEntropy_inIce[0] = sumEntropy_inIce / 15

        averageRMS_surface[0] = sumRMS_surface / 9
        averageSNR_surface[0] = sumSNR_surface / 9
        averageRPR_surface[0] = sumRPR_surface / 9
        averageKurtosis_surface[0] = sumKurtosis_surface / 9
        averageEntropy_surface[0] = sumEntropy_surface / 9

        ### CSW variables ###
        refIndex_PA, refIndex_inIce, refIndex_surface = MakeVariables.getReferenceTraceIndices(SNR)

        # Calculations: CSW & Impulsivity (PA channels)
        csw = trace_utilities.get_coherent_sum( np.delete(trace_PA, refIndex_PA, axis=0), trace_PA[refIndex_PA] )
        impulsivity_PA[0] = trace_utilities.get_impulsivity(csw)
        coherentRMS_PA[0] = trace_utilities.get_split_trace_noise_RMS(csw)
        coherentSNR_PA[0] = trace_utilities.get_signal_to_noise_ratio(csw, coherentRMS_PA[0])
        coherentKurtosis_PA[0] = trace_utilities.get_kurtosis(csw)
        coherentEntropy_PA[0] = trace_utilities.get_entropy(csw)

        # Calculations: CSW & Impulsivity (in-ice channels)
        csw = trace_utilities.get_coherent_sum( np.delete(trace_inIce, refIndex_inIce, axis=0), trace_inIce[refIndex_inIce] )
        impulsivity_inIce[0] = trace_utilities.get_impulsivity(csw)
        coherentRMS_inIce[0] = trace_utilities.get_split_trace_noise_RMS(csw)
        coherentSNR_inIce[0] = trace_utilities.get_signal_to_noise_ratio(csw, coherentRMS_inIce[0])
        coherentKurtosis_inIce[0] = trace_utilities.get_kurtosis(csw)
        coherentEntropy_inIce[0] = trace_utilities.get_entropy(csw)

        # Calculations: CSW & Impulsivity (surface channels)
        csw = trace_utilities.get_coherent_sum( np.delete(trace_surface, refIndex_surface, axis=0), trace_surface[refIndex_surface] )
        impulsivity_surface[0] = trace_utilities.get_impulsivity(csw)
        coherentRMS_surface[0] = trace_utilities.get_split_trace_noise_RMS(csw)
        coherentSNR_surface[0] = trace_utilities.get_signal_to_noise_ratio(csw, coherentRMS_surface[0])
        coherentKurtosis_surface[0] = trace_utilities.get_kurtosis(csw)
        coherentEntropy_surface[0] = trace_utilities.get_entropy(csw)


        tree_out.Fill()

    tree_out.Write()

    file_in.Close()
    file_out.Close()
