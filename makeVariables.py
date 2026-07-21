import argparse
import numpy as np
from array import array

import ROOT
from ROOT import TFile, TTree

from NuRadioReco.utilities import trace_utilities


nChannels = 24
inIceChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
nChannels_inIce = len(inIceChannels)


def getReferenceTraceIndices(vars, find_max=False):
    if len(vars) not in {nChannels, nChannels_inIce}:
        raise ValueError("Need vars for all 24 channels or for all 15 in-ice channels.")

    inIceOnly = len(vars) == nChannels_inIce
    vars = np.array(vars)

    vars_PA = vars[:4]
    vars_inIce = vars if inIceOnly else vars[inIceChannels]
    vars_surface = [] if inIceOnly else np.delete(vars, inIceChannels)

    func = np.argmax if find_max else np.argmin

    refIndex_PA = func(vars_PA)
    refIndex_inIce = func(vars_inIce)
    refIndex_surface = func(vars_surface) if not inIceOnly else 0

    return refIndex_PA, refIndex_inIce, refIndex_surface


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
    shower_energy = array('f', [0.])
    inelasticity = array('f', [0.])
    interaction_type = array('i', [0])

    trigger_time = array('d', [0.])

    true_radius = array('f', [0.])
    true_theta = array('f', [0.])
    true_phi = array('f', [0.])
    true_source_theta = array('i', [0])
    true_source_phi = array('i', [0])

    reco_max_corr = array('f', [np.nan])
    reco_surf_corr_z = array("f", [np.nan])
    reco_surf_corr_zen = array("f", [np.nan])
    reco_rho = array('f', [np.nan])
    reco_phi = array('f', [np.nan])
    reco_z = array('f', [np.nan])

    passed_hit_filter = array('i', [0])
    nCoincidentPairs_PA = array('i', [0])
    nHighHits_PA = array('i', [0])
    nCoincidentPairs_inIce = array('i', [0])
    nHighHits_inIce = array('i', [0])


    averageRMS_PA = array('f', [0.])
    averageSNR_PA = array('f', [0.])
    averageKurtosis_PA = array('f', [0.])
    averageEntropy_PA = array('f', [0.])
    averageImpulsivity_PA = array('f', [0.])

    coherentRMS_PA = array('f', [0.])
    coherentSNR_PA = array('f', [0.])
    coherentKurtosis_PA = array('f', [0.])
    coherentEntropy_PA = array('f', [0.])
    coherentImpulsivity_PA = array('f', [0.])


    averageRMS_inIce = array('f', [0.])
    averageSNR_inIce = array('f', [0.])
    averageKurtosis_inIce = array('f', [0.])
    averageEntropy_inIce = array('f', [0.])
    averageImpulsivity_inIce = array('f', [0.])

    coherentRMS_inIce = array('f', [0.])
    coherentSNR_inIce = array('f', [0.])
    coherentKurtosis_inIce = array('f', [0.])
    coherentEntropy_inIce = array('f', [0.])
    coherentImpulsivity_inIce = array('f', [0.])


    averageRMS_surface = array('f', [0.])
    averageSNR_surface = array('f', [0.])
    averageKurtosis_surface = array('f', [0.])
    averageEntropy_surface = array('f', [0.])
    averageImpulsivity_surface = array('f', [0.])

    coherentRMS_surface = array('f', [0.])
    coherentSNR_surface = array('f', [0.])
    coherentKurtosis_surface = array('f', [0.])
    coherentEntropy_surface = array('f', [0.])
    coherentImpulsivity_surface = array('f', [0.])


    tree_out.Branch("station_number", station_number, 'station_number/I')
    tree_out.Branch("run_number", run_number, 'run_number/I')
    tree_out.Branch("event_number", event_number, 'event_number/I')

    tree_out.Branch("sim_energy", sim_energy, 'sim_energy/F')
    tree_out.Branch("shower_energy", shower_energy, 'shower_energy/F')
    tree_out.Branch("inelasticity", inelasticity, 'inelasticity/F')
    tree_out.Branch("interaction_type", interaction_type, 'interaction_type/I')

    tree_out.Branch("trigger_time", trigger_time, 'trigger_time/D')

    tree_out.Branch("true_radius", true_radius, 'true_radius/F')
    tree_out.Branch("true_theta", true_theta, 'true_theta/F')
    tree_out.Branch("true_phi", true_phi, 'true_phi/F')
    tree_out.Branch("true_source_theta", true_source_theta, 'true_source_theta/I')
    tree_out.Branch("true_source_phi", true_source_phi, 'true_source_phi/I')

    tree_out.Branch("reco_max_corr", reco_max_corr, 'reco_max_corr/F')
    tree_out.Branch("reco_surf_corr_z", reco_surf_corr_z, 'reco_surf_corr_z/F')
    tree_out.Branch("reco_surf_corr_zen", reco_surf_corr_zen, 'reco_surf_corr_zen/F')
    tree_out.Branch("reco_rho", reco_rho, 'reco_rho/F')
    tree_out.Branch("reco_phi", reco_phi, 'reco_phi/F')
    tree_out.Branch("reco_z", reco_z, 'reco_z/F')

    tree_out.Branch("passed_hit_filter", passed_hit_filter, 'passed_hit_filter/I')
    tree_out.Branch("nCoincidentPairs_PA", nCoincidentPairs_PA, 'nCoincidentPairs_PA/I')
    tree_out.Branch("nHighHits_PA", nHighHits_PA, 'nHighHits_PA/I')
    tree_out.Branch("nCoincidentPairs_inIce", nCoincidentPairs_inIce, 'nCoincidentPairs_inIce/I')
    tree_out.Branch("nHighHits_inIce", nHighHits_inIce, 'nHighHits_inIce/I')


    tree_out.Branch("averageRMS_PA", averageRMS_PA, 'averageRMS_PA/F')
    tree_out.Branch("averageSNR_PA", averageSNR_PA, 'averageSNR_PA/F')
    tree_out.Branch("averageKurtosis_PA", averageKurtosis_PA, 'averageKurtosis_PA/F')
    tree_out.Branch("averageEntropy_PA", averageEntropy_PA, 'averageEntropy_PA/F')
    tree_out.Branch("averageImpulsivity_PA", averageImpulsivity_PA, 'averageImpulsivity_PA/F')

    tree_out.Branch("coherentRMS_PA", coherentRMS_PA, 'coherentRMS_PA/F')
    tree_out.Branch("coherentSNR_PA", coherentSNR_PA, 'coherentSNR_PA/F')
    tree_out.Branch("coherentKurtosis_PA", coherentKurtosis_PA, 'coherentKurtosis_PA/F')
    tree_out.Branch("coherentEntropy_PA", coherentEntropy_PA, 'coherentEntropy_PA/F')
    tree_out.Branch("coherentImpulsivity_PA", coherentImpulsivity_PA, 'coherentImpulsivity_PA/F')


    tree_out.Branch("averageRMS_inIce", averageRMS_inIce, 'averageRMS_inIce/F')
    tree_out.Branch("averageSNR_inIce", averageSNR_inIce, 'averageSNR_inIce/F')
    tree_out.Branch("averageKurtosis_inIce", averageKurtosis_inIce, 'averageKurtosis_inIce/F')
    tree_out.Branch("averageEntropy_inIce", averageEntropy_inIce, 'averageEntropy_inIce/F')
    tree_out.Branch("averageImpulsivity_inIce", averageImpulsivity_inIce, 'averageImpulsivity_inIce/F')

    tree_out.Branch("coherentRMS_inIce", coherentRMS_inIce, 'coherentRMS_inIce/F')
    tree_out.Branch("coherentSNR_inIce", coherentSNR_inIce, 'coherentSNR_inIce/F')
    tree_out.Branch("coherentKurtosis_inIce", coherentKurtosis_inIce, 'coherentKurtosis_inIce/F')
    tree_out.Branch("coherentEntropy_inIce", coherentEntropy_inIce, 'coherentEntropy_inIce/F')
    tree_out.Branch("coherentImpulsivity_inIce", coherentImpulsivity_inIce, 'coherentImpulsivity_inIce/F')


    tree_out.Branch("averageRMS_surface", averageRMS_surface, 'averageRMS_surface/F')
    tree_out.Branch("averageSNR_surface", averageSNR_surface, 'averageSNR_surface/F')
    tree_out.Branch("averageKurtosis_surface", averageKurtosis_surface, 'averageKurtosis_surface/F')
    tree_out.Branch("averageEntropy_surface", averageEntropy_surface, 'averageEntropy_surface/F')
    tree_out.Branch("averageImpulsivity_surface", averageImpulsivity_surface, 'averageImpulsivity_surface/F')

    tree_out.Branch("coherentRMS_surface", coherentRMS_surface, 'coherentRMS_surface/F')
    tree_out.Branch("coherentSNR_surface", coherentSNR_surface, 'coherentSNR_surface/F')
    tree_out.Branch("coherentKurtosis_surface", coherentKurtosis_surface, 'coherentKurtosis_surface/F')
    tree_out.Branch("coherentEntropy_surface", coherentEntropy_surface, 'coherentEntropy_surface/F')
    tree_out.Branch("coherentImpulsivity_surface", coherentImpulsivity_surface, 'coherentImpulsivity_surface/F')


    for i_event in range(nEvents):
        tree_in.GetEntry(i_event)

        station_number[0] = tree_in.station_number
        run_number[0] = tree_in.run_number
        event_number[0] = tree_in.event_number

        sim_energy[0] = tree_in.sim_energy
        shower_energy[0] = tree_in.shower_energy
        inelasticity[0] = tree_in.inelasticity
        interaction_type[0] = tree_in.interaction_type

        trigger_time[0] = tree_in.trigger_time

        true_radius[0] = tree_in.true_radius
        true_theta[0] = tree_in.true_theta
        true_phi[0] = tree_in.true_phi
        true_source_theta[0] = tree_in.true_source_theta
        true_source_phi[0] = tree_in.true_source_phi

        reco_max_corr[0] = tree_in.reco_max_corr
        reco_surf_corr_z[0] = tree_in.reco_surf_corr_z
        reco_surf_corr_zen[0] = tree_in.reco_surf_corr_zen
        reco_rho[0] = tree_in.reco_rho
        reco_phi[0] = tree_in.reco_phi
        reco_z[0] = tree_in.reco_z

        passed_hit_filter[0] = tree_in.passed_hit_filter
        nCoincidentPairs_PA[0] = tree_in.nCoincidentPairs_PA
        nCoincidentPairs_inIce[0] = tree_in.nCoincidentPairs_inIce
        nHighHits_PA[0] = tree_in.nHighHits_PA
        nHighHits_inIce[0] = tree_in.nHighHits_inIce

        trace_PA = []
        trace_surface = []
        trace_inIce = []

        RMS = np.array([])
        SNR = np.array([])
        kurtosis = np.array([])
        entropy = np.array([])
        impulsivity = np.array([])

        sumRMS_PA = 0
        sumSNR_PA = 0
        sumKurtosis_PA = 0
        sumEntropy_PA = 0
        sumImpulsivity_PA = 0

        sumRMS_inIce = 0
        sumSNR_inIce = 0
        sumKurtosis_inIce = 0
        sumEntropy_inIce = 0
        sumImpulsivity_inIce = 0

        sumRMS_surface = 0
        sumSNR_surface = 0
        sumKurtosis_surface = 0
        sumEntropy_surface = 0
        sumImpulsivity_surface = 0

        for i_channel in range(nChannels):
            wf = graph_vector[i_channel]
            y = np.array( wf.GetY() )
            x = np.array( wf.GetX() )

            # Calculations: RMS, SNR, Kurtosis, Entropy
            RMS = np.append( RMS, trace_utilities.get_split_trace_noise_RMS(y) )
            SNR = np.append( SNR, trace_utilities.get_signal_to_noise_ratio(y, RMS[i_channel], 100) )
            kurtosis = np.append( kurtosis, trace_utilities.get_kurtosis(y) )
            entropy = np.append( entropy, trace_utilities.get_entropy(y) )
            impulsivity = np.append( impulsivity, trace_utilities.get_impulsivity(y) )

            if i_channel < 4:
                trace_PA.append(y)
                sumRMS_PA += RMS[i_channel]
                sumSNR_PA += SNR[i_channel]
                sumKurtosis_PA += kurtosis[i_channel]
                sumEntropy_PA += entropy[i_channel]
                sumImpulsivity_PA += impulsivity[i_channel]

            if i_channel in inIceChannels:
                trace_inIce.append(y)
                sumRMS_inIce += RMS[i_channel]
                sumSNR_inIce += SNR[i_channel]
                sumKurtosis_inIce += kurtosis[i_channel]
                sumEntropy_inIce += entropy[i_channel]
                sumImpulsivity_inIce += impulsivity[i_channel]
            else:
                trace_surface.append(y)
                sumRMS_surface += RMS[i_channel]
                sumSNR_surface += SNR[i_channel]
                sumKurtosis_surface += kurtosis[i_channel]
                sumEntropy_surface += entropy[i_channel]
                sumImpulsivity_surface += impulsivity[i_channel]

            del wf


        trace_PA = np.array(trace_PA)
        trace_surface = np.array(trace_surface)
        trace_inIce = np.array(trace_inIce)


        ### Variables ###
        averageRMS_PA[0] = sumRMS_PA / 4
        averageSNR_PA[0] = sumSNR_PA / 4
        averageKurtosis_PA[0] = sumKurtosis_PA / 4
        averageEntropy_PA[0] = sumEntropy_PA / 4
        averageImpulsivity_PA[0] = sumImpulsivity_PA / 4

        averageRMS_inIce[0] = sumRMS_inIce / 15
        averageSNR_inIce[0] = sumSNR_inIce / 15
        averageKurtosis_inIce[0] = sumKurtosis_inIce / 15
        averageEntropy_inIce[0] = sumEntropy_inIce / 15
        averageImpulsivity_inIce[0] = sumImpulsivity_inIce / 15

        averageRMS_surface[0] = sumRMS_surface / 9
        averageSNR_surface[0] = sumSNR_surface / 9
        averageKurtosis_surface[0] = sumKurtosis_surface / 9
        averageEntropy_surface[0] = sumEntropy_surface / 9
        averageImpulsivity_surface[0] = sumImpulsivity_surface / 9

        ### CSW variables ###
        refIndex_PA, refIndex_inIce, refIndex_surface = getReferenceTraceIndices(entropy)

        # Calculations: CSW & Impulsivity (PA channels)
        csw = trace_utilities.get_coherent_sum( np.delete(trace_PA, refIndex_PA, axis=0), trace_PA[refIndex_PA] )
        coherentRMS_PA[0] = trace_utilities.get_split_trace_noise_RMS(csw)
        coherentSNR_PA[0] = trace_utilities.get_signal_to_noise_ratio(csw, coherentRMS_PA[0], 100)
        coherentKurtosis_PA[0] = trace_utilities.get_kurtosis(csw)
        coherentEntropy_PA[0] = trace_utilities.get_entropy(csw)
        coherentImpulsivity_PA[0] = trace_utilities.get_impulsivity(csw)

        # Calculations: CSW & Impulsivity (in-ice channels)
        csw = trace_utilities.get_coherent_sum( np.delete(trace_inIce, refIndex_inIce, axis=0), trace_inIce[refIndex_inIce] )
        coherentRMS_inIce[0] = trace_utilities.get_split_trace_noise_RMS(csw)
        coherentSNR_inIce[0] = trace_utilities.get_signal_to_noise_ratio(csw, coherentRMS_inIce[0], 100)
        coherentKurtosis_inIce[0] = trace_utilities.get_kurtosis(csw)
        coherentEntropy_inIce[0] = trace_utilities.get_entropy(csw)
        coherentImpulsivity_inIce[0] = trace_utilities.get_impulsivity(csw)

        # Calculations: CSW & Impulsivity (surface channels)
        csw = trace_utilities.get_coherent_sum( np.delete(trace_surface, refIndex_surface, axis=0), trace_surface[refIndex_surface] )
        coherentRMS_surface[0] = trace_utilities.get_split_trace_noise_RMS(csw)
        coherentSNR_surface[0] = trace_utilities.get_signal_to_noise_ratio(csw, coherentRMS_surface[0], 100)
        coherentKurtosis_surface[0] = trace_utilities.get_kurtosis(csw)
        coherentEntropy_surface[0] = trace_utilities.get_entropy(csw)
        coherentImpulsivity_surface[0] = trace_utilities.get_impulsivity(csw)


        tree_out.Fill()

    tree_out.Write()

    file_in.Close()
    file_out.Close()
