#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <fstream>
//#include <nlohmann/json.hpp>

#include "TCanvas.h"
#include "TGraph.h"
#include "TLine.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#include "TMVA/PyMethodBase.h"

using namespace TMVA;
using json = nlohmann::json;


void stage3_testCNN(TString file_in, int station, TString dir_trainedData, TString dir_out)
{
  if (!dir_trainedData.EndsWith("/")) dir_trainedData += "/";
  if (!dir_out.EndsWith("/")) dir_out += "/";

  TString station_str = TString::Format("s%d", station);

  // Target signal efficiency
  float targetEff = 0.98;

  // Methods
  //TString method = "TMVA_DNN_CPU";
  //TString method = "TMVA_CNN_CPU";
  //TString method = "BDT";
  TString method = "PyTorch";

  // Out files
  TString jsonFileName = "falsePositiveEvents_images_" + station_str + ".json";
  TString targetFileName = "testTree_images_" + station_str + ".root";
  TString graphFileName = "testedResults_images_" + station_str + ".pdf";

  int N = 48;
  int ntot = N * N;

  // This loads the library
  TMVA::Tools::Instance();
  if (method == "PyTorch") TMVA::PyMethodBase::PyInitialize();

  std::cout << std::endl;
  std::cout << "==> Start CNN Testing" << std::endl;

  // Create the Reader object
  TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );

  // Create a set of variables and declare them to the reader
  // - the variable names MUST corresponds in name and type to those given in the weight file(s) used
  std::vector<float> vars(ntot);
  int station_number;
  int run_number;
  int event_number;
  float sim_energy;
  //float trigger_time_difference;

  for (unsigned int i = 0; i < vars.size(); i++)
  {
    std::string varName = "image";
    reader->AddVariable(varName.c_str(), &vars[i]);
  }

  reader->AddSpectator("station_number", &station_number);
  reader->AddSpectator("run_number", &run_number);
  reader->AddSpectator("event_number", &event_number);
  reader->AddSpectator("sim_energy", &sim_energy);
  //reader->AddSpectator("trigger_time_difference", &trigger_time_difference);

  // Book the MVA method
  TString prefix = "TMVA_CNN_Classification";
  TString methodName;
  if (method == "PyTorch") methodName = method;
  else methodName = method + " method";
  TString weightfile = dir_trainedData + prefix + TString("_") + method + TString(".weights.xml");
  reader->BookMVA( methodName, weightfile );

  // Input file
  TFile *input(0);
  input = TFile::Open( file_in );
  if (!input)
  {
    std::cout << "ERROR: could not open data file" << std::endl;
    exit(1);
  }
  std::cout << "--- TMVA_CNN_ClassificationApp    : Using input file: " << input->GetName() << std::endl;

  // Prepare the event tree
  std::vector<float> * userVars = nullptr;

  // Input data
  TTree* tree_S = (TTree*)input->Get("images_sig");
  tree_S->SetBranchAddress( "image", &userVars );
  tree_S->SetBranchAddress( "station_number", &station_number );
  tree_S->SetBranchAddress( "run_number", &run_number );
  tree_S->SetBranchAddress( "event_number", &event_number );
  tree_S->SetBranchAddress( "sim_energy", &sim_energy );
  //tree_S->SetBranchAddress( "trigger_time_difference", &trigger_time_difference );
  int nEvents_S = tree_S->GetEntries();
  std::cout << "--- SIGNAL: " << nEvents_S << " events" << std::endl;

  TTree* tree_B = (TTree*)input->Get("images_bkg");
  tree_B->SetBranchAddress( "image", &userVars );
  tree_B->SetBranchAddress( "station_number", &station_number );
  tree_B->SetBranchAddress( "run_number", &run_number );
  tree_B->SetBranchAddress( "event_number", &event_number );
  tree_B->SetBranchAddress( "sim_energy", &sim_energy );
  //tree_B->SetBranchAddress( "trigger_time_difference", &trigger_time_difference );
  int nEvents_B = tree_B->GetEntries();
  std::cout << "--- BACKGROUND: " << nEvents_B << " events" << std::endl;

  TFile *output = new TFile( dir_out+targetFileName, "RECREATE" );
  output->cd();

  // Book output histograms
  UInt_t nbin = 100;
  TH1F *hist_S(0);
  TH1F *hist_B(0);

  float xMin, xMax;
  if (method == "BDT")
  {
    xMin = -0.8;
    xMax = 0.8;
  }
  else
  {
    xMin = -0.1;
    xMax = 1.1;
  }

  TString histTitle = "TMVA response for classifier: " + method + TString::Format(" (S%d)", station);
  hist_S = new TH1F("hist_S", histTitle, nbin, xMin, xMax);
  hist_S->SetLineColor(kAzure+2);
  hist_S->SetLineWidth(3);
  hist_S->SetFillColorAlpha(kAzure-7, 0.7);
  hist_B = new TH1F("hist_B", "", nbin, xMin, xMax);
  hist_B->SetLineColor(kRed+1);
  hist_B->SetLineWidth(3);
  hist_B->SetFillColor(kRed+1);
  hist_B->SetFillStyle(3354);

  // Test tree (Output)
  float EvaluateMVA;
  TTree* testTree_S = new TTree("TestTree_S", "TestTree_S");
  testTree_S->SetDirectory(output);
  testTree_S->Branch( method, &EvaluateMVA, method+"/F" );
  testTree_S->Branch( "station_number", &station_number, "station_number/I" );
  testTree_S->Branch( "run_number", &run_number, "run_number/I" );
  testTree_S->Branch( "event_number", &event_number, "event_number/I" );
  testTree_S->Branch( "sim_energy", &sim_energy, "sim_energy/F" );
  //testTree_S->Branch( "trigger_time_difference", &trigger_time_difference, "trigger_time_difference/F" );
  testTree_S->SetDirectory(output);

  TTree* testTree_B = new TTree("TestTree_B", "TestTree_B");
  testTree_B->SetDirectory(output);
  testTree_B->Branch( method, &EvaluateMVA, method+"/F" );
  testTree_B->Branch( "station_number", &station_number, "station_number/I" );
  testTree_B->Branch( "run_number", &run_number, "run_number/I" );
  testTree_B->Branch( "event_number", &event_number, "event_number/I" );
  testTree_B->Branch( "sim_energy", &sim_energy, "sim_energy/F" );
  //testTree_B->Branch( "trigger_time_difference", &trigger_time_difference, "trigger_time_difference/F" );
  testTree_B->SetDirectory(output);

  TStopwatch sw;
  sw.Start();
  for (Long64_t ievt = 0; ievt < nEvents_S; ievt++)
  {
    tree_S->GetEntry(ievt);
    std::copy(userVars->begin(), userVars->end(), vars.begin());
    EvaluateMVA = reader->EvaluateMVA(methodName);
    testTree_S->Fill();
    hist_S->Fill( EvaluateMVA );
  }
  // Get elapsed time
  sw.Stop();
  std::cout << "--- End of event loop (SIGNAL): ";
  sw.Print();

  sw.Start();
  for (Long64_t ievt = 0; ievt < nEvents_B; ievt++)
  {
    tree_B->GetEntry(ievt);
    std::copy(userVars->begin(), userVars->end(), vars.begin());
    EvaluateMVA = reader->EvaluateMVA(methodName);
    testTree_B->Fill();
    hist_B->Fill( EvaluateMVA );
  }
  // Get elapsed time
  sw.Stop();
  std::cout << "--- End of event loop (BACKGROUND): ";
  sw.Print();

  TGraph *graph = new TGraph();

  vector<double> cutValues;
  float cutStart;
  if (method == "BDT") cutStart = -1.0;
  else cutStart = 0.0;
  for (double cut = cutStart; cut <= 0.95; cut += 0.01) cutValues.push_back(cut);
  for (double cut = 0.95; cut <= 1.0; cut += 0.001) cutValues.push_back(cut);

  int nCounts_S = testTree_S->GetEntries();
  int nCounts_B = testTree_B->GetEntries();

  float minDiff = 1.;
  float cut_selected;
  float count_B_selected;
  float eff_selected;
  float rej_selected;
  int nEvents_FP;
  int i;
  json info_FP;
  std::vector<int> bkgRuns;
  std::vector<int> bkgEvents;
  TString threshold;
  for ( double cut : cutValues )
  {
    threshold = method + TString::Format(" > %f", cut);

    double count_S = testTree_S->GetEntries(threshold);
    double count_B = testTree_B->GetEntries(threshold);

    double eff = count_S / nCounts_S;
    double rej = 1 - count_B / nCounts_B;

    graph->SetPoint(graph->GetN(), eff, rej);

    //printf(" Cut %f   Eff %f   Rej %f\n", cut, eff, rej);

    double effDiff = std::fabs(eff - targetEff);
    if (effDiff < minDiff)
    {
      minDiff = effDiff;
      cut_selected = cut;
      eff_selected = eff;
      rej_selected = rej;
    }
  }

  printf("*** Signal Efficiency: %f\n", eff_selected);
  printf("*** Background Rejection: %f\n", rej_selected);
  nEvents_FP = 0;
  i = 0;
  info_FP.clear();
  bkgRuns.clear();
  bkgEvents.clear();
  threshold = method + TString::Format(" > %f", cut_selected);
  count_B_selected = testTree_B->GetEntries(threshold);
  while (nEvents_FP < count_B_selected)
  {
    testTree_B->GetEntry(i);
    if (EvaluateMVA > cut_selected)
    {
      bkgRuns.push_back(run_number);
      bkgEvents.push_back(event_number);
      //TString run_str = TString::Format("%d", run_number);
      std::string run_str = std::to_string(run_number);
      info_FP[run_str] = json::array();
      nEvents_FP += 1;
    }
    i += 1;
  }
  printf("*** Number of False Positive Events: %d\n", nEvents_FP);

  int j = 0;
  for ( int run : bkgRuns )
  {
    std::string run_str = std::to_string(run);
    info_FP[run_str].push_back(bkgEvents[j]);
    j += 1;
  }

  std::ofstream jsonFile(dir_out+jsonFileName);
  jsonFile << info_FP.dump(4); // 4 is for pretty-printing with indentation
  jsonFile.close();

  TCanvas canvas = TCanvas("c1", histTitle, 10, 10, 850, 500);
  gStyle->SetOptStat(0);
  gPad->SetLogy(1);

  canvas.cd();
  hist_S->Draw();
  hist_B->Draw("same");

  float leg_xMin;
  float leg_xMax;
  float leg_yMin;
  float leg_yMax;

  if (method == "BDT")
  {
    if (nEvents_S > nEvents_B)
    {
      leg_xMin = 0.1;
      leg_xMax = 0.3;
    }
    else
    {
      leg_xMin = 0.7;
      leg_xMax = 0.9;
    }
  }
  else
  {
    leg_xMin = 0.4;
    leg_xMax = 0.6;
  }

  leg_yMin = 0.75;
  leg_yMax = 0.9;

  TLegend *leg_hist = new TLegend(leg_xMin, leg_yMin, leg_xMax, leg_yMax);
  leg_hist->AddEntry(hist_S, "Signal", "f");
  leg_hist->AddEntry(hist_B, "Background", "f");
  leg_hist->Draw();

  canvas.Print(dir_out+graphFileName+"(", "pdf");
  canvas.Clear("D");

  // Set the grid for the graph background
  TPad *grid = new TPad("grid", "", 0, 0, 1, 1);
  grid->Draw();
  grid->cd();
  grid->SetGrid();

  gPad->SetLeftMargin(0.15);
  TString graphTitle = TString::Format("Signal efficiency vs. Background rejection (S%d)", station);
  graph->SetTitle(graphTitle);
  graph->GetXaxis()->SetTitle("Signal efficiency (Sensitivity)");
  graph->GetYaxis()->SetTitle("Background rejection (Specificity)");
  graph->SetLineWidth(2);
  graph->SetLineColor(4);
  graph->GetXaxis()->SetRangeUser(0, 1.01);
  graph->GetYaxis()->SetRangeUser(0, 1.01);
  graph->Draw("AL");

  TLegend *leg = new TLegend(0.2, 0.15, 0.35, 0.3);
  leg->SetHeader("MVA Method", "");
  leg->AddEntry(graph, method, "l");
  leg->Draw();

  canvas.Print(dir_out+graphFileName, "pdf");
  canvas.Clear("D");

  float gr_xMin = 0.92;
  float gr_xMax = 1.0015;
  float gr_yMin = 0.9;
  float gr_yMax = 1.001;

  graph->GetXaxis()->SetRangeUser(gr_xMin, gr_xMax);
  graph->GetYaxis()->SetRangeUser(gr_yMin, gr_yMax);
  graph->Draw("AL");

  leg->Draw();

  TLine *vLine = new TLine(targetEff, gr_yMin, targetEff, rej_selected);
  vLine->SetLineStyle(2);
  vLine->SetLineWidth(2);
  vLine->SetLineColor(6);
  vLine->Draw("same");

  canvas.Print(dir_out+graphFileName+")", "pdf");
  canvas.Clear("D");

  output->cd();
  graph->Write();
  testTree_S->Write();
  hist_S->Write();
  testTree_B->Write();
  hist_B->Write();

  output->Close();
  input->Close();
  delete reader;
  std::cout << "==> CNN Testing is done!" << std::endl << std::endl;
}
