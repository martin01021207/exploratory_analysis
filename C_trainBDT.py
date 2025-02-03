import argparse
from ROOT import TMVA, TFile, TTree, TCut

parser = argparse.ArgumentParser(description='train_BDT')
parser.add_argument('dir_in', type=str, help="Input directory")
parser.add_argument('station', type=str, help="Station number")
args = parser.parse_args()

dir_in = args.dir_in
if not dir_in.endswith("/"):
    dir_in += "/"
station_number = args.station

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

outputFile = TFile.Open(f'factoryOutput_vars_s{station_number}.root', 'RECREATE')

factory = TMVA.Factory('TMVA_Classification', outputFile,
                       '!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=Classification')

dataLoader = TMVA.DataLoader(f'dataLoader_vars_s{station_number}')

inputFileName_train = f"vars_s{station_number}_train.root"
inputFile_train = TFile.Open(dir_in+inputFileName_train)
if inputFile_train is None:
    ROOT.Warning("TMVA_Classification", "Error opening input file %s (TRAINING) - exit", inputFileName_train.Data())

inputFileName_test = f"vars_s{station_number}_test.root"
inputFile_test = TFile.Open(dir_in+inputFileName_test)
if inputFile_test is None:
    ROOT.Warning("TMVA_Classification", "Error opening input file %s (TESTING) - exit", inputFileName_test.Data())

signalTree_train = inputFile_train.Get('vars_sig')
backgroundTree_train = inputFile_train.Get('vars_bkg')
nEventsSig_train = signalTree_train.GetEntries()
nEventsBkg_train = backgroundTree_train.GetEntries()

signalTree_test = inputFile_test.Get('vars_sig')
backgroundTree_test = inputFile_test.Get('vars_bkg')
nEventsSig_test = signalTree_test.GetEntries()
nEventsBkg_test = backgroundTree_test.GetEntries()

signalWeight = 1.0
backgroundWeight = 1.0

#mycuts = TCut("!TMath::IsNaN(averageKurtosis_surface)")
#mycutb = TCut("!TMath::IsNaN(coherentKurtosis_surface)")
mycuts = TCut("")
mycutb = TCut("")

dataLoader.AddTree(signalTree_train, "Signal", signalWeight, mycuts, TMVA.Types.kTraining)
dataLoader.AddTree(backgroundTree_train, "Background", backgroundWeight, mycutb, TMVA.Types.kTraining)
dataLoader.AddTree(signalTree_test, "Signal", signalWeight, mycuts, TMVA.Types.kTesting)
dataLoader.AddTree(backgroundTree_test, "Background", backgroundWeight, mycutb, TMVA.Types.kTesting)

dataLoader.AddVariable( "nCoincidentPairs_PA"   , 'I')
dataLoader.AddVariable( "nHighHits_PA"          , 'I')
dataLoader.AddVariable( "averageSNR_PA"         , 'F')
dataLoader.AddVariable( "averageKurtosis_PA"    , 'F')
dataLoader.AddVariable( "averageEntropy_PA"     , 'F')
dataLoader.AddVariable( "impulsivity_PA"        , 'F')
dataLoader.AddVariable( "coherentSNR_PA"        , 'F')
dataLoader.AddVariable( "coherentKurtosis_PA"   , 'F')
dataLoader.AddVariable( "coherentEntropy_PA"    , 'F')
dataLoader.AddVariable( "corrMax_PA"            , 'F')

dataLoader.AddVariable( "nCoincidentPairs_inIce", 'I')
dataLoader.AddVariable( "nHighHits_inIce"       , 'I')
dataLoader.AddVariable( "averageSNR_inIce"      , 'F')
dataLoader.AddVariable( "averageKurtosis_inIce" , 'F')
dataLoader.AddVariable( "averageEntropy_inIce"  , 'F')
dataLoader.AddVariable( "impulsivity_inIce"     , 'F')
dataLoader.AddVariable( "coherentSNR_inIce"     , 'F')
dataLoader.AddVariable( "coherentKurtosis_inIce", 'F')
dataLoader.AddVariable( "coherentEntropy_inIce" , 'F')
dataLoader.AddVariable( "corrMax_inIce"         , 'F')

#dataLoader.AddVariable( "averageSNR_surface"      , 'F')
#dataLoader.AddVariable( "averageKurtosis_surface" , 'F')
#dataLoader.AddVariable( "averageEntropy_surface"  , 'F')
#dataLoader.AddVariable( "impulsivity_surface"     , 'F')
#dataLoader.AddVariable( "coherentSNR_surface"     , 'F')
#dataLoader.AddVariable( "coherentKurtosis_surface", 'F')
#dataLoader.AddVariable( "coherentEntropy_surface" , 'F')
#dataLoader.AddVariable( "corrMax_surface"         , 'F')

# Spectators will not be trained or tested,
# but they will be in the final results,
# so you know what events are in the false positive group
dataLoader.AddSpectator( "station_number" )
dataLoader.AddSpectator( "run_number" )
dataLoader.AddSpectator( "event_number" )
dataLoader.AddSpectator( "sim_energy" )


####################
### Book methods ###
####################

### LD ###
factory.BookMethod(dataLoader, TMVA.Types.kLD, "LD",
                   'H:!V:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10')

### BDT ###
factory.BookMethod(dataLoader, TMVA.Types.kBDT, "BDTD",
                   '!H:!V:NTrees=400:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:VarTransform=Decorrelate')

### TMVA DNN ###
# General layout
layoutString = "Layout=TANH|128,TANH|128,TANH|128,LINEAR"
# Training strategies
trainingString1 = "LearningRate=1e-2,Momentum=0.9,ConvergenceSteps=20,BatchSize=100,TestRepetitions=1,WeightDecay=1e-4,Regularization=None,DropConfig=0.0+0.5+0.5+0.5"
trainingStrategyString = "TrainingStrategy="
trainingStrategyString += trainingString1 # + "|" + trainingString2 + "|" + trainingString3 # for concatenating more training strings
# General Options
dnnOptions = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM"
dnnOptions += ":"
dnnOptions += layoutString
dnnOptions += ":"
dnnOptions += trainingStrategyString
dnnOptions += ":Architecture=CPU"
dnnMethodName = "TMVA_DNN_CPU"
factory.BookMethod(dataLoader, TMVA.Types.kDL, dnnMethodName, dnnOptions)

### Train Methods
factory.TrainAllMethods()

### Test and Evaluate Methods
factory.TestAllMethods()
factory.EvaluateAllMethods()

### Plot ROC Curves
roc = factory.GetROCCurve(dataLoader)
roc.Draw()
roc.Print(f"rocCurves_vars_s{station_number}.pdf", "pdf")

# Close outputfile
outputFile.Close()
