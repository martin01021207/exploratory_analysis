import argparse
import ROOT

parser = argparse.ArgumentParser(description='train_CNN')
parser.add_argument('dir_in', type=str, help="Input directory")
parser.add_argument('station', type=str, help="Station number")
parser.add_argument('--path_to_PyTorch_CNN_model', type=str, default=None, help="Assign a path to PyTorch CNN model")
args = parser.parse_args()

dir_in = args.dir_in
if not dir_in.endswith("/"):
    dir_in += "/"
station_number = args.station
pyTorchFileName = args.path_to_PyTorch_CNN_model
if not pyTorchFileName:
    pyTorchFileName = "PyTorch_Generate_CNN_Model.py"

N = 32

#switch off MT in OpenMP (BLAS)
ROOT.gSystem.Setenv("OMP_NUM_THREADS", "1")

TMVA = ROOT.TMVA
TFile = ROOT.TFile

import os
import importlib

TMVA.Tools.Instance()

opt = [1, 1, 1, 1]
useTMVACNN = opt[0] if len(opt) > 0 else False
useTMVADNN = opt[1] if len(opt) > 1 else False
useTMVABDT = opt[2] if len(opt) > 2 else False
usePyTorchCNN = opt[3] if len(opt) > 3 else False

hasCPU = ROOT.gSystem.GetFromPipe("root-config --has-tmva-cpu") == "yes"
if not hasCPU:
    ROOT.Warning("TMVA_CNN_Classificaton", "TMVA is not build with CPU support. Cannot use TMVA Deep Learning for CNN")
    useTMVACNN = False
    useTMVADNN = False

if not useTMVACNN:
    ROOT.Warning("TMVA_CNN_Classificaton", "TMVA is not build with CPU support. Cannot use TMVA Deep Learning for CNN")

if ROOT.gSystem.GetFromPipe("root-config --has-tmva-pymva") != "yes":
    usePyTorchCNN = False
else:
    TMVA.PyMethodBase.PyInitialize()

torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    usePyTorchCNN = False
    ROOT.Warning("TMVA_CNN_Classificaton", "Skip using PyTorch since torch is not installed")

# use default threads
num_threads = 4
# do enable MT running
if num_threads >= 0:
    ROOT.EnableImplicitMT(num_threads)
print("Running with nthreads  = ", ROOT.GetThreadPoolSize())

writeOutputFile = True
outputFile = None
factoryFileName = f"factoryOutput_images_s{station_number}.root"
if writeOutputFile:
    outputFile = TFile.Open(factoryFileName, "RECREATE")


factory = TMVA.Factory("TMVA_CNN_Classification", outputFile,
    "!V:ROC:!Silent:Color:AnalysisType=Classification:Transformations=None:!Correlations")

dataLoader = TMVA.DataLoader(f"dataLoader_images_s{station_number}")

imgSize = N * N

inputFileName_train = f"images_s{station_number}_train.root"
inputFile_train = TFile.Open(dir_in+inputFileName_train)
if inputFile_train is None:
    ROOT.Warning("TMVA_CNN_Classification", "Error opening input file %s (TRAINING) - exit", inputFileName_train.Data())

inputFileName_test = f"images_s{station_number}_test.root"
inputFile_test = TFile.Open(dir_in+inputFileName_test)
if inputFile_test is None:
    ROOT.Warning("TMVA_CNN_Classification", "Error opening input file %s (TESTING) - exit", inputFileName_test.Data())

signalTree_train = inputFile_train.Get("images_sig")
backgroundTree_train = inputFile_train.Get("images_bkg")
nEventsSig_train = signalTree_train.GetEntries()
nEventsBkg_train = backgroundTree_train.GetEntries()

signalTree_test = inputFile_test.Get("images_sig")
backgroundTree_test = inputFile_test.Get("images_bkg")
nEventsSig_test = signalTree_test.GetEntries()
nEventsBkg_test = backgroundTree_test.GetEntries()

signalWeight = 1.0
backgroundWeight = 1.0

mycuts = ""  # for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1"
mycutb = ""  # for example: TCut mycutb = "abs(var1)<0.5"

dataLoader.AddTree(signalTree_train, "Signal", signalWeight, mycuts, TMVA.Types.kTraining)
dataLoader.AddTree(backgroundTree_train, "Background", backgroundWeight, mycutb, TMVA.Types.kTraining)
dataLoader.AddTree(signalTree_test, "Signal", signalWeight, mycuts, TMVA.Types.kTesting)
dataLoader.AddTree(backgroundTree_test, "Background", backgroundWeight, mycutb, TMVA.Types.kTesting)

dataLoader.AddVariablesArray( "image", imgSize )
dataLoader.AddSpectator( "station_number" )
dataLoader.AddSpectator( "run_number" )
dataLoader.AddSpectator( "event_number" )
dataLoader.AddSpectator( "sim_energy" )
#dataLoader.AddSpectator( "trigger_time_difference" )


####################
### Book methods ###
####################

### TMVA BDT ###
if useTMVABDT:
    factory.BookMethod(dataLoader, TMVA.Types.kBDT, "BDT",
        "!V:NTrees=400:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20")


### TMVA DNN ###
if useTMVADNN:
    layoutString = "Layout=DENSE|100|RELU,BNORM,DENSE|100|RELU,BNORM,DENSE|100|RELU,BNORM,DENSE|100|RELU,DENSE|1|LINEAR"

    trainingString1 = "LearningRate=1e-3,Momentum=0.9,Repetitions=1,ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,MaxEpochs=20,WeightDecay=1e-4,Regularization=None,Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0." # + "|" + trainingString2 + ...
    trainingStrategyString = "TrainingStrategy="
    trainingStrategyString += trainingString1
    dnnOptions = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:WeightInitialization=XAVIER"
    dnnOptions += ":"
    dnnOptions += layoutString
    dnnOptions += ":"
    dnnOptions += trainingStrategyString
    dnnOptions += ":Architecture=CPU"
    dnnMethodName = "TMVA_DNN_CPU"
    factory.BookMethod(dataLoader, TMVA.Types.kDL, dnnMethodName, dnnOptions)


### TMVA CNN ###
if useTMVACNN:
    inputLayoutString = f"InputLayout=1|{N}|{N}"
    layoutString = "Layout=CONV|10|3|3|1|1|1|1|RELU,BNORM,CONV|10|3|3|1|1|1|1|RELU,MAXPOOL|2|2|2|2,CONV|10|3|3|1|1|1|1|RELU,MAXPOOL|2|2|1|1,RESHAPE|FLAT,DENSE|225|RELU,DENSE|1|LINEAR"
    trainingString1 = "LearningRate=1e-3,Momentum=0.9,Repetitions=1,ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,MaxEpochs=20,WeightDecay=1e-4,Regularization=None,Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0.0" # + "|" + trainingString2 + ...
    trainingStrategyString = "TrainingStrategy="
    trainingStrategyString += trainingString1
    cnnOptions = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:WeightInitialization=XAVIER"
    cnnOptions += ":"
    cnnOptions += inputLayoutString
    cnnOptions += ":"
    cnnOptions += layoutString
    cnnOptions += ":"
    cnnOptions += trainingStrategyString
    cnnOptions += ":Architecture=CPU"
    cnnMethodName = "TMVA_CNN_CPU"
    factory.BookMethod(dataLoader, TMVA.Types.kDL, cnnMethodName, cnnOptions)


### PyTorch CNN ###
if usePyTorchCNN:
    ROOT.Info("TMVA_CNN_Classification", "Using Convolutional PyTorch Model")
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None and os.path.exists(pyTorchFileName):
        ROOT.Info("TMVA_CNN_Classification", "Booking PyTorch CNN model")
        methodOpt = f"H:!V:VarTransform=None:FilenameModel=PyTorchModelCNN.pt:FilenameTrainedModel=PyTorchTrainedModelCNN_s{station_number}.pt:NumEpochs=20:BatchSize=100"
        methodOpt += ":UserCode=" + pyTorchFileName
        factory.BookMethod(dataLoader, TMVA.Types.kPyTorch, "PyTorch", methodOpt)
    else:
        ROOT.Warning("TMVA_CNN_Classification",
            "PyTorch is not installed or model building file is not existing - skip using PyTorch")


### Train Methods
factory.TrainAllMethods()

### Test and Evaluate Methods
factory.TestAllMethods()
factory.EvaluateAllMethods()

### Plot ROC Curve
roc = factory.GetROCCurve(dataLoader)
roc.Draw()
roc.Print(f"rocCurves_images_s{station_number}.pdf", "pdf")

# Close outputfile
outputFile.Close()
