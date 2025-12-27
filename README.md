# Exploratory Analysis
This workflow serves as an exploratory path of event filtering and event selection in the deep CR search and potentially in the future neutrino search.


# 1. Event Filtering & Event Selection
A three-staged event filtering approach is taken in this analysis.
## (1) Working On 10% Burn Sample Data
### Stage 1: Hit Filter
The hit filter rejects **thermal events** with background rejections greater than 90%; and it allows more than 95% of NuRadioMC simualted neutrino events pass the filter, which shows also high signal efficiencies (neutrino simulations can be replaced with CR simulations when ready). Survived events go to the next stage.
### Classifiers' Training (BDT & CNN)
Events that passed the hit filter (real data & sim data) are split into two sets: **TRAIN** & **TEST**. Several variables are calculated and images are made of all these events, a BDT classifier will be trained on the variables and a CNN will be trained on the images, in parallel.
### Stage 2: BDT Classification (variables)
The trained BDT classifier is then used on classifying the test set real data (background) and simulated data (signal). A very high signal efficiency will be targeted, which yields a lower background rejection, to select background events that were classified as signals (false positive events).
### Stage 3: CNN Classification (images)
Real (background) event images are picked out of the image test set according to the false positive events from stage 2; along with all the simulated (signal) event images are picked out, then both will be passed into the already trained CNN to further suppress background and to select candidate events.
## (2) Working On Full Data
Without re-training the classifiers, all data will go through stage 1, stage 2, and stage 3 as shown above.


# 2. Simulations
## (1) NuRadioMC Simulated Neutrinos
### [Simulated Station Data Sets](https://radio.uchicago.edu/wiki/index.php/Simulations)
## (2) Deep CR simulations
### (In Process)


# 3. Software Prerequisites
### ROOT, Mattak, NuRaioMC, PyTorch


# 4. Code Execution
You will see two flowcharts, **burn sample workflow** and **full data workflow**, when you scroll all the way down. There are many scripts, hopefully it will help understand the workflows with the flowcharts.
## FILE 1:  stage1_applyHitFilter.py  
To use the Hit Filter, we need to import the module `stationHitFilter.py` from NuRadioReco.
### (1) Real Data Input .root Files
**Case 1: Burn sample 10% data (with a JSON file)**
```
python stage1_applyHitFilter.py /PATH/TO/INPUT/DIR/ /PATH/TO/OUTPUT/DIR/ <station> --run <run> --json_select /PATH/TO/BURN_SAMPLE.json
```
**Case 2: Full data (burn data excluded)**
```
python stage1_applyHitFilter.py /PATH/TO/INPUT/DIR/ /PATH/TO/OUTPUT/DIR/ <station> --run <run> --json_select /PATH/TO/BURN_SAMPLE.json --isExcluded
```
### (2) Simulated Data Input .nur Files
```
python stage1_applyHitFilter.py /PATH/TO/INPUT/DIR/ /PATH/TO/OUTPUT/DIR/ <station> --isSim --sim_E <energy>
```
(energy can be  16.00  16.50  17.00  ...)
### (3) Output
The output file is a ROOT file containing C++ vectors saved in a tree, each event (passed HF) has a vector containing 24 TGraphs (channel waveforms).

## FILE 2:  combineROOTfiles.sh
This script combines multiple ROOT files into one ROOT file.  
**Case 1: Combine real data only or sim data only**  
types: filtered, filtered_sim, vars, vars_sim, images, images_sim
```
sh combineROOTfiles.sh -i /PATH/TO/INPUT/DIR/ -s <station> -t <type> -o /PATH/TO/OUTPUT/DIR/
```
**Case 2: Combine real data files and sim data files into one ROOT file**  
types: filtered, vars, images
```
sh combineROOTfiles.sh -i /PATH/TO/INPUT/DIR/ -s <station> -t <type> -o /PATH/TO/OUTPUT/DIR/ -m /PATH/TO/SIM/DIR/
```

## FILE 3:  splitEventData.py
This script splits events of one ROOT file into two ROOT files: **TRAIN** & **TEST**
```
python splitEventData.py /PATH/TO/INPUT/ROOT/FILTERED/DATA/file.root /PATH/TO/OUTPUT/DIR/ <integer_division>
```

## FILE 4:  A_makeVariables.py
To make variables, file `MakeVariables.py` needs to be imported.
```
python A_makeVariables.py /PATH/TO/INPUT/ROOT/FILTERED/DATA/file.root /PATH/TO/OUTPUT/DIR/
```

## FILE 5:  B_makeImages.py  
To make images, file `MakeImages.py` needs to be imported.  
**Case 1: Burn sample 10% data**
```
python B_makeImages.py /PATH/TO/INPUT/ROOT/FILTERED/DATA/file.root /PATH/TO/OUTPUT/DIR/
```
**Case 2: Full data (after BDT, for stage 3 classification CNN)**
```
python B_makeImages.py /PATH/TO/INPUT/ROOT/FILTERED/DATA/file.root /PATH/TO/OUTPUT/DIR/ --json_FP /PATH/TO/FALSE_POSITIVES.json
```

## FILE 6:  C_trainBDT.py
Make sure you have these two files in the input directory:  
**vars_s{station}_train.root** and **vars_s{station}_test.root**
```
python C_trainBDT.py /PATH/TO/INPUT/DIR/ <station>
```

## FILE 7:  D_trainCNN.py
Make sure you have these two files in the input directory:  
**images_s{station}_train.root** and **images_s{station}_test.root**
### (PyTorch Needed)
First execute FILE 8: `PyTorch_Generate_CNN_Model.py`
```
python D_trainCNN.py /PATH/TO/INPUT/DIR/ <station>
```

## FILE 8:  PyTorch_Generate_CNN_Model.py
```
python PyTorch_Generate_CNN_Model.py
```

## FILE 9:  stage2_testBDT.py
```
python stage2_testBDT.py <station> /PATH/TO/INPUT/TEST/file.root /PATH/TO/SIM/sim_file.root /PATH/TO/TMVA/TRAINED/weights/ /PATH/TO/OUTPUT/DIR/
```

## FILE 10:  collectStage3Images.py
In the input directory, you should have this file: **images_s{station}_test.root**
```
python collectStage3Images.py /PATH/TO/INPUT/DIR/ <station> /PATH/TO/FALSE_POSITIVES.json /PATH/TO/OUTPUT/DIR/ --clean_mode
```

## FILE 11:  stage3_testCNN.C
Unfortunately, this script has to be in C++ because there's one function missing in `TMVA::Reader` causing the technical difficulty to test the CNN in Python with images. Open the ROOT interface:
```
root -l
```
Then we can execute the script easily in the ROOT interface:
```
.x stage3_testCNN.C(station, "/PATH/TO/INPUT/TEST/file.root", "/PATH/TO/SIM/sim_file.root", "/PATH/TO/TMVA/TRAINED/weights/", "/PATH/TO/OUTPUT/DIR/")
```

## FLOWCHART 1:  
![image](https://github.com/user-attachments/assets/25a88045-5092-4c45-b5d6-c8596a1baa83)


## FLOWCHART 2:  
![image](https://github.com/user-attachments/assets/efda53e3-21b0-44e2-8dec-2dc65ea794d8)
