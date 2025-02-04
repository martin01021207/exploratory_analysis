# Exploratory Analysis
This workflow serves as an exploratory path of event filtering and event selection in the deep CR search and potentially in the future neutrino search.

# Event Filtering & Event Selection
A 3-staged event filtering approach is taken in this analysis.
## Working On 10% Data (Burn Sample)
### Stage 1: Hit Filter
The hit filter rejects **thermal events** with background rejections greater than 90%; and it allows more than 95% of NuRadioMC simualted neutrino events pass the filter, which shows also high signal efficiencies (neutrino simulations can be replaced with CR simulations when ready). Survived events go to the next stage.
### Classifiers' Training (BDT & CNN)
Events that passed the hit filter (real data & sim data) are split into two sets: **TRAIN** & **TEST**. Several variables are calculated and images are made of all these events, a BDT classifier will be trained on the variables and a CNN will be trained on the images, in parallel.
### Stage 2: BDT Classification (variables)
The trained BDT classifier is then used on classifying the test set real data (background) and simulated data (signal). A very high signal efficiency will be targeted, which yields a lower background rejection, to select background events that were classified as signals (false positive events).
### Stage 3: CNN Classification (images)
Real (background) event images are picked out of the image test set according to the false positive events from stage 2; along with all the simulated (signal) event images are picked out, then both will be passed into the already trained CNN to further suppress background and to select candidate events.
## Working On Full Data
Without re-training the classifiers, all data will go through stage 1, stage 2, and stage 3 as shown above.

# Simulations
## NuRadioMC Simulated Neutrinos
### [Simulated Station Data Sets](https://radio.uchicago.edu/wiki/index.php/Simulations)
## Deep CR simulations
### (In Process)


# Software Prerequisites
### ROOT, Mattak, NuRaioMC, PyTorch


# Code Execution

## stage1_applyHitFilter.py
### Real Data Input (.root Files)
**Case 1: Burn sample 10% data (with a JSON file)**
```
python stage1_applyHitFilter.py /PATH/TO/INPUT/DIR/ /PATH/TO/OUTPUT/DIR/ <station> --run <run> --json_select </PATH/TO/BURN_SAMPLE.json>
```
**Case 2: Full data (burn data excluded)**
```
python stage1_applyHitFilter.py /PATH/TO/INPUT/DIR/ /PATH/TO/OUTPUT/DIR/ <station> --run <run> --json_select </PATH/TO/BURN_SAMPLE.json> --isExcluded
```
### Simulated Data Input (.nur Files)
```
python stage1_applyHitFilter.py /PATH/TO/INPUT/DIR/ /PATH/TO/OUTPUT/DIR/ <station> --isSim --sim_E <energy>
```
(<energy> can be 16.00 16.50 17.00 etc)
### Outputs
An output file is a ROOT file containing C++ vectors saved in a tree, each event (passed HF) has a vector containing 24 TGraphs (channel waveforms).

## A_makeVariables.py
```
python A_makeVariables.py /PATH/TO/INPUT/ROOT/FILTERED/DATA/file.root /PATH/TO/OUTPUT/DIR/
```

## B_makeImages.py
**Case 1: Burn sample 10% data**
```
python B_makeImages.py /PATH/TO/INPUT/ROOT/FILTERED/DATA/file.root /PATH/TO/OUTPUT/DIR/
```
**Case 2: Full data (after BDT, for stage 3 classification CNN)**
```
python B_makeImages.py /PATH/TO/INPUT/ROOT/FILTERED/DATA/file.root /PATH/TO/OUTPUT/DIR/ --json_FP </PATH/TO/FALSE_POSITIVES.json>
```

## C_trainBDT.py
Make sure you have two files in the input directory:
**vars_s{station}_train.root** and **vars_s{station}_test.root**
```
python C_trainBDT.py /PATH/TO/INPUT/DIR/ <station>
```

## D_trainCNN.py
### (PyTorch Needed)
Make sure you have two files in the input directory:
**images_s{station}_train.root** and **images_s{station}_test.root**
First execute `PyTorch_Generate_CNN_Model.py`
```
python PyTorch_Generate_CNN_Model.py
```
```
python D_trainCNN.py /PATH/TO/INPUT/DIR/ <station>
```
