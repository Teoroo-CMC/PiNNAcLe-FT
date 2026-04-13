# PiNNAcLe-FT
This is an update to the PiNNAcLe repository (https://github.com/Teoroo-CMC/PiNNAcLe) introducing support for fine-tuning (FT) of pre-trained models at the DFT level.

# What’s New
+ Eliminate the need to construct a large initial DFT dataset.
+ Enable workflows to start from a pre-trained model rather than from scratch.
+ Automatically switch the models from the pre-training level (e.g., foundation model level) to the fine-tuning level (i.e., DFT level).

# Motivation
+ The original PiNNAcLe workflow starts from the scratch, requiring a large initial DFT dataset to build PiNet2-P3 models in the first generation (i.e., gen0).
+ Constructing a large DFT dataset is a time-consuming and inefficient process.
+ If the initial DFT dataset is insufficient or lacks conformational diversity, unstable PiNet2-driven MD simulations in gen0 can slow down both DFT labeling and training convergence.

# Approach
+ Construct a large and conformationally diverse dataset for target systems using low-cost, state-of-the-art foundation models (e.g., https://github.com/acesuit/mace).
+ Pre-train PiNet2-P3 models on the constructed dataset, and ensure stable MD simulations can be achieved by using them.
+ Initiate the PiNNAcLe-FT workflow to fine-tune the pre-trained PiNet2-P3 models at the target DFT level.

# Installation
+ Download the PiNNAcLe-FT repo
```
git clone https://github.com/zzy2014/PiNNAcLe-FT.git
```
+ Create the conda environment
```
cd PiNNAcLe-FT
conda env create -f environment.yml
conda activate pinnacle
```
+ Install Nextflow by following https://www.nextflow.io/docs/latest/install.html
+ Build singularity image of cp2k-2023.2
```
cd docker
apptainer build cp2k2023_2.sif cp2k-v2023.def
```
Then, change the full path of cp2k2023_2.sif in nextflow.config file
+ Install the PiNN package
```
cd ..
pip install git+https://github.com/Teoroo-CMC/PiNN.git --no-deps
```
+ Install the tips package developed by Yunqi, and update the modifiations
```
pip install git+https://github.com/yqshao-archive/tips.git
cp PiNNAcLe-FT/io/*.py {TIPS_DIR}/io/
```

# Usage
+ Step 1: Run MD simulations by foundation models, and construct a large and conformationally diverse dataset from the trajctories.
 - To improve the conformation deversity, MD simulations can be run at different temperatures/concentrations
+ Step 2: Pre-train the PiNet2-P3 models on the above dataset
 - Please find the details in the PiNN documentation (https://teoroo-cmc.github.io/PiNN/master/)
+ Step 3: Prepare the input files for PiNNAcLe-FT
  - Modify the CP2K input parameters in _input/cp2k/r2SCAN-sp.inp_ according to your system.
  - Copy the xyz file of your system into _input/geo_ sub-folder, and copy the pre-trained models into _input/models_ sub-folder.
+ Step 4: Change the hyper-parameters of PiNNAcLe-FT in _nextflow/acle-cp2k-from-user-model.nf_ according to your task.
  - The params.change_edress hyper-parameter determines if the e_dress of PiNet2-P3 models be updated during the fine-tuning
  - The params.root_path hyper-parameter is the current working path for the release_space chnnel, which will delete intermediate files to release the disk space.
  - For other hyper-parameters, please find the details in the PiNNAcLe documentation https://teoroo-cmc.github.io/PiNNAcLe/recipe/acle/
+ Step 5: Fine-tune the PiNet2-P3 model
```
cd PiNNAcLe-FT
nextflow run main.nf -profile alvisacle -bg > log.out
```
