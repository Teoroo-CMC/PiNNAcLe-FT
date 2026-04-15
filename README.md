# PiNNAcLe-FT
PiNNAcLe (https://github.com/Teoroo-CMC/PiNNAcLe) was originaly developed for active learn-on-the-fly tasks in machine learning interatomic potential (MLIPs). PiNNAcLe-FT is an extension of PiNNAcLe for efficient foundational model distillation and fine-tuning (FT) with DFT labels. 

# What’s New
+ Eliminate the need to construct a large initial DFT dataset and ensure stable dynamics of gen0 model via foundation model distillation.
+ Enable workflows to start from a pre-trained model (gen0) via atomic dress matching between foundation model and DFT labeller.

# Motivation
+ The original PiNNAcLe workflow requires a large initial DFT dataset to build PiNet2-P3 models in the first generation, i.e., gen0 model.
+ Constructing a large DFT dataset is a time-consuming and inefficient process.
+ Insufficient initial DFT data or limited conformational diversity often leads to unstable gen0 model, thereby slowing down both subsequent DFT labelling and model convergence in the PiNNAcLe.

# Approach
+ Construct a diverse dataset for target systems using low-cost foundation models (e.g., MACE-MP-0, https://github.com/ACEsuit/mace-foundations).
+ Pre-train PiNet2-P3 models on this dataset for the gen0 model, i.e. foundation model distillation.
+ Initiate the PiNNAcLe-FT workflow to fine-tune the pre-trained PiNet2-P3 models with DFT labels and the matched atomic dress.
  - In each generation, numbers of new snapshots are collected from the MD trajectory driven by the latest PiNet2-P3 models.
  - The collected snapshots are then labeled using the CP2K package at the predefined DFT level and added to the DFT dataset.
  - The atomic dresses of the PiNet2-P3 models are updated based on the new training set, and training continues.

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
+ Install Nextflow by following the documentation https://www.nextflow.io/docs/latest/install.html
+ Build singularity image of cp2k-2023.2
```
cd docker
apptainer build cp2k2023_2.sif cp2k-v2023.def
```
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
+ Step 1: Run MD simulations using foundation models to construct a large and conformationally diverse dataset from the trajectories.
  - Conformational diversity can be enhanced by performing MD simulations at varying temperatures or concentrations.
+ Step 2: Pre-train PiNet2-P3 models on the constructed dataset.
  - For more usage details, please refer to the PiNN documentation (https://teoroo-cmc.github.io/PiNN/master/).
+ Step 3: Prepare input and configuration files for PiNNAcLe-FT.
  - Modify the CP2K input parameters in _input/cp2k/r2SCAN-sp.inp_ to match your target system.
  - Copy the system XYZ file to _input/geo_ and the pre-trained models to _input/models_.
  - Update the environment path, cp2k2023_2.sif path, GPU configuration, and SLURM account in the _nextflow.config_ file.
+ Step 4: Adjust the PiNNAcLe-FT hyperparameters in _nextflow/acle-cp2k-from-user-model.nf_ based on your specific task.
  - The _params.change_edress_ hyperparameter controls whether the e_dress of PiNet2-P3 models is updated during fine-tuning.
  - The _params.root_path_ hyperparameter defines the working directory for the **release_space** channel, which removes intermediate files to free disk space.
  - For other hyperparameters, please refer to the original PiNNAcLe documentation: https://teoroo-cmc.github.io/PiNNAcLe/recipe/acle/
+ Step 5: Fine-tune the PiNet2-P3 model
```
cd PiNNAcLe-FT
nextflow run main.nf -profile alvis -bg > log.out
```
