# PiNNAcLe-FT
This is an update to the PiNNAcLe repository (https://github.com/Teoroo-CMC/PiNNAcLe) introducing support for fine-tuning (FT) of pre-trained models at the DFT level.

# What’s New
+ Eliminate the need to construct a large initial DFT dataset.
+ Enable workflows to start from a pre-trained model rather than from scratch.

# Motivation
+ The original PiNNAcLe workflow starts from the scratch, requiring a large initial DFT dataset to build PiNet2-P3 models in the first generation (i.e., gen0).
+ However, constructing a large DFT dataset is a time-consuming and inefficient process.
+ Insufficient initial DFT data or limited conformational diversity can lead to unstable PiNet2-driven MD simulations, thereby slowing down both DFT labeling and model convergence.

# Approach
+ Construct a large and conformationally diverse dataset for target systems using low-cost, state-of-the-art foundation models (e.g., https://github.com/acesuit/mace).
+ Pre-train PiNet2-P3 models on the constructed dataset (i.e., model distillation), and ensure stable MD simulations can be achieved by using them.
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
