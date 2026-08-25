# PiNNAcLe-FT
PiNNAcLe (https://github.com/Teoroo-CMC/PiNNAcLe) was originally developed for active learn-on-the-fly tasks in machine learning interatomic potential (MLIPs). PiNNAcLe-FT is an extension of PiNNAcLe for efficient foundational model distillation and fine-tuning (FT) with DFT labels. 

<img src="/PiNNAcLe-FT.png" alt="pinnnacle-ft" style="height: 121px; width:598px;"/>

# What’s New
+ Eliminate the need to construct a large initial DFT dataset and ensure stable dynamics of gen0 model via foundation model distillation.
+ Enable workflows to start from a pre-trained model (gen0) via atomic dress matching between foundation model and DFT labeller.

# Motivation
+ The original PiNNAcLe workflow requires a large initial DFT dataset to build PiNet2-P3 models in the first generation, i.e., gen0 model.
+ Constructing a large DFT dataset is a time-consuming and inefficient process.
+ Insufficient initial DFT data or limited conformational diversity often leads to unstable gen0 model, thereby slowing down both subsequent DFT labelling and model convergence in the PiNNAcLe.

# Approach
+ Construct a diverse dataset for target systems using low-cost foundation models (e.g., MACE-MP-0, https://github.com/ACEsuit/mace-foundations).
+ Pre-train PiNet2-P3 models on this dataset for the gen0 model, i.e. foundation model distillation (see https://doi.org/10.1016/j.electacta.2026.149136).
+ Initiate the PiNNAcLe-FT workflow to fine-tune the pre-trained PiNet2-P3 models with DFT labels and the matched atomic dress.
  - In each generation, a number of new snapshots are collected from the MD trajectory driven by the latest PiNet2-P3 models.
  - The collected snapshots are then labeled using the CP2K package at the predefined DFT level and added to the DFT dataset.
  - The atomic dresses of the PiNet2-P3 model are updated based on the new training set, followed by model fine-tuning.
+ The energy and force weights of outliers with _f_max_ values exceeding twice the tolerance threshold were set to zero to stabilize the fine-tuning process.
+ A _start_idx_ option was added to params.collect_flags to skip the several initial snapshots when sampling the trajectory, thereby avoiding potential data leakage.

# Installation on Alvis
+ Download the PiNNAcLe-FT repo
```
git clone https://github.com/Teoroo-CMC/PiNNAcLe-FT.git
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
cp PiNNAcLe-FT/pinn_modified/*.py {PINN_DIR}/models/
```
+ Install the tips package developed by Yunqi, and update the modifications
```
pip install git+https://github.com/yqshao-archive/tips.git
cp PiNNAcLe-FT/tips_modified/io/*.py {TIPS_DIR}/io/
cp PiNNAcLe-FT/tips_modified/cli/*.py {TIPS_DIR}/cli/
```

# Installation on Arrhenius

+ Install Nextflow by following the documentation https://www.nextflow.io/docs/latest/install.html

+ Prepare the GPU environment for installation
```
srun -A naiss2025-5-447-gpu -p gpu --gres=gpu:nvidia_gh200_120gb:1 -t 2:00:00 --mem=120G --pty bash
module load GPU/Miniforge/26.3.2-2-eb
```

+ Download the PiNNAcLe-FT repo
```
git clone https://github.com/Teoroo-CMC/PiNNAcLe-FT.git
```
+ Create the conda environment
```
cd PiNNAcLe-FT
conda env create -f environment_arrhenius.yml -p /nobackup/proj/disk/snic2022-5-322/shared/zhanyun/conda_envs/pinnacle
conda activate /nobackup/proj/disk/snic2022-5-322/shared/zhanyun/conda_envs/pinnacle
```
+ Build singularity image of cp2k-2023.2
```
cd docker
apptainer build cp2k2023_2.sif cp2k-v2023.def
```
+ Build the PiNN docker image
```
git clone https://github.com/Teoroo-CMC/PiNN.git
cp ../pinn_modified/*.py PiNN/pinn/models/
cp ../pinn_modified/Singularity-acle.gpu PiNN/
cp -r ../tips_modified PiNN
cd PiNN
mkdir apptainer-cache
mkdir apptainer-tmp
export APPTAINER_CACHEDIR=/nobackup/proj/disk/snic2022-5-322/shared/zhanyun/PiNNAcLe-FT/docker/PiNN/apptainer-cache
export APPTAINER_TMPDIR=/nobackup/proj/disk/snic2022-5-322/shared/zhanyun/PiNNAcLe-FT/docker/PiNN/apptainer-tmp
apptainer build pinn-gpu-acle.sif Singularity-acle.gpu
apptainer exec "pinn-gpu-acle.sif" pinn --help # check
```
+ Check if the tips package was updated correctly
```
apptainer exec "pinn-gpu-acle.sif" tips --help
apptainer exec pinn-gpu-acle.sif bash -c 'TIPS_DIR=$(python -c "import tips, os; print(os.path.dirname(tips.__file__))"); echo "tips: $TIPS_DIR"; cat "$TIPS_DIR/io/dataset.py"'
```

+ Press Ctrl+D to exit the GPU environment.

# Usage
+ Step 1: Run MD simulations using foundation models to construct a large and conformationally diverse dataset from the trajectories.
  - Conformational diversity can be enhanced by performing MD simulations at varying temperatures or concentrations.
+ Step 2: Pre-train PiNet2-P3 models on the constructed dataset.
  - For more usage details, please refer to the PiNN documentation (https://teoroo-cmc.github.io/PiNN/master/).
+ Step 3: Prepare input and configuration files for PiNNAcLe-FT.
  - Modify the CP2K input parameters in _input/cp2k/r2SCAN-sp.inp_ to match your target system.
  - Copy the system XYZ file to _input/geo_ and the pre-trained models to _input/models_.
  - Update the environment path, PYTHONPATH, cp2k2023_2.sif path, GPU configuration, and SLURM account in the _nextflow.config_ file.
+ Step 4: Adjust the PiNNAcLe-FT hyperparameters in _nextflow/acle-cp2k-from-user-model.nf_ based on your specific task.
  - The _params.change_edress_ hyperparameter controls whether the e_dress of PiNet2-P3 models is updated during fine-tuning.
  - If disk space is limited, uncomment **release_space** channel to remove some intermediate files.
  - For other hyperparameters, please refer to the original PiNNAcLe documentation: https://teoroo-cmc.github.io/PiNNAcLe/recipe/acle/
+ Step 5: Fine-tune the PiNet2-P3 model
```
cd PiNNAcLe-FT
nextflow run main.nf -profile alvis -bg > log.out                  # on Alvis
NXF_VER=23.10.1 nextflow run main.nf -profile arrhen -bg > log.out # on Arrhenius
```
