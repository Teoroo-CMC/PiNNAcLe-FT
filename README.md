# PiNNAcLe-Proton-aq
This is the repository for the datasets, the computational workflows and outputs associate with the preprint "Reactive and Polarizable Molecular Dynamics of the Excess Proton in Water with Machine Learning Interatomic Potentials"

# Project structure
In the following project structure, _*_ denotes the random seed, _<...>_ indicates an arbitrary sequence of characters.
```text
PiNNAcLe-Proton-aq/
├── 1_Datasets/                              # Directory: the datasets built from the FM trajectories
│   ├── geo/                                 # The input structures of liquid water with an excess proton
│   │   ├── wat64_h3o+.vasp                  # In the vasp format
│   │   └── wat64_h3o+.xyz                   # In the ASE extxyz format                                       
│   ├── MACE-r2scan-<...>/                   # The NVT FM trajectories at the temerature of <...>
│   │   ├── NVT-<...>-1000ps-wat64_h3o+.log  # The placeholder for the ASE trajectory log file
│   │   └── NVT-<...>-1000ps-wat64_h3o+.traj # The placeholder for the ASE trajectory file
│   ├── mix_dataset/                         # The final dataset in the Tensorflow dataset format
│   │   ├── mixProton_330K_400K_500K.yml     # The configuration file of the dataset
│   │   └── mixProton_330K_400K_500K.tfr     # The record file of the dataset   
│   ├── nvt_md_mace_r2scan.nf                # The Nextflow script for MD driven by the mace-matpes-r2scan-0
│   ├── nextflow.config                      # The configuration file of Nextflow for MACE
│   └── build_dataset.py                     # The python code to build the dataset
├── 2_PiNet2_init/                           # Directory: the initial PiNet2-P3 models trained on FM datasets
│   ├── PiNet2_Seed*_Init/                   # The three PiNet2-P3 models      
│   │   ├── eval/events.<...>                # The validation event file
│   │   ├── checkpoint                       # The file storing the paths of actual checkpoint files
│   │   ├── params.yml                       # The hyper-parameter file
│   │   ├── graph.pbtxt                      # The text-format file of TensorFlow computation graph
│   │   ├── events.<...>                     # The training event files
│   │   └── model.ckpt-<...>                 # The actual Tensorflow checkpoint files
│   └── build_pinet2.py                      # The python code to train PiNet2-P3 models
├── 3_PiNet2_ft/                             # Directory: the fine-tuning of PiNet2-P3 models
│   │                                          based on the PiNNAcLe workflow developed by Yunqi
│   │                                          (https://github.com/yqshao-archive/pinnacle)
│   ├── io                                   # The modified python files for io from the tips package
│   │   │                                       (https://github.com/yqshao-archive/tips/tree/main)
│   │   ├── cp2klog.py                       # Read the information from CP2K log file
│   │   └── ase.py.py                        # Read the information from ASE trajectory file
│   ├── output-cp2k-from-model-seed*/        # The output of the PiNNAcLe workflow
│   │   ├── md/                              # The information of MD simulation
│   │   │   └── gen<...>/wat64_h3o+/asemd.log # The ASE trajectory log file in generation <...>
│   │   ├── models/                          # The information of fine-tuning of models
│   │   │   ├── gen<...>/model1/pinn.log     # The validation log file for models in generation <...>
│   │   │   └── gen<...>/model1/append.log   # The net force per atom on datasets based on CP2K labels
│   │   └── pinnacle.log                     # The log file of PiNNAcLe
│   ├── input/cp2k/r2SCAN-sp.inp             # The input file for single point calculation by CP2K
│   ├── nextflow/                            # The modified Nextflow scripts based those in PiNNAcLe package
│   │   │   ├── module/
│   │   │   │   ├── pinn.nf                  # The script for training models and running MD simulations
│   │   │   │   ├── cp2k.nf                  # The script for single point calculation by CP2K
│   │   │   │   ├── tips.nf                  # The script for sampling new trajectories, and updating datasets
│   │   │   │   └── tools.nf                 # The script for releasing the disk space by removing some files
│   │   └── └── acle-cp2k-from-user-model.nf # The script of main PiNNAcLe workflow started from an user model
│   ├── cp2k-v2023.def                       # The defination file for the singularity image of cp2k-2023.2
│   ├── main.nf                              # The entry point of PiNNAcLe
│   └── nextflow.config                      # The configuration file of Nextflow for PiNNAcLe
├── 4_MD_equ/                                # Directory: the product simulation by fine-tuned PiNet2-P3 models
│   ├── models-ft/                           
│   │   ├── PiNet2_Seed*_ft/                 # The fine-tuned three PiNet2-P3 models      
│   │   │   ├── eval/events.<...>            # The validation event file
│   │   │   ├── checkpoint                   # The file storing the paths of actual checkpoint files
│   │   │   ├── params.yml                   # The hyper-parameter file
│   │   │   ├── graph.pbtxt                  # The text-format file of TensorFlow computation graph
│   │   │   ├── events.<...>                 # The training event files
│   │   └── └── model.ckpt-<...>             # The actual Tensorflow checkpoint files
│   ├── mds/                                 # The MD trajectories and analyse results
│   │   ├── PiNet2_Seed*_ft/
│   │   │   ├── <...>.rdf                    # The RDF results between given species pairs
│   │   │   ├── wat64_h3o+-330K-O-zzy.msd    # The MSD results of all oxygen atoms
│   │   └── └── wat64_h3o+-330K-proton-zzy-nearestOxyg # The MSD results of proton
│   │                                                  (represented by the oxygen in hydronium)
│   ├── nvt_md_pinet2.nf                     # The Nextflow script for MD driven by PiNet2-P3 models
│   ├── nextflow.config                      # The configuration file of Nextflow for MD by PiNet2-P3 models
│   └── analyse.py                           # The python code to analyse the trajectories
├── 5_MD_nonequ/                             # Directory: the placeholder for non-equilibrium MD simulations
├── 6_Plotting/                              # Directory: the plotting results
│   ├── learning_curve_seed*.jpg             # The fine-tuning curve for PiNNAcLe
│   ├── net_force_e_dress_seed*.jpg          # The net force per atom on datasets based on CP2K labels
│   └── plotting.py                          # The python code for plotting
├── setting.py                               # The global setting file for this project
└── environment.yml                          # The conda environment file for this project
```
# Installation
+ Download the PiNNAcLe-Proton-aq project
```
git clone https://github.com/zzy2014/PiNNAcLe-Proton-aq.git
```
Then, change the PROJ_DIR variable in the setting.py file to the full path of _PiNNAcLe-Proton-aq_
+ Create the conda environment
```
cd PiNNAcLe-Proton-aq
conda env create -f environment.yml
conda activate pinn
```
+ Install Nextflow by following https://www.nextflow.io/docs/latest/install.html
+ Install Singularity image of cp2k-2023.2
```
cd 3_PiNet2_ft
apptainer build cp2k2023_2.sif cp2k-v2023.def
```
Then, change the full path of cp2k2023_2.sif in 3_PiNet2_ft/nextflow.config file
+ Install the PiNN package
```
cd ..
pip install git+https://github.com/Teoroo-CMC/PiNN.git --no-deps
```
+ Install the tips package developed by Yunqi, and update the modifiations
```
pip install git+https://github.com/yqshao-archive/tips.git
cp {PROJ_DIR}/3_PiNet2_ft/io/*.py {TIPS_DIR}/io/
```
+ Export the PiNNAcLe-Proton-aq project path to PYTHONPATH
```
export PYTHONPATH={PROJ_DIR}:$PYTHONPATH
```
# Usage
+ Run simulations by FM and construct the datasets
```
cd 1_Datasets
nextflow run nvt_md_mace_r2scan.nf -profile alvismace -bg > log.out
```
After the simulations finished, move the trajectory files to the corresponding subfolder MACE-r2scan-<...>, and built the dataset
```
python build_dataset.py
```
+ Train the initial PiNet2-P3 models (by slurm)
```
cd ../2_PiNet2_init
python build_pinet2.py  # using the build_model function, and change the random seed manully
```
After the training finished, estimate the performance of PiNet2-P3 models on energy and force
```
python build_pinet2.py # using the get_ener_force_metrics function, and change the random seed manully
```
+ Fine-tune the PiNet2-P3 models by CP2K lables
Change the hyperparameters in nextflow/acle-cp2k-from-user-model.nf according to your task, especially the four tolerances (frmsetol, ermsetol, fmaxtol, emaxtol) for convergence check.
If disk space is limited, uncommenting the _nx_inp_ channel in acle-cp2k-from-user-model.nf to delete some intermediate files is a good option.
Then, we can start the fine-tuning.
```
cd 3_PiNet2_ft
nextflow run main.nf -profile alvisacle -bg > log.out
```
+ Product equilibrium MD simulations by the fine-tuned PiNet2-P3 models
Copy the fine-tuned models into 4_MD_equ/models-ft folder, and run the simulations
```
cd 4_MD_equ
nextflow run nvt_md_pinet2.nf -profile alvispinn -bg > log.out
```
Then, we can analyze the trajectories to get the concerned quantities (e.g., RDF and MSD).
```
python analyse.py
```
+ Product non-equilibrium MD simulations by the fine-tuned PiNet2-P3 models
  coming soon..
+ Plot the figures
```
cd 6_Plotting
python plotting.py
```
