# PiNNAcLe-Proton-aq
Foundation Model (FM) Distillation and Fine-tuning by PiNNAcLe: application to the excess proton in liquid water

# Project structure
In the following project structure, _*_ denotes the random seed, _<...>_ indicates an arbitrary sequence of characters.
```text
PiNNAcLe-Proton-aq/
├── 1_Datasets/                              # Directory: the datasets built from the trajectories of FM
│   ├── geo/                                 # The input structures of liquid water with an excess proton
│   │   ├── wat64_h3o+.vasp                  # In the vasp format
│   │   └── wat64_h3o+.xyz                   # In the ASE extxyz format                                       
│   ├── MACE-r2scan-<...>/                   # The NVT trajectories at the temerature of <...>
│   │   ├── NVT-<...>-1000ps-wat64_h3o+.log  # The placeholder for the ASE trajectory log file
│   │   └── NVT-<...>-1000ps-wat64_h3o+.traj # The placeholder for the ASE trajectory file
│   ├── mix_dataset/                         # The final dataset in the Tensorflow dataset format
│   │   ├── mixProton_330K_400K_500K.yml     # The configuration file of the dataset
│   │   └── mixProton_330K_400K_500K.tfr     # The record file of the dataset   
│   ├── nvt_md_mace_r2scan.nf                # The Nextflow script for MD driven by mace-matpes-r2scan-0 FM
│   ├── nextflow.config                      # The configuration file of Nextflow on the Alvis platform
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
├── 3_PiNet2_ft/                             # Directory: the fine-tuning of initial PiNet2-P3 models
│   │                                          based on the PiNNAcLe workflow developed by Yunqi
│   │                                          (https://github.com/yqshao-archive/pinnacle)
│   ├── output-cp2k-from-model-seed*/        # The output of the PiNNAcLe workflow
│   │   ├── md/                              # The information of MD simulation in each generation
│   │   │   └── gen<...>/wat64_h3o+/asemd.log # The ASE trajectory log file in generation <...>
│   │   ├── models/                          # The information of MD simulation in each generation
│   │   │   ├── gen<...>/model1/pinn.log     # The validation log file for models in generation <...>
│   │   │   └── gen<...>/model1/append.log   # The predicted net force per atom by models on current datasets
│   ├── input/cp2k/r2SCAN-sp.inp             # The input file for single point calculation by CP2K
│   ├── nextflow/                            # The modified Nextflow scripts based that in PiNNAcLe package
│   │   │   ├── module/
│   │   │   │   ├── module

```
# Installation
+ download the PiNNAcLe-Proton-aq project
```
git clone https://github.com/zzy2014/PiNNAcLe-Proton-aq.git
```
  - then, change the PROJ_DIR variable in the setting.py file to full path of _PiNNAcLe-Proton-aq-main_
+ create the conda environment
```
cd PiNNAcLe-Proton-aq-main
conda env create -f environment.yml
conda activate pinn
```
+ install the PiNN package
```
pip install git+https://github.com/Teoroo-CMC/PiNN.git --no-deps
```
+ install the tips package developed by Yunqi
```
cd ..
pip install git+https://github.coom/teoroo-cmc/tips.git
cp -r {PROJ_DIR}/3_PiNet2_ft/ase.py {TIPS_DIR}/io/
```
+ export the PiNNAcLe-Proton-aq project path to PYTHONPATH
```
export PYTHONPATH={PROJ_DIR}:$PYTHONPATH
```
# Usage
+ run simulations by FM and construct the FM datasets
```
cd 1_Datasets
nextflow run nvt_md_mace_r2scan.nf -profile alvismace -bg > log.out
```
after the simulations finished, move the trajectory files to the corresponding folder MACE-r2scan-<...>
```
python build_dataset.py
```
+ training the initial PiNet2-P3 models
```
cd ../2_PiNet2_init
python build_pinet2.py  # using the build_model function, and change the random seed manully
```
after the training finished, estimate the performance of PiNet2-P3 models on energy and force
```
python build_pinet2.py # using the get_ener_force_metrics function
```

