# Example: an excess proton in liquid water
This example demonstrates how to build PiNet2-P3 models at the CP2K r2SCAN/TZV2P level for an excess proton in liquid water.

# Construct the dataset
  + Run MD simulations using the mace-matpes-r2scan-0 foundation model at three different temperatures (i.e., 330 K, 400 K, and 500 K).
```
cd 1_dataset
nextflow run nvt_md_mace_r2scan.nf -profile alvismace -bg > log.out
```
  + Move the trajectory files to the corresponding subfolder MACE-r2scan-*, and built the dataset
```
python build_dataset.py
```
# Pre-train the PiNet2-P3 models
  + Run the training script
```
cd ../2_pinet2_pr
python build_pinet2.py  # run by slurm, and change the random seed manully
```
  + After the training finished, estimate the performance of PiNet2-P3 models on energy and force
```
python comput_error.py # run by slurm, and change the random seed manully
```
# Fine-tune the PiNet2-P3 models at the CP2K r2SCAN/TZV2P level
```
cd ..
cp ../main.nf 3_pinet2_ft
cp ../nextflow.config 3_pinet2_ft
cp -r ../nextflow/module 3_pinet2_ft/nextflow
cd 3_pinet2_ft
```
 + Modify the hyperparameters in nextflow/acle-cp2k-from-user-model.nf, especially the four tolerances (frmsetol, ermsetol, fmaxtol, emaxtol) for convergence check.
 + If disk space is limited, using the release_space channel to delete some intermediate files is a good option.
 + Then, we can start the fine-tuning
```
nextflow run main.nf -profile alvis -bg > log.out
```
# Plot the fine-tuning curve
```
cd ../4_plotting
python plotting.py
```
