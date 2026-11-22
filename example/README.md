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

## Adaptive MD walltime (optional)

Enable a walltime estimate for each MD job with:

```bash
nextflow run main.nf -profile alvis --adaptive_md_walltime \
  --md_seconds_per_ps 40 --md_walltime_overhead_minutes 3 \
  --md_walltime_min_minutes 10 -bg > log.out
```

Use your usual profile (`arrhen` on Arrhenius). For an existing copied example,
refresh `nextflow.config` while retaining your account, paths and other local
settings. Remove any local `--time`/`-t` scheduler options from `clusterOptions`;
set scheduler limits through Nextflow's `time` directive.

The estimate is `ceil(max(minimum, requested_ps * seconds_per_ps / 60 + overhead))`
minutes. The values above give 10 min for 5 ps, 30 min for 40 ps, 7 h 10 min
for 640 ps and 11 h 10 min for 1000 ps. Calibrate seconds/ps for your model,
system and hardware; 40 is an Arrhenius example setting based on pilot timings.

This changes the MD job allocation, not its simulated duration, forces or
convergence criteria. Training and labeling keep their separate fixed limits.
Adaptive allocation is off by default: PiNN jobs use 4 days on Alvis or 1 day
on Arrhenius. The profile limits previously embedded in `clusterOptions` are
now expressed once through `time`. This option does not add MD checkpointing
or automatic continuation; choose estimates that allow the full job to finish.

# Plot the fine-tuning curve
```
cd ../4_plotting
python plotting.py
```
