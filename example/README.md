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

## Residual atomic-dress fitting (optional)

For an existing pretrained model, fit the atomic dress after accounting for
its frozen neural-network energy contribution:

```bash
nextflow run main.nf -profile alvis --edress_target residual -bg > log.out
```

Use your usual profile (e.g. `arrhen` on Arrhenius). The default
`--edress_target total` retains the original total-DFT-energy fit.
`params.change_edress = true` keeps refitting before each retraining;
generations that reuse a model do not refit.

The training split alone determines an additive correction to the existing
`e_dress`. Predictions used here already include the old dress:
`new_dress = old_dress + pinv(atom_counts) @ (E_DFT - E_old_model)`.
All training frames enter this unweighted fit, matching the original dress
fitter; sample weights still govern subsequent network training. With fixed
composition, only the combined energy baseline is identifiable.

The workflow copies the input model, writes `model.params.e_dress` in the
output `model/params.yml`, and trains with `--no-init`. It checks that the
fitted dress survives training. The input model is not edited. Distribute
the output checkpoint and its `params.yml` together; the fit details are in
`model/atomic-dress-fit.json`. Its RMSE is the algebraic fit; reloaded
float32 predictions can differ slightly. A dress-only update does not change
forces.

For existing example directories, refresh the copied `nextflow/module` and
example workflow. Ensure the job's `PYTHONPATH` points to this updated
repository's `bin` directory (`params.acle_dir` in the Arrhenius profile).

# Plot the fine-tuning curve
```
cd ../4_plotting
python plotting.py
```
