# Atomic-dress regressions

Run the numerical tests with NumPy and PyYAML:

```bash
python -m unittest discover -s tests/atomic_dress -v
```

The smoke workflow invokes the actual training module on a supplied pretrained
checkpoint and existing labelled TFRecord pair. Use a disposable output/work
directory and your normal PiNN runtime; it does not run MD or DFT.

```bash
nextflow run tests/atomic_dress/smoke.nf \
  --model /path/to/pretrained-model \
  --dataset '/path/to/labels.{yml,tfr}' \
  --train_steps 5000001 --edress_target residual \
  --publish smoke-residual -work-dir work-residual
```

Set train_steps to the checkpoint's global_step plus one. The split is 9:1
with seed 1. The training module verifies the saved atomic dress after
training. Check that global_step increases by one and the input checkpoint
and params.yml hashes stay unchanged. For the original behavior, omit
edress_target (or use total) and use separate output/work directories.
With residual selected but init_flag set to --no-init, no dress fit is made.

For a dress-only inference check, copy the original checkpoint and replace
only its dress with new_e_dress from atomic-dress-fit.json. Compare its
energies/forces with the original on the same frames, accounting for
float32 rounding; the fit uses the nine training frames, not the held-out
frame. Distribute the trained checkpoint with its own params.yml.
