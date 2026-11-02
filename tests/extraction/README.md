# Extraction integration regression

Requires Python 3, Nextflow and the repository's supported TIPS environment. ASE and NumPy are needed only to generate the synthetic fixture. No models, CP2K, research trajectories or DFT labels are required.

```sh
python tests/extraction/regression.py --output /tmp/extraction-regression
```

The test runs the original module (the added label removed), the patched module with no override, and optionally the patched module with an extraction override. It compares five selected XYZ filenames and file hashes, then checks that each repeated run is `CACHED`. Outputs and traces are retained for inspection. Use a fresh output directory each time.

Container/environment settings can be supplied with `--config /absolute/path/runtime.config`. To test Slurm placement, add `--override /absolute/path/site-extraction.config`; this submits a real allocation. Verify its native job ID against scheduler accounting. Queue delays are not bounded by the process walltime.

For a container-only ASE installation, generate the fixture separately:

```sh
apptainer exec /path/to/image.sif python tests/extraction/regression.py \
  --generate-only /shared/path/synthetic.traj
python tests/extraction/regression.py --fixture /shared/path/synthetic.traj \
  --config /shared/path/runtime.config --output /shared/path/test-results
```

Run local tests only where small extraction tasks are permitted. For cross-executor comparisons, use the same architecture-compatible container on both execution surfaces. A successful small fixture test establishes regression behavior, not large-trajectory performance.
