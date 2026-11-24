# MD walltime checks

The submission test runs the real MD module through Nextflow's Slurm executor,
using the H3O example geometry and a local `sbatch` recorder. It checks the
generated job headers without submitting jobs or executing MD:

```bash
python tests/md_walltime/test_submission.py --output /tmp/walltime-submissions
```

Python 3.9+, Nextflow and Java are required; no PiNN installation or Slurm
allocation is needed. Use `--nextflow /path/to/nextflow` if necessary. Each
output directory must be new. The cases cover both profiles with scaling off
and on, custom estimates, repeated duration flags, invalid inputs, exactly
one scheduler time directive per job, and unchanged training/labeling limits.

For a real short H3O MD smoke test, use a suitable PiNN runtime and your own
site/account configuration, retaining the updated MD time directive:

```bash
nextflow run tests/md_walltime/smoke.nf -profile arrhen \
  --model /path/to/pretrained-H3O-model --durations 0.01 \
  --adaptive_md_walltime --publish /path/to/disposable-smoke-output
```

This invokes the actual MD process for 20 steps (0.01 ps at 0.5 fs) and writes
three frames. Inspect `.command.run` for a single 10-minute time directive,
check the trajectory's atom count/cell and finite energies/forces, and confirm
the input checkpoint is unchanged. This is a scheduling/integration check;
it is not a long-trajectory stability test. No checkpoint/restart behavior is
added by this change.
