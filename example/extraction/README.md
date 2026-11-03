# Schedule trajectory extraction separately

Add `-c example/extraction/slurm.config` to your normal Nextflow command. Adapt the example to your scheduler and account. Without this override, existing execution settings remain unchanged. The added label applies only to `convert`, not all TIPS processes.

Choose a container compatible with the target node architecture. GPU allocation is not inherently required for extraction; sites with GPU-only access may require it. Set numerical-library thread limits appropriate for the allocated CPUs through your container runtime. Estimate walltime from a representative extraction pilot: frame count, atom count and filters all affect cost.

After an extraction failure, correct resource settings and use `-resume` with the original work directory/cache retained. Configuration changes can invalidate affected task hashes; inspect the trace rather than assuming all tasks were cached.

Example invocation (retain your normal scientific parameters):

```sh
nextflow run main.nf -profile arrhen -c example/extraction/slurm.config -resume
```

The supplied CPU, memory and time values are starting examples, not performance guarantees. On GPU-only partitions, add your site's GPU request to `clusterOptions`; do not add a second conflicting time/account request. Specific `withName` selectors can override label settings, so inspect the generated job script and trace when combining configurations.

See [the integration regression](../../tests/extraction/README.md) for a synthetic-data test of default execution, output equivalence and caching.
