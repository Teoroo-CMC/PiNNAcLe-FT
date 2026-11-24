nextflow.enable.dsl = 2

params.model = null
params.geometry = "${projectDir}/../../example/3_pinet2_ft/input/geo/wat64_h3o+.xyz"
params.durations = '0.01'
params.publish = 'smoke-output'
params.md_extra_flags = ''
params.resource_probes = false
params.mace_probe = false

include { md } from '../../nextflow/module/pinn.nf'

// These probes only check that MD's time override does not affect other labels.
process trainProbe {
    label 'pinn'
    output: path('training-probe')
    script: 'touch training-probe'
}
process cp2kProbe {
    label 'cp2k'
    output: path('labeling-probe')
    script: 'touch labeling-probe'
}
process maceProbe {
    label 'mace'
    output: path('mace-probe')
    script: 'touch mace-probe'
}

workflow h3o {
    assert params.model : 'Supply a pretrained H3O model directory with --model'
    md(Channel.fromList(params.durations.toString().split(',').toList()).map { duration ->
        tuple("h3o-${duration}", file(params.model, checkIfExists: true),
              file(params.geometry, checkIfExists: true),
              "--ensemble nvt --dt 0.5 --T 330 --log-every 10 ${params.md_extra_flags} --t ${duration}")
    })
}

workflow {
    h3o()
    if (params.resource_probes) {
        trainProbe()
        cp2kProbe()
        if (params.mace_probe) maceProbe()
    }
}
