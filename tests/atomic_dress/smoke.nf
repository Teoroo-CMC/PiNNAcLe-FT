nextflow.enable.dsl = 2

params.model = null
params.dataset = null
params.train_steps = null
params.publish = 'smoke-output'
params.edress_target = 'total'
params.init_flag = '--init'
params.fmaxtol = 0.500
params.emaxtol = 0.0025
params.frmsetol = 0.031
params.ermsetol = 0.001

include { train } from '../../nextflow/module/pinn.nf'

workflow {
    assert params.model && params.dataset && params.train_steps
    train(Channel.of(tuple('gen1/model1',
        file(params.dataset, checkIfExists: true),
        file(params.model, checkIfExists: true),
        "--seed 1 --train-steps ${params.train_steps} --batch 1 --eval-bs 1 " +
        "--shuffle-buffer 10 --log-every 1 --ckpt-every 1 --max-ckpts 2 ${params.init_flag}")))
}
