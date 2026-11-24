nextflow.enable.dsl=2

params.publish = 'cp2k'
params.cp2k_aux = null

include { labelAttempt as cp2kFirst; labelAttempt as cp2kRetry } from './cp2k-label.nf'

process cp2k {
  tag "$name"
  label 'cp2k'
  publishDir "$params.publish/$name"

  input:
    tuple val(name), path(input)

  output:
    tuple val(name), path('cp2k.log'), emit:logs
    tuple val(name), path('*.{ener,xyz,stress,cell}'), emit:traj, optional: true
    tuple val(name), path('*.restart'), emit:restart, optional:true

  script:
    """
    #!/bin/bash
    $params.cp2k_cmd -i $input | tee cp2k.log
    """
}


process cp2kGenInp {
  tag "$name"
  label 'tips'
  publishDir "$params.publish/$name"

  input:
    tuple val(name), path(input,stageAs: 'ZnO_Single.inp'), path(ds), val(flags)

  output:
    tuple val(name), path('*.inp')

  script:
  """
  tips cp2kinp $input $ds $flags
  """
}


workflow md {
  take:
    ch // [name, input, init, flags]

  main:
    ch | cp2kGenInp // -> [name, inp] \
       | map {name, inp -> [name, inp]} \
       | cp2k

  emit:
    traj = cp2k.out.traj
    logs = cp2k.out.logs
    restart = cp2k.out.restart
}

workflow sp {
  take:
    ch // [name, inp, geo]

  main:
    ch.map {name, inp, geo -> [name, inp, geo, '-f asetraj']} | cp2kGenInp
    cp2kGenInp.out.join(ch.map {name, inp, geo -> [name, geo]})
      .map {name, inp, geo -> [name, inp, geo, 1, file("${moduleDir}/../../bin/cp2k_label.py")]} | cp2kFirst

    cp2kFirst.out.branch {name, result ->
      retry: new groovy.json.JsonSlurper().parseText(result.resolve('label-outcome.json').text).retryable
      done: true
    }.set {first}
    first.retry.join(ch.map {name, inp, geo -> [name, geo]})
      .map {name, result, geo -> [name, result, geo, 2, file("${moduleDir}/../../bin/cp2k_label.py")]} | cp2kRetry
    first.done.mix(cp2kRetry.out)
      .map {name, result -> [name, result.resolve('cp2k.log'), result.resolve('label-outcome.json')]}
      .set {outcomes}

  emit:
    outcomes
}
