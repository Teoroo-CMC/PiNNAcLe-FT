#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include {entry} from './nextflow/acle-cp2k-from-user-model.nf'

workflow {entry()}

