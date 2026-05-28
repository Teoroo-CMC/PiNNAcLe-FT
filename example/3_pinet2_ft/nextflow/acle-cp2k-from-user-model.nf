#!/usr/bin/env nextflow

// The activated learning workflow  ======================================================
//
// The '--proj' parameter controls the output directory. See the parameters
// sections below for other parameters that can be tuned for the workflow.
//
//                                         written by  Yunqi Shao, first ver.: 2022.Aug.29
//                                                 adapted as PiNNAcLe recipe: 2023.Apr.24
//========================================================================================

nextflow.enable.dsl = 2
nextflow.preview.recursion = true

def logger (msg) {
  logfile = file("$params.publish/pinnacle.log")
  if (!logfile.getParent().exists()) {logfile.getParent().mkdirs()}
  logfile.append("$msg \n")
  logfile.withWriterAppend { it.flush() }
}

// entrypoint parameters ==================================================================
params.proj = 'Proton'
params.publish       = 'output-cp2k-from-model-seed4'   // output folder name, change the seed manuly
params.init_geo      = 'input/geo/*.xyz' // XYZ files
params.init_model    = '../2_pinet2_pr/PiNet2_Seed4_Init/' // change the seed manuly
params.init_ds       = '/dev/null' //in this case, make sure the sp_points*valid_split_ratio >= 1, otherwise no model training
params.init_time     = 5.0 // initial MD time
params.init_steps    = 5000000 // initial training time for mlp, the already trained steps should be added
params.ens_size      = 1 // ensamble size of mlp models
params.restart       = false // is restarting the job from generation params.restart_from
params.restart_from  = 5  // restart from gen5
params.restart_conv  = true // is the last gen converged? If you want to use your own initial models, set it as true.
//========================================================================================

// acle parameters =======================================================================
params.ref           = 'cp2k' // reference (module name)
params.ref_inp       = 'input/cp2k/r2SCAN-sp.inp'
params.mlp           = 'pinn' // machine learning potential (module name)
params.train_flags   = '--log-every 1000 --ckpt-every 1000 --batch 1 --max-ckpts 1 --shuffle-buffer 3000'
params.train_init    = '--init' // intialize the parameters for mlp, e.g., e_dress
params.exit_at_max_time = false
params.max_gen       = 100 // maximal number of generation loops
params.min_time      = 5.0 // minimal MD simulation time, used to update the simulation time for next gen
params.max_time      = 5000.0 // maximal MD simulation time, used to update the simulation time for next gen
params.md_flags      = '--ensemble nvt --dt 0.5 -t 10 --log-every 100 --T 330'
// -of idx.xyz: inform tips to convert the asetraj to xyz files using the snapshot index as the file name
// --nsample: number of sampling snapshots from asetraj
// --start-idx: to avoid data leakage during testing, some snapshots from the begining of MD should be deleted
// --filter 'mindist>0.3': only use the snapshots in which the minimal distance between any atom pairs > 0.3
params.collect_flags = "-f asetraj --subsample uniform --nsample 10 --start-idx 20 -of idx.xyz -o ds --filter 'mindist>0.3'"
params.sp_points     = 10 // number of single point calculation jobs, usually it = nsample
params.merge_flags   = '-f cp2klog' // format the single point calculation output file
params.old_flag      = '--subsample uniform --psample 100' // when mixing the old and new datasets, 240 old samples were used
params.new_flag      = '--subsample uniform --psample 100' // when mixing the old and new datasets, all new samples (100%) were used
params.frmsetol      = 0.031 // the RMSE of force components (eV/ang)
params.ermsetol      = 0.001 // the RMSE of energy per atom (eV)
params.fmaxtol       = 0.500 // the maximal absolute differnece of force components (eV/ang)
params.emaxtol       = 0.0025 // the maximal absolute differnece of energy per atom (eV)
params.retrain_step  = 5000
params.acc_fac       = 2.0 // multiplier for the next MD simulation time if the last gen converged
params.brake_fac     = 1.0 // multiplier for the next MD simulation time if the last gen not converged
params.filters       = "--filter 'abs(force)<1000.0'" // used in check and dsmix to remove any structures with DFT force component > 1000 eV/Ang
params.change_edress = true // changing the e_dress of PiNet2 model (e.g., from MACE-r2SCAN to DFT-r2SCAN), which used for the changing of energy level
//========================================================================================

// Imports (publish directories are set here) ============================================
include { convert} from './module/tips.nf' addParams(publish: "$params.publish/collect") // pick up snapshots, and converte ase traj to xyz 
include { dsmix } from './module/tips.nf' addParams(publish: "$params.publish/dsmix") // new dataset.tfr
include { merge } from './module/tips.nf' addParams(publish: "$params.publish/merge") // read cp2k outputs and merge snapshots in to extxyz file
include { check } from './module/tips.nf' addParams(publish: "$params.publish/check") // check if force and energy converged, and save geo.xyz for next MD
include { train } from "./module/${params.mlp}.nf" addParams(publish: "$params.publish/models")
include { md } from "./module/${params.mlp}.nf" addParams(publish: "$params.publish/md")
include { sp } from "./module/${params.ref}.nf" addParams(publish: "$params.publish/label")
include { release_space } from "./module/tools.nf" addParams(publish: "$params.publish/tools")
//========================================================================================

// Entry point
workflow entry {
  logger('Starting an AcLe Loop')
  init_ds = file(params.init_ds)
  init_geo = file(params.init_geo)
  params.geo_size = init_geo.size
  ens_size = params.ens_size.toInteger()
  logger("Initial dataset: ${init_ds.name};")
  logger("Initial geometries ($params.geo_size) in ${params.init_geo}")
  logger("Convergence tolerance: emaxtol=$params.emaxtol; ermsetol=$params.ermsetol; fmaxtol=$params.fmaxtol; frmsetol=$params.frmsetol.")

  if (params.restart) {
    init_gen = params.restart_from.toString()
    init_models = file("${params.publish}/models/gen${init_gen}/*/model", type:'dir')
    init_geo = file("${params.publish}/check/gen${init_gen}/*/*.xyz")
    init_ds = file("${params.publish}/dsmix/${init_gen}/mix-ds.{yml,tfr}")
    logger("restarting from gen$init_gen ensemble of size $ens_size;")
    init_gen = (init_gen.toInteger()+1).toString()
  } else{
    init_gen = '0'
    init_models = file(params.init_model, type:'any')
    if (!(init_models instanceof Path)) {
      logger("restarting from an ensemble of size $ens_size;")
    } else {
      init_models = [init_models] * ens_size
      logger("starting from scratch with the input $init_models.name of size $ens_size;")
    }
  }
  assert ens_size == init_models.size : "ens_size ($ens_size) does not match input ($init_models.size)"

  steps = params.init_steps.toInteger()
  time = params.init_time.toFloat()
  converge = params.restart_conv.toBoolean()

  init_inp = [init_gen, init_geo, init_ds, init_models, steps, time, converge]
  ch_inp = Channel.value(init_inp)
  acle(ch_inp)
}

// Main Iteration and Loops ==============================================================
workflow acle {
  take:
    ch_init

  main:
  loop.recurse(ch_init)
    .until{ it[0].toInteger()>params.max_gen || (it[5]>=params.max_time.toFloat() && params.exit_at_max_time) }
}

// Loop for each iteration =================================================================
workflow loop {
  take: ch_inp

  main:

  // print info 
  ch_inp.view{logger("[gen${it[0]}] ${it[-1]? 'not training': 'training'} the models."+'\n'+"[gen${it[0]}] current MD time scale ${it[5]} ps.")}

  // retrain or keep the model ============================================================
  // when converge == true, use the models directly; otherwise train the models
  ch_inp \
    | branch {gen, geo, ds, models, step, time, converge -> \
              keep: converge
                return [gen, models]
              retrain: !converge
                return [gen, models, ds, (1..params.ens_size).toList(), step]} \
    | set {ch_model}

  // train_init only be used in gen0, which means the e_dress only be estimated at the begining
  ch_model.retrain.transpose(by:[1,3]) \
    | map {gen, model, ds, seed, steps -> \
           ["gen$gen/model$seed", ds, model,
            params.train_flags+
            " --seed $seed --train-steps $steps"+
            ((gen.toInteger()==0 || params.change_edress)?" $params.train_init":'')]}\
    | train

  train.out.model \
    | map {name, model -> (name=~/gen(\d+)\/model(\d+)/)[0][1,2]+[model]} \
    | map {gen, seed, model -> [gen, model]} \
    | mix (ch_model.keep.transpose()) \
    | groupTuple(size:params.ens_size) \
    | set {nx_models}

  //nx_models.view()

  //=======================================================================================

  // sampling with ensable NN =============================================================
  ch_inp | map {[it[0], it[1], it[5]]} | transpose | set {ch_init_t} // init and time

  nx_models \
    | combine (ch_init_t, by:0)  \
    | map {gen, models, init, t -> \
           ["gen$gen/$init.baseName", models, init, params.md_flags+" --t $t"]} \
    | md
  md.out.traj.set {ch_trajs}
  //=======================================================================================

  // relabel with reference ===============================================================
  // ch_trajs.view()
  ref_inp = file(params.ref_inp)
  ch_trajs \
    | map {name, traj -> [name, traj, params.collect_flags]} \
    | convert \
    | flatMap {name, inps -> inps.collect {["$name/$it.baseName", it]}} \
    | map {name, geo -> [name, ref_inp, geo]} \
    | sp

  sp.out \
    | map {name, logs -> (name=~/(gen\d+\/.+)\/(\d+)/)[0][1,2]+[logs]} \
    | map {name, idx, logs -> [name, idx.toInteger(), logs]} \
    | groupTuple(size:params.sp_points) \
    | map {name, idx, logs -> [name, idx, logs, params.merge_flags]} \
    | merge \
    | set {ch_new_ds}
  //=======================================================================================

  // check convergence ====================================================================
  ch_new_ds \
    | join(ch_trajs) \
    | check \

  check.out \
    | map{name,geo,msg-> \
          [(name=~/gen(\d+)\/.+/)[0][1], geo, msg.contains('Converged')]} \
    | groupTuple(size:params.geo_size.toInteger()) \
    | map {gen, geo, conv -> [gen, geo, conv.every()]}
    | set {nx_geo_converge}

  //=======================================================================================

  // mix the new dataset ==================================================================
  ch_inp.map {[it[0], it[2]]}.set{ ch_old_ds }
  ch_new_ds \
    | map {name, idx, ds -> [(name=~/gen(\d+)\/.+/)[0][1], ds]} \
    | groupTuple(size:params.geo_size.toInteger()) \
    | join(ch_old_ds) \
    | map {it+[params.new_flag, params.old_flag]} \
    | dsmix \
    | set {nx_ds}
  //=======================================================================================

  // combine everything for new inputs ====================================================
  ch_inp.map{[it[0], it[4]]}.set {nx_step}
  ch_inp.map{[it[0], it[5]]}.set {nx_time}

  acc_fac = params.acc_fac.toFloat()
  brake_fac = params.brake_fac.toFloat()
  min_time = params.min_time.toFloat()
  max_time = params.max_time.toFloat()
  retrain_step = params.retrain_step.toInteger()

  nx_geo_converge | join(nx_models) | join(nx_ds) | join(nx_time) | join (nx_step) \
    | map {gen, geo, converge, models, ds, time, step -> \
           [(gen.toInteger()+1).toString(),
            geo, ds, models, \
            converge ? step : step+retrain_step, \
            converge ? Math.min(time*acc_fac, max_time) : Math.max(time*brake_fac, min_time), \
            converge]} \
    | set {nx_inp}
  //=======================================================================================

  check.out.subscribe {name, geo, msg -> logger("[$name] ${msg.trim()}")}
  nx_inp.subscribe {logger("[gen${it[0].toInteger()-1}] next time scale ${it[5]} ps, ${it[6] ? 'next no training planned' : 'next training step '+it[4]}."+'\n'+'-'*80) } \

  // Uncomment the following release_space channel to delete intermediate files and save disk space when storage space is limited.
  // By default, intermediate files in the "models" and "md" subfolders will be deleted. You can change this setting in the module/tools.nf file.
  // nx_inp \
  //   | map { it -> tuple(it[0], file(params.publish)) } \
  //   | release_space

  emit:
  nx_inp
}
