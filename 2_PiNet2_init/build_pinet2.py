import math
import os
import numpy as np
from pathlib import Path

# build a Model for potentials
def build_model(nSeed):

    import yaml
    import warnings
    from pinn import get_model, get_network
    from pinn.utils import init_params
    from pinn.io import write_tfrecord, load_tfrecord, sparse_batch
    import tensorflow as tf
    from tensorflow.python.lib.io.file_io import FileIO
    from tempfile import mkdtemp, mkstemp
    from setting import proj_params

    # important parameters
    nDepth = 5
    dRc = 6.0
    nGaussBasis = 10
    nModelSize = 16
    dLearnRate = 5.0e-05
    dLearnRateDecay = 0.994
    nTrainSteps = 5000000
    nEvalSteps = 100
    nBatch = 1
    nTrainRatio = 8

    # other parameters
    nCkpts = 1
    nlog_every = 1000
    nckpt_every = 10000
    nmax_ckpts = 1
    nshuffleBuffer = 1000
    bShuffle = True
    bPreprocess = True
    bCache = True
    bEarlyStop = False

    # set the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # split training and test sets
    # make sure the energy is eV, the force is eV/Ang
    strDataFile = proj_params["dataset_dir"] / "mix_dataset" / "mixProton_330K_400K_500K.yml"
    dataset = load_tfrecord(str(strDataFile), splits={'train':nTrainRatio, 'vali':10-nTrainRatio}, shuffle=bShuffle, seed=nSeed)
    write_tfrecord(str(proj_params["pinet2_init_dir"] / f'train-seed{nSeed}.yml'), dataset['train'])
    write_tfrecord(str(proj_params["pinet2_init_dir"] / f'vali-seed{nSeed}.yml'), dataset['vali'])

    # get model parameters
    params = {}

    # define model
    params["model"] = { "name": "potential_model",
                        "params":
                             {  "use_force": True,
                                "e_loss_multiplier": 0.1,
                                "f_loss_multiplier": 100.0,
                                "e_scale": 1.0,
                                "e_unit": 1.0,
                                "use_e_per_atom": False,
                                "log_e_per_atom": True,
                             }
                       }

    # model para
    params["network"] = {   "name": "PiNet2",
                            "params":
                                {  "depth": nDepth,
                                    "rc": dRc,
                                    "n_basis": nGaussBasis,
                                    "basis_type": "gaussian",
                                    "pi_nodes": [nModelSize],
                                    "pp_nodes": [nModelSize]*4,
                                    "ii_nodes": [nModelSize]*4,
                                    "out_nodes": [nModelSize],
                                    "rank": 3,
                                    "weighted": False
                                }
                            }

    # adam
    params["optimizer"] = {     "class_name": "Adam",
                                "config":
                                    {
                                        "global_clipnorm": 0.01,
                                        "learning_rate":
                                            {
                                                "class_name": "ExponentialDecay",
                                                "config":
                                                {
                                                    "decay_rate": dLearnRateDecay,
                                                    "decay_steps": 100000,
                                                    "initial_learning_rate": dLearnRate,
                                                }
                                            }
                                    }
                            }

    # with open(strWorkDir + "pinet2.yml", 'w') as f:
    #     yaml.dump(params, f)
    strModelName = "PiNet2_Seed%d_Init"%(nSeed)

    # continue training / fine-tunning / transfer learning
    if os.path.exists(proj_params["pinet2_init_dir"] / strModelName):
        print("!!!!!!!Note: continue training, and use params[model] and params[network] in old model!")

    params['model_dir'] = strModelName

    # initial e_dress is necessary, even running in a fine-tuning proceess
    ds = load_tfrecord(str(proj_params["pinet2_init_dir"] / f"train-seed{nSeed}.yml"))
    init_params(params, ds)

    scratch_dir = None
    if scratch_dir is not None:
        scratch_dir = mkdtemp(prefix='pinn', dir=scratch_dir)
    def _dataset_fn(fname):
        dataset = load_tfrecord(fname)
        if nBatch is not None:
            dataset = dataset.apply(sparse_batch(nBatch))
        if bPreprocess:
            def pre_fn(tensors):
                with tf.name_scope("PRE") as scope:
                    network = get_network(params['network'])
                    tensors = network.preprocess(tensors)
                return tensors
            dataset = dataset.map(pre_fn)
        if bCache:
            if scratch_dir is not None:
                cache_dir = mkstemp(dir=scratch_dir)
            else:
                cache_dir = ''
            dataset = dataset.cache(cache_dir)
        return dataset

    train_fn = lambda: _dataset_fn(str(proj_params["pinet2_init_dir"] / f'train-seed{nSeed}.yml')).repeat().shuffle(nshuffleBuffer)
    eval_fn = lambda: _dataset_fn(str(proj_params["pinet2_init_dir"] / f'vali-seed{nSeed}.yml'))
    config = tf.estimator.RunConfig(keep_checkpoint_max=nmax_ckpts,
                                    log_step_count_steps=nlog_every,
                                    save_summary_steps=nlog_every,
                                    save_checkpoints_steps=nckpt_every)

    model = get_model(params, config=config)

    if bEarlyStop:
        early_stop = "loss:1000"
        stops = {s.split(':')[0]: float(s.split(':')[1])
                 for s in early_stop.split(',')}
        hooks = [tf.estimator.experimental.stop_if_no_decrease_hook(
            model, k, v) for k,v in stops.items()]
    else:
        hooks=None

    # tensorflow set the mode=tf.estimator.ModeKeys.TRAIN atomaticlly
    train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=nTrainSteps, hooks=hooks)

    # tensorflow set the mode=tf.estimator.ModeKeys.EVAL atomaticlly
    eval_spec  = tf.estimator.EvalSpec(input_fn=eval_fn, steps=nEvalSteps)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


# get the inital energy and force metricx for selecting the hyper-parameters in PiNNAncL package
def get_ener_force_metrics(nSeed):

    from pinn import get_calc
    from ase import Atoms
    from pinn.io import load_tfrecord
    from setting import proj_params

    dataset = load_tfrecord(str(proj_params["dataset_dir"] / "mix_dataset" / "mixProton_330K_400K_500K.yml"))
    fields = ['elems', 'coord', 'cell', 'e_data', 'f_data']
    refData = {k: [] for k in fields}
    for example in dataset:
        for k in fields:
            refData[k].append(example[k].numpy())

    nNumFrame = len(refData['e_data'])
    print("Number of frames = %d"%(nNumFrame))

    # get predictions 

    strModelPath = str(proj_params["pinet2_init_dir"] / f"PiNet2_Seed{nSeed}_Init")
    calc = get_calc(strModelPath)
    calc.properties = ['energy', 'force']

    # based on the code in tips.nf of PiNNAncL, the energy is per atom value
    e_label = []
    f_label = []
    e_pred = []
    f_pred = []
    for nFramIndex in range(0, nNumFrame):

        e_label.append(refData['e_data'][nFramIndex] / len(refData['elems'][nFramIndex]))
        f_label.append(refData['f_data'][nFramIndex])

        atoms = Atoms(numbers=refData['elems'][nFramIndex],
                      positions=refData['coord'][nFramIndex],
                      cell=refData['cell'][nFramIndex],
                      pbc=True)

        atoms.set_calculator(calc)
        
        dPredPotEner = atoms.get_potential_energy()
        dPredForce = atoms.get_forces()

        e_pred.append(dPredPotEner / len(refData['elems'][nFramIndex]))
        f_pred.append(dPredForce)

    e_label = np.array(e_label)
    f_label = np.array(f_label)
    e_pred = np.array(e_pred)
    f_pred = np.array(f_pred)

    emax = np.max(np.abs(e_pred-e_label))
    fmax = np.max(np.abs(f_pred-f_label))
    ermse = np.sqrt(np.mean((e_pred-e_label)**2))
    frmse = np.sqrt(np.mean((f_pred-f_label)**2))
    print(f"emax={1000*emax} meV/Atom, fmax={1000*fmax} meV/Ang, ermse={1000*ermse} meV/Atom, frmse={1000*frmse} meV/Ang")

#
if __name__ == '__main__':

    # build initial PiNet2 models on the MACE datasets
    # execute by slurm
    random_seed = 1 # [1,2,4]
    build_model(random_seed)
    
    # compute the predictive performance to help use determining the hyper-parameters in Pinnacle
    # get_ener_force_metrics(random_seed)
