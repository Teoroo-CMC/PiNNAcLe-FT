import math
import os
import numpy as np
from pathlib import Path

WORK_DIR = Path(os.getcwd())
DATASET_DIR = WORK_DIR.parent / "1_dataset"

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
    strDataFile = DATASET_DIR / "mixProton_330K_400K_500K.yml"
    dataset = load_tfrecord(str(strDataFile), splits={'train':nTrainRatio, 'vali':10-nTrainRatio}, shuffle=bShuffle, seed=nSeed)
    write_tfrecord(str(WORK_DIR / f'train-seed{nSeed}.yml'), dataset['train'])
    write_tfrecord(str(WORK_DIR / f'vali-seed{nSeed}.yml'), dataset['vali'])

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
    if os.path.exists(WORK_DIR / strModelName):
        print("!!!!!!!Note: continue training, and use params[model] and params[network] in old model!")

    params['model_dir'] = strModelName

    # initial e_dress is necessary, even running in a fine-tuning proceess
    ds = load_tfrecord(str(WORK_DIR / f"train-seed{nSeed}.yml"))
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

    train_fn = lambda: _dataset_fn(str(WORK_DIR / f'train-seed{nSeed}.yml')).repeat().shuffle(nshuffleBuffer)
    eval_fn = lambda: _dataset_fn(str(WORK_DIR / f'vali-seed{nSeed}.yml'))
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

#
if __name__ == '__main__':

    # pre-train PiNet2 models on the MACE datasets
    # execute by slurm
    random_seed = 1 # [1,2,4]
    build_model(random_seed)
