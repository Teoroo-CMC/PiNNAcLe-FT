nextflow.enable.dsl=2

process train {
  label 'pinn'
  publishDir "$params.publish/$name"

  input:
    tuple val(name), path(dataset), path(input, stageAs:'input'), val(flags)

  output:
    tuple val(name), path('model', type:'dir'), emit: model
    tuple val(name), path('pinn.log'), emit: log
    path('append.log') // publish only

  script:
    convert_flag = "${(flags =~ /--seed[\s,\=]\d+/)[0]}"
    train_flags = "${flags.replaceAll(/--seed[\s,\=]\d+/, '')}"
    dataset = (dataset instanceof Path) ? dataset : dataset[0].baseName+'.yml'
    """
    #!/bin/bash

    pinn convert $dataset -o 'train:9,eval:1' $convert_flag

    if [ ! -f $input/params.yml ];  then
        mkdir -p model; cp $input model/params.yml
    else
        cp -rL $input model
    fi
    pinn train model/params.yml --model-dir='model'\
        --train-ds='train.yml' --eval-ds='eval.yml'\
        $train_flags
    pinn log model/eval > pinn.log

    python << 'EOF'
import numpy as np
from ase import Atoms
from pinn import get_calc
from pinn.io import load_tfrecord

fields = ['elems', 'coord', 'cell', 'e_data', 'f_data']

# calculate the net force per atom on whole dataset
whole_dataset = {k: [] for k in fields}
for example in load_tfrecord("${dataset}"):
    for k in fields:
        whole_dataset[k].append(example[k].numpy())

net_force_x = []
net_force_y = []
net_force_z = []
num_frame = len(whole_dataset['e_data'])
for i in range(num_frame):
    num_atom = len(whole_dataset['elems'][i])
    f_label = whole_dataset['f_data'][i]
    assert len(f_label[:,0]) == num_atom
    net_force_x.append(np.abs(np.sum(f_label[:,0]) / num_atom))
    net_force_y.append(np.abs(np.sum(f_label[:,1]) / num_atom))
    net_force_z.append(np.abs(np.sum(f_label[:,2]) / num_atom))

msg = []
net_force_msg = "Net force per atom (meV/Ang/Atom) on whole set,"
net_force_msg += f"x={1000*np.mean(net_force_x)},"
net_force_msg += f"y={1000*np.mean(net_force_y)},"
net_force_msg += f"z={1000*np.mean(net_force_z)}."
msg.append(net_force_msg)

# calculate the e_max, f_max on validation set
# based on the code in tips.nf of PiNNAncL, the energy is per atom value
valid_dataset = {k: [] for k in fields}
for example in load_tfrecord("eval.yml"):
    for k in fields:
        valid_dataset[k].append(example[k].numpy())

calc = get_calc('model')
calc.properties = ['energy', 'force']

e_label = []
f_label = []
e_pred = []
f_pred = []
num_frame = len(valid_dataset['e_data'])
for i in range(0, num_frame):

    num_atom = len(valid_dataset['elems'][i])
    e_label.append(valid_dataset['e_data'][i] / num_atom)
    f_label.append(valid_dataset['f_data'][i])

    atoms = Atoms(numbers=valid_dataset['elems'][i],
                    positions=valid_dataset['coord'][i],
                    cell=valid_dataset['cell'][i],
                    pbc=True)
    atoms.set_calculator(calc)

    e_pred.append(atoms.get_potential_energy() / num_atom)
    f_pred.append(atoms.get_forces())

e_label = np.array(e_label)
f_label = np.array(f_label)
e_pred = np.array(e_pred)
f_pred = np.array(f_pred)

emax = np.max(np.abs(e_pred-e_label))
fmax = np.max(np.abs(f_pred-f_label))
ermse = np.sqrt(np.mean((e_pred-e_label)**2))
frmse = np.sqrt(np.mean((f_pred-f_label)**2))

thresh_msg = "energy and force measures on validation set, "
thresh_msg += f"emax={1000*emax} meV/Atom, fmax={1000*fmax} meV/Ang, ermse={1000*ermse} meV/Atom, frmse={1000*frmse} meV/Ang"
msg.append(thresh_msg)

with open("append.log", "w") as f:
    for item in msg:
        print(item, file=f)

EOF
    """
}

process md {
  label 'pinn'
  publishDir "$params.publish/$name"

  input:
    tuple val(name), path(model,stageAs:'model*'), path(init, stageAs:'init*'), val(flags)

  output:
    tuple val(name), path('asemd.traj'), emit: traj
    tuple val(name), path('asemd.log'), emit: log

  script:
    """
    #!/usr/bin/env python
    import re
    import pinn
    import tensorflow as tf
    from ase import units
    from ase.io import read
    from ase.io.trajectory import Trajectory
    from ase.md import MDLogger
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.bussi import Bussi
    # from ase.md.nvtberendsen import NVTBerendsen
    from tips.bias import EnsembleBiasedCalculator

    # ------------ patch ase properties to write extra cols --------------------
    from ase.calculators.calculator import all_properties
    all_properties+=[f'{prop}_{extra}' for prop in ['energy', 'forces', 'stress'] for extra in ['avg','std','bias']]
    # --------------------------------------------------------------------------

    setup = {
      'ensemble': 'nvt', # ensemble
      'T': 330, # temperature in K
      't': 50, # time in ps
      'dt': 0.5, # timestep is fs
      'taut': 100, # thermostat damping in steps
      'taup': 1000, # barastat dampling in steps
      'log-every': 20, # log interval in steps
      'pressure': 1, # pressure in bar
      'compressibility': 4.57e-4, # compressibility in bar^{-1}
      'bias': None,
      'kb': 0,
      'sigma0': 0,
    }

    flags = {
      k: v for k,v in
        re.findall('--(.*?)[\\s,\\=]([^\\s]*)', "$flags")
    }
    setup.update(flags)
    ensemble=setup['ensemble']
    T=float(setup['T'])
    t=float(setup['t'])*units.fs*1e3
    dt=float(setup['dt'])*units.fs
    taut=int(setup['taut'])
    taup=int(setup['taup'])
    every=int(setup['log-every'])
    pressure=float(setup['pressure'])
    compressibility=float(setup['compressibility'])

    ${(model instanceof Path) ?
    "calc = pinn.get_calc('$model')" :
    """
    models = ["${model.join('", "')}"]
    calcs = [pinn.get_calc(model) for model in models]
    if len(calcs) == 1:
        calc =  calcs[0]
    else:
        calc = EnsembleBiasedCalculator(calcs,
                                        bias=setup['bias'],
                                        kb=float(setup['kb']),
                                        sigma0=float(setup['sigma0']))
    """}

    atoms = read("$init")
    atoms.set_calculator(calc)
    if not atoms.has('momenta'):
        MaxwellBoltzmannDistribution(atoms, T*units.kB)

    if ensemble == 'npt':
        dyn = NPTBerendsen(atoms, timestep=dt, temperature=T, pressure=pressure,
                      taut=dt * taut, taup=dt * taup, compressibility=compressibility)
    if ensemble == 'nvt':
        # 0.2 ps was selected based on Table 1 in https://arxiv.org/pdf/0803.4060
        dyn = Bussi(atoms, timestep=dt, temperature_K=T, taut=dt*400)
        # dyn = NVTBerendsen(atoms, timestep=dt, temperature=T, taut=dt * taut)

    dyn.attach(
        MDLogger(dyn, atoms, 'asemd.log', stress=True, mode="w"),
        interval=int(every))
    dyn.attach(
        Trajectory('asemd.traj', 'w', atoms).write,
        interval=int(every))
    dyn.run(int(t/dt))
    """
}
