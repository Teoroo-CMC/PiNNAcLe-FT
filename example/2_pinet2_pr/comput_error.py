import math
import os
import numpy as np
from pathlib import Path

WORK_DIR = Path(os.getcwd())
DATASET_DIR = WORK_DIR.parent / "1_dataset"

# get the inital energy and force metricx for selecting the hyper-parameters in PiNNAncL package
def get_ener_force_metrics(nSeed):

    from pinn import get_calc
    from ase import Atoms
    from pinn.io import load_tfrecord

    dataset = load_tfrecord(str(DATASET_DIR / "mixProton_330K_400K_500K.yml"))
    fields = ['elems', 'coord', 'cell', 'e_data', 'f_data']
    refData = {k: [] for k in fields}
    for example in dataset:
        for k in fields:
            refData[k].append(example[k].numpy())

    nNumFrame = len(refData['e_data'])
    print("Number of frames = %d"%(nNumFrame))

    # get predictions 

    strModelPath = str(WORK_DIR / f"PiNet2_Seed{nSeed}_Init")
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

    # compute the predictive performance to help use determining the hyper-parameters in Pinnacle
    # execute by slurm
    random_seed = 1 # [1,2,4]
    get_ener_force_metrics(random_seed)
