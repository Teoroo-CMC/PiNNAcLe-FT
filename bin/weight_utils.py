import json
from pathlib import Path

import numpy as np
import yaml


def fit_residual_dress(model_path, elements, e_label, e_pred):
    """Fit an additive dress correction from existing full-model predictions.

    Energies are per atom, as returned by compute_weight. Labels and dress
    use input units; predictions include PiNN's e_unit conversion.
    """
    model_path = Path(model_path)
    params_path = model_path / "params.yml"
    config = yaml.safe_load(params_path.read_text())
    if config["model"]["name"] != "potential_model":
        raise ValueError("Residual dress requires a potential_model")
    params = config["model"]["params"]
    unit = float(params.get("e_unit", 1.0))
    prior = {int(z): float(a) for z, a in params.get("e_dress", {}).items()}
    labels = np.asarray(e_label, dtype=np.float64)
    predictions = np.asarray(e_pred, dtype=np.float64)
    if not elements or any(len(z) == 0 for z in elements):
        raise ValueError("Residual dress requires nonempty training structures")
    if labels.shape != (len(elements),) or predictions.shape != labels.shape:
        raise ValueError("Residual dress energy/structure count mismatch")
    species = sorted(set(prior).union(*(set(map(int, z)) for z in elements)))
    counts = np.array([[np.count_nonzero(np.asarray(z) == s) for s in species]
                       for z in elements], dtype=np.float64)
    old = np.array([prior.get(z, 0.0) for z in species])
    if (not np.isfinite(unit) or unit <= 0 or
            not all(np.isfinite(a).all() for a in (labels, predictions, old))):
        raise ValueError("Non-finite energies/dress or invalid e_unit")
    n_atoms = counts.sum(axis=1)
    # e_pred includes the old dress: ADD, rather than replace, this fit.
    residual = (labels - predictions / unit) * n_atoms
    delta = np.linalg.pinv(counts) @ residual
    new = old + delta
    new[np.all(counts == 0, axis=0)] = old[np.all(counts == 0, axis=0)]
    fitted = {int(z): float(a) for z, a in zip(species, new)}
    report = {
        "target": "E_DFT - E_NN_bare",
        "frames": len(elements),
        "rank": int(np.linalg.matrix_rank(counts)),
        "weighting": "unweighted training split, as in the total-energy fitter",
        "energy_units": "input label units",
        "old_e_dress": {int(z): float(a) for z, a in zip(species, old)},
        "new_e_dress": fitted,
        "rmse_per_atom_before": float(np.sqrt(np.mean((residual / n_atoms)**2))),
        "rmse_per_atom_after": float(np.sqrt(np.mean(
            ((residual - counts @ (new - old)) / n_atoms)**2))),
    }
    params["e_dress"] = fitted
    temporary = params_path.with_suffix(".yml.tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    temporary.replace(params_path)
    (model_path / "atomic-dress-fit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_residual_dress(model_path):
    """Fail if training has overwritten the residual-fitted params.yml."""
    model_path = Path(model_path)
    report = json.loads((model_path / "atomic-dress-fit.json").read_text())
    expected = {int(z): float(a) for z, a in report["new_e_dress"].items()}
    config = yaml.safe_load((model_path / "params.yml").read_text())
    actual = {int(z): float(a) for z, a in config["model"]["params"]["e_dress"].items()}
    if actual != expected:
        raise ValueError("Training overwrote the residual-fitted e_dress")
    print("Residual-fitted e_dress preserved in params.yml")

# compute metrics by given true values, predictions and mask
def compute_metrics(e_pred, e_label, f_pred, f_label, mask=None):

    if mask is None:
        e_diff = e_pred - e_label
        f_diff = f_pred - f_label
    elif np.sum(mask) == 0:
        return "No valid samples in this mask."
    else:
        e_diff = e_pred[mask] - e_label[mask]
        f_diff = f_pred[mask] - f_label[mask]
        
    emax = np.max(np.abs(e_diff))
    fmax = np.max(np.abs(f_diff))
    ermse = np.sqrt(np.mean(e_diff**2))
    frmse = np.sqrt(np.mean(f_diff**2))

    msg = f'energy: emax={1000*emax} meV/Atom, ermse={1000*ermse} meV/Atom;'
    msg += f'force: fmax={1000*fmax} meV/Ang, frmse={1000*frmse} meV/Ang'
    return msg

# compute weight to training and evaluation sets
# if emax/fmax is extremely large, it indicates that the PiNet2-P3 model has never encountered these structures
# discarding them is not always desirable, because they may still correspond to physically valid configurations
# to keep the fine-tuning stable, we therefore reduce their weights for both energies and forces
# change the weights of training and evaluation samples seperately, to estimate the evaluation errors
def compute_weight(model_abs_path, ds_abs_path, tfr_file_name, tolarence,
                   dress_output=None):
    from ase import Atoms
    from pinn import get_calc
    from pinn.io import load_tfrecord

    if dress_output is not None:
        import tensorflow as tf
        if tfr_file_name != "train":
            raise ValueError("Residual dress must use the training split only")
        if Path(dress_output).resolve() == Path(model_abs_path).resolve():
            raise ValueError("Fit the copied output model, not the input model")
        if tf.train.latest_checkpoint(model_abs_path) is None:
            raise ValueError("Residual dress requires a pretrained checkpoint")

    if not ds_abs_path.endswith("/"):
        ds_abs_path += "/"

    # get current model
    calc = get_calc(model_abs_path)
    calc.properties = ['energy', 'force']

    # get dataset
    fields = ['elems', 'coord', 'cell', 'e_data', 'f_data']
    ds = {k: [] for k in fields}
    for example in load_tfrecord(ds_abs_path + f"{tfr_file_name}.yml"):
        for k in fields:
            ds[k].append(example[k].numpy())

    # get the true label and predictions
    e_label = []
    f_label = []
    e_pred = []
    f_pred = []
    num_frame = len(ds['e_data'])
    for i in range(0, num_frame):
        num_atom = len(ds['elems'][i])
        e_label.append(ds['e_data'][i] / num_atom)
        f_label.append(ds['f_data'][i])
        atoms = Atoms(numbers=ds['elems'][i],
                        positions=ds['coord'][i],
                        cell=ds['cell'][i],
                        pbc=True)
        atoms.set_calculator(calc)
        e_pred.append(atoms.get_potential_energy() / num_atom)
        f_pred.append(atoms.get_forces())
    e_label = np.array(e_label)
    f_label = np.array(f_label)
    e_pred = np.array(e_pred)
    f_pred = np.array(f_pred)

    # calculate weight based on the maximal difference between true forces and predicted forces
    fmax_list = np.max(np.abs(f_pred-f_label), axis=(1, 2))
    assert len(fmax_list) == num_frame, "length mismatch"
    weights = []
    with open(ds_abs_path + f"weight-{tfr_file_name}.csv", "w") as g:
        for i in range(0, num_frame):
            weight = 1.0
            if fmax_list[i] > 2.0 * tolarence["fmaxtol"]:
                weight = 0.0
            print(f'{i};{weight}', file=g)
            weights.append(weight)
        g.flush()
    weights = np.array(weights)

    # print the emax per atom and fmax on this set
    cases = [
        ("all samples", None),
        ("with weight > 0.99", weights > 0.99),
        ("with weight < 0.1", weights < 0.1),
    ]

    msg_list = []
    for desc, mask in cases:
        msg = f"before training on {tfr_file_name} set {desc}: "
        msg += compute_metrics(e_pred, e_label, f_pred, f_label, mask)
        msg_list.append(msg)

    with open(ds_abs_path + "append.log", "a") as f:
        for msg in msg_list:
            print(msg, file=f)

    if dress_output is not None:
        fit_residual_dress(dress_output, ds["elems"], e_label, e_pred)

#
def add_weight_to_ds(ds_abs_path, tfr_file_name):
    from pinn.io import load_tfrecord, load_numpy, write_tfrecord


    if not ds_abs_path.endswith("/"):
        ds_abs_path += "/"

    # get dataset
    fields = ['elems', 'coord', 'cell', 'e_data', 'f_data', 'e_weight', 'f_weights']
    ds = {k: [] for k in fields}
    for example in load_tfrecord(ds_abs_path + f"{tfr_file_name}.yml"):
        for k in fields:
            if k in example.keys():
                ds[k].append(example[k].numpy())
    num_frame = len(ds['e_data'])

    # read weight for samples
    weights = []
    with open(ds_abs_path + f"weight-{tfr_file_name}.csv", "r") as f:
        lines = f.readlines()
        weights = [float(l.strip().split(";")[1]) for l in lines]
    assert len(weights) == num_frame

    # write dataset with weight
    for i in range(0, num_frame):
        ds['e_weight'].append(weights[i])
        num_atom = len(ds['elems'][i])
        f_weights = np.full((num_atom, 3), weights[i], dtype=float)
        ds['f_weights'].append(f_weights)

    ds['elems'] = np.array(ds['elems'], dtype=np.int32)
    ds['coord'] = np.array(ds['coord'], dtype=np.float64)
    ds['cell']  = np.array(ds['cell'], dtype=np.float64)
    ds['f_data'] = np.array(ds['f_data'], dtype=np.float64)
    ds['e_data'] = np.array(ds['e_data'], dtype=np.float64)
    ds['e_weight'] = np.array(ds['e_weight'], dtype=np.float64)
    ds['f_weights'] = np.array(ds['f_weights'], dtype=np.float64)

    ds_weight = load_numpy(ds)
    write_tfrecord(ds_abs_path + f"{tfr_file_name}-weight.yml", ds_weight)

# 
def analyse_after_train(model_abs_path, ds_abs_path, tfr_file_name):
    from ase import Atoms
    from pinn import get_calc
    from pinn.io import load_tfrecord


    if not ds_abs_path.endswith("/"):
        ds_abs_path += "/"

    # get current model
    calc = get_calc(model_abs_path)
    calc.properties = ['energy', 'force']

    # calculate the net force per atom
    fields = ['elems', 'coord', 'cell', 'e_data', 'f_data', 'e_weight', 'f_weights']
    ds = {k: [] for k in fields}
    for example in load_tfrecord(ds_abs_path + f"{tfr_file_name}-weight.yml"):
        for k in fields:
            ds[k].append(example[k].numpy())

    net_force_x = []
    net_force_y = []
    net_force_z = []
    num_frame = len(ds['e_data'])
    for i in range(num_frame):
        num_atom = len(ds['elems'][i])
        f_label = ds['f_data'][i]
        assert len(f_label[:,0]) == num_atom
        net_force_x.append(np.abs(np.sum(f_label[:,0]) / num_atom))
        net_force_y.append(np.abs(np.sum(f_label[:,1]) / num_atom))
        net_force_z.append(np.abs(np.sum(f_label[:,2]) / num_atom))

    msg_list = []
    msg = f"after training on {tfr_file_name} set: Net force per atom (meV/Ang/Atom),"
    msg += f"x={1000*np.mean(net_force_x)},"
    msg += f"y={1000*np.mean(net_force_y)},"
    msg += f"z={1000*np.mean(net_force_z)}."
    msg_list.append(msg)

    # get the true label and predictions
    e_label = []
    f_label = []
    e_pred = []
    f_pred = []
    num_frame = len(ds['e_data'])
    for i in range(0, num_frame):
        num_atom = len(ds['elems'][i])
        e_label.append(ds['e_data'][i] / num_atom)
        f_label.append(ds['f_data'][i])
        atoms = Atoms(numbers=ds['elems'][i],
                        positions=ds['coord'][i],
                        cell=ds['cell'][i],
                        pbc=True)
        atoms.set_calculator(calc)
        e_pred.append(atoms.get_potential_energy() / num_atom)
        f_pred.append(atoms.get_forces())
    e_label = np.array(e_label)
    f_label = np.array(f_label)
    e_pred = np.array(e_pred)
    f_pred = np.array(f_pred)

    weights = np.array(ds['e_weight'])
    # print the emax per atom and fmax on this set
    # based on the code in tips.nf of PiNNAncL, the energy is per atom value
    cases = [
        ("all samples", None),
        ("with weight > 0.99", weights > 0.99),
        ("with weight < 0.1", weights < 0.1),
    ]

    msg_list = [] # comment this line if want to print Net force per atom
    for desc, mask in cases:
        msg = f"after training on {tfr_file_name} set {desc}: "
        msg += compute_metrics(e_pred, e_label, f_pred, f_label, mask)
        msg_list.append(msg)

    with open(ds_abs_path + "append.log", "a") as f:
        for msg in msg_list:
            print(msg, file=f)

