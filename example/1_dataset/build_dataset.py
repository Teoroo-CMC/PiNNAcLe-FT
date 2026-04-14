import os
import numpy as np
from pathlib import Path

WORK_DIR = Path(os.getcwd())

# uniform sampling the ASE trajectory from MACE foundation models and convert to Tensorfow dataset format
def build_dataset_from_mace_traj():
    
    from ase.io.trajectory import Trajectory

    time_span = 0.4 # ps
    data = {'e_data':[], 'f_data':[], 'elems':[], 'coord':[], 'cell':[]}

    for subdir_name in ["MACE-r2scan-330K", "MACE-r2scan-400K", "MACE-r2scan-500K"]:

        traj_file = ""
        log_file = ""
        sub_dir = WORK_DIR / subdir_name
        file_list = os.listdir(sub_dir)
        for idx in range(0, len(file_list)):
            if file_list[idx].endswith("traj"):
                traj_file = sub_dir / file_list[idx]
                log_file = sub_dir / file_list[idx].replace("traj", "log")
                break

        traj_time = None
        traj = Trajectory(traj_file)
        with open(log_file, "r") as f:
            lines = f.readlines()
            traj_time = [float(l.split()[0]) for l in lines[1:]]
            traj_time = np.array(traj_time)

        time_interval = traj_time[1] - traj_time[0]  # ps
        traj_span = int(time_span / time_interval)

        # remove the first snapshot
        for idx in range(1, len(traj_time), traj_span):
            data['coord'].append(traj[idx].get_positions(wrap=True)) # ang
            data['elems'].append(traj[idx].numbers)
            data['cell'].append(traj[idx].get_cell(complete=True))
            data['f_data'].append(traj[idx].calc.results["forces"])  # eV/ang
            data['e_data'].append(traj[idx].calc.results["energy"]) # eV
            print(data['e_data'][-1])

    # save to file
    from pinn.io import load_numpy,write_tfrecord
    data = {k:np.array(v) for k,v in data.items()}
    dataset = load_numpy(data)
    tfr_file = WORK_DIR / "mixProton_330K_400K_500K.yml"
    write_tfrecord(str(tfr_file), dataset)

# 
if __name__ == '__main__':

    build_dataset_from_mace_traj()
