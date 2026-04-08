import os
import numpy as np

# uniform sampling the ASE trajectory from MACE foundation models and convert to Tensorfow dataset format
def build_dataset_from_mace_traj():
    
    from ase.io.trajectory import Trajectory
    from setting import proj_params

    data = []
    time_span = 0.4 # ps
    for subdir_name in ["1_MACE-r2scan-330K", "2_MACE-r2scan-400K", "3_MACE-r2scan-500K"]:

        traj_file = ""
        log_file = ""
        sub_dir = proj_params["dataset_dir"] / subdir_name
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
            frame = {}
            frame['coord'] = traj[idx].get_positions(wrap=True) # ang
            frame['elems'] = traj[idx].get_chemical_symbols()
            frame['cell'] = traj[idx].get_cell(complete=True)
            frame['f_data'] = traj[idx].calc.results["forces"]  # eV/ang
            frame['e_data'] = traj[idx].calc.results["energy"] # eV
            print(frame['e_data'])
            data.append(frame)

        print(subdir_name, "Number of snapshots = %d"%(len(data)))

    # save to file
    from pinn.io import load_numpy,write_tfrecord
    dataset = load_numpy(data)
    tfr_file = proj_params["dataset_dir"] / "mix_dataset" / "mixProton_330K_400K_500K.yml"
    write_tfrecord(tfr_file, dataset)

# 
if __name__ == '__main__':

    build_dataset_from_mace_traj()
