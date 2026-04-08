import os
import numpy as np
from pathlib import Path
from collections import namedtuple
from setting import proj_params

# for analyze to name the trajectorys, note 
MD_TIME = "5000ps"
MD_SYSTEM = "wat64_h3o+" # wat64_h3o+
MD_TEMPERATURE = "330K"

# subdir (for different product runs)
TRAJ_DIR = proj_params["equ_md_dir"] / "mds" / "PiNet2_Seed1_ft"
TRAJ_FORMAT = "aseTraj" # aseTraj, xyz, pdb (from Gromacs)

# define bonds and angels
# MAX_DIST_HO_BOND = 1.20  J. Phys. Chem. C 2018, 122, 26965−26973
# MAX_DIST_HO_BOND = 1.23, Precis. Chem. 2024, 2, 12, 644–654
# MAX_DIST_HO_BOND = 1.25, Nature Communications | (2023)14:6131. also confirmed by the test results of MACE
# MAX_DIST_HO_BOND = 1.24, NAtuRe CHeMiStRY | VOL 10 | APRIL 2018 | 413–419, https://doi.org/10.1038/s41557-018-0010-2
MAX_DIST_HO_BOND = 1.5   # the distance between O and H in water molecules, suggested by Chao. > 1.27 in 7Net trajectory
# DIST_HH_THRESH = 3.5     # the distance to define H bond, get from Nature Communications | ( 2023) 14:6131 and J. Phys. Chem. C 2018, 122, 26965−26973
ANGEL_H_BOND = 140   # the O-H-O angle to define H bond, get from J. Phys. Chem. C 2018, 122, 26965−26973

# define species
OXY_ATOM_SPECIES = ["OH3", "OH2", "OH", "O"]

# use numba to acclerate the get_all_distances(mic=True)
from numba import njit

@njit()
def mic_distance_single(diff, cell, inv_cell):
    frac = diff @ inv_cell
    frac = frac - np.round(frac)
    diff_mic = frac @ cell
    return np.linalg.norm(diff_mic)

@njit()
def mic_distance_matrix(pos, cell, inv_cell):
    N = len(pos)
    dist_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            diff = pos[i] - pos[j]
            dist = mic_distance_single(diff, cell, inv_cell)
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat

# 
def cal_water_density():

    cell = [[15.66, 0.0000000000, 0.0000000000],
             [0.0000000000, 15.66, 0.0000000000],
             [0.0000000000, 0.0000000000, 15.66]]

    volume = np.abs(np.linalg.det(cell))
    volume /= 1e24  # Ang3 to cm3

    oxy_num = 128
    mass = oxy_num * 18.0154 / 6.022e23 # g

    density = mass / volume # g.cm-3

    print(density)


# read the trajectory file, remove the equilibrium time
def load_trajectory():

    # remove the equilibrium time
    # note: when computing the MSD of OH3, the traj_span should be very small, so that the moved distance of proton < L/2
    # to make sure this, set traj_span = 1 in each case
    traj_span = 1
    equilibrium_time = 0.0 # ps
    if float(MD_TIME.strip("ps")) > 2000.0:  # MLMD
        traj_span = 1
        equilibrium_time = 100.0
    elif float(MD_TIME.strip("ps")) > 190.0:  # MLMD
        traj_span = 1
        equilibrium_time = 100.0
    elif float(MD_TIME.strip("ps")) > 90.0:  # MLMD
        traj_span = 1
        equilibrium_time = 20.0
    else: #DFT-MD
        traj_span = 1
        equilibrium_time = 1.0

    traj_obj = None
    traj_time = []

    if TRAJ_FORMAT == "aseTraj": # MLMD

        traj_file = "NVT-%s-%s-%s" % (MD_TEMPERATURE, MD_TIME, MD_SYSTEM)
        with open(TRAJ_DIR / f"{traj_file}.log", "r") as f:
            lines = f.readlines()
            traj_time = [float(l.split()[0]) for l in lines[1:]]
            traj_time = np.array(traj_time)

        from ase.io.trajectory import Trajectory
        traj_obj = Trajectory(TRAJ_DIR / f"{traj_file}.traj")

    elif TRAJ_FORMAT == "pdb": # classical MD from Gromacs

        from ase.io import read
        traj_file = TRAJ_DIR / "traj_unwrap.pdb"
        with open(traj_file, "r") as g:
            lines = g.readlines()
            for line in lines:
                if "TITLE" not in line:
                    continue
                items = line.strip().split()
                traj_time.append(float(items[-3])) # ps
        traj_time = np.array(traj_time)
        traj_obj = read(traj_file, index=':')
        assert len(traj_time) == len(traj_obj)

    elif TRAJ_FORMAT == "xyz": # from cp2k

        cell = []
        with open(TRAJ_DIR / f"{MD_SYSTEM}.vasp", "r") as f:
            lines = f.readlines()
            x_vector = [float(j) for j in lines[2].strip().split()]
            y_vector = [float(j) for j in lines[3].strip().split()]
            z_vector = [float(j) for j in lines[4].strip().split()]
            cell.append(x_vector)
            cell.append(y_vector)
            cell.append(z_vector)
        cell = np.array(cell)

        with open(TRAJ_DIR / "proton-water-pos-1.xyz", "r") as g:
            lines = g.readlines()
            line_num_per_frame = int(lines[0].strip()) + 2
            assert len(lines) % line_num_per_frame == 0
            for index in range(1, len(lines), line_num_per_frame):
                assert "time" in lines[index]
                items = lines[index].strip().split(",")
                traj_time.append(float(items[1].split("=")[1])/1000.0) # fs to ps
        traj_time = np.array(traj_time)

        from ase.io import read
        traj_obj = read(TRAJ_DIR / "proton-water-pos-1.xyz", index=':', format="xyz")
        for atoms in traj_obj:
            atoms.set_cell(cell)
            atoms.set_pbc((True, True, True))

    print("total number of snapshots = %d"%(len(traj_obj)))

    # remove the equilibrium snapshots, and shift the time to zero
    time_interval = traj_time[1] - traj_time[0]  # ps
    print(equilibrium_time)
    first_index = round(equilibrium_time / time_interval)
    traj_time = traj_time[first_index::traj_span]
    traj_time -= np.min(traj_time)
    traj_obj = traj_obj[first_index::traj_span]
    traj_obj = list(traj_obj) # to list for modification

    print("used number of snapshots = %d"%(len(traj_obj)))

    # if using Bussi, then the COM not fixed, remove the COM
    for frame_index in range(0, len(traj_obj)):
        atoms = traj_obj[frame_index].copy()
        atoms.positions -= atoms.get_center_of_mass()
        atoms.positions.flags.writeable = False # make is unwritable
        atoms.cell.array.flags.writeable = False
        traj_obj[frame_index] = atoms

    print("the COM has been shift to zero, for MSD calculations")

    return traj_obj, traj_time

# analyse the species in one snapshot
def find_all_species(ASEAtoms):

    symbols = ASEAtoms.get_chemical_symbols()
    hyd_atom_index_list = [index for index in range(0, len(symbols)) if symbols[index] == "H"]
    oxy_atom_index_list = [index for index in range(0, len(symbols)) if symbols[index] == "O"]

    # fast version
    cell = ASEAtoms.cell[:]
    inv_cell = np.linalg.inv(cell)
    atom_pair_dist = mic_distance_matrix(ASEAtoms.positions, cell, inv_cell)

    # define proton ref Chem. Sci.,2018, 9,7126–7132 (DOI: 10.1039/c8sc01253a)
    # first: assign two closest hydrogens to one Oxygen
    assigned_hyd_atom_list = []
    oxy_to_hyd_atoms = dict([(k, []) for k in oxy_atom_index_list])
    for oxy_atom_index in oxy_atom_index_list:
        oxy_hyd_dist = atom_pair_dist[oxy_atom_index][hyd_atom_index_list]
        sorted_idx = np.argsort(oxy_hyd_dist)
        if oxy_hyd_dist[sorted_idx[0]] < MAX_DIST_HO_BOND:
            oxy_to_hyd_atoms[oxy_atom_index].append(hyd_atom_index_list[sorted_idx[0]])
            assigned_hyd_atom_list.append(hyd_atom_index_list[sorted_idx[0]])
        if oxy_hyd_dist[sorted_idx[1]] < MAX_DIST_HO_BOND:
            oxy_to_hyd_atoms[oxy_atom_index].append(hyd_atom_index_list[sorted_idx[1]])
            assigned_hyd_atom_list.append(hyd_atom_index_list[sorted_idx[1]])
        # sometimes the water molecules could be dissociated, i.e., the OH- is existing
        # assign the third nearest hydrogen to this oxygen
        if len(oxy_to_hyd_atoms[oxy_atom_index]) == 1:
            oxy_to_hyd_atoms[oxy_atom_index].append(hyd_atom_index_list[sorted_idx[2]])
            assigned_hyd_atom_list.append(hyd_atom_index_list[sorted_idx[2]])
        elif len(oxy_to_hyd_atoms[oxy_atom_index]) == 0:
            raise ValueError("an isolated oxygen atoms!")

    # second: assign the last hydrogen as proton, and linked it to the nearest Oxygen
    other_hyd_atom_index = [x for x in hyd_atom_index_list if x not in assigned_hyd_atom_list]
    if len(other_hyd_atom_index) > 0:
        assert len(other_hyd_atom_index) == 1
        hyd_oxy_dist = atom_pair_dist[other_hyd_atom_index[0]][oxy_atom_index_list]
        min_dist_index = np.argmin(hyd_oxy_dist)
        assert hyd_oxy_dist[min_dist_index] < MAX_DIST_HO_BOND
        nearest_oxy_atom_index = oxy_atom_index_list[min_dist_index]
        oxy_to_hyd_atoms[nearest_oxy_atom_index].append(other_hyd_atom_index[0])

    # O atoms
    species_to_atom_index = {species: [] for species in OXY_ATOM_SPECIES}
    for oxy_atom_index in oxy_atom_index_list:

        cur_species = "None"
        link_hyd_num = len(oxy_to_hyd_atoms[oxy_atom_index])
        if link_hyd_num == 0:
            cur_species = "O"
        elif link_hyd_num == 1:
            cur_species = "OH"
        elif link_hyd_num == 2:
            cur_species = "OH2"
        elif link_hyd_num == 3:
            cur_species = "OH3"
        else:
            print("Error: oxy_index=%d, link_hyd_num = %d"%(oxy_atom_index, link_hyd_num))
            assert 1 == 0

        if cur_species not in OXY_ATOM_SPECIES:
            print("Error", cur_species)
            assert 1 == 0

        species_to_atom_index[cur_species].append(oxy_atom_index)

    # get the proton atom index, which is supposed to have the largest dist to its neghbour O
    # see J. Chem. Phys. 157, 024503 (2022); doi: 10.1063/5.0094944, H* is all hydrogen in H3O*
    proton_atom_index = None
    hyd_atom_index_in_H3O = []
    OH3_atom_index_list = species_to_atom_index["OH3"]
    if len(OH3_atom_index_list) > 0:
        assert len(OH3_atom_index_list) == 1
        hyd_atom_index_in_H3O = oxy_to_hyd_atoms[OH3_atom_index_list[0]]
        assert len(hyd_atom_index_in_H3O) == 3
        proton_atom_index = other_hyd_atom_index[0]
        assert proton_atom_index in hyd_atom_index_in_H3O

    # return a namedtuple
    Result = namedtuple("Result", ["species_to_atom_index","atom_pair_dist",
                                  "hyd_atom_index_in_H3O","proton_atom_index",])
    
    result = Result(species_to_atom_index, atom_pair_dist,
                    hyd_atom_index_in_H3O, proton_atom_index)
    
    return result

# 
def analyse_species_traj(traj_obj):
    # get all species
    traj_res = []
    for frame_index in range(0, len(traj_obj)):
        print("analysing species: %d"%(frame_index))
        traj_res.append(find_all_species(traj_obj[frame_index]))
    assert len(traj_obj) == len(traj_res)
    return traj_res

# yunqi, https://github.com/Teoroo-CMC/blyp_benchmark/blob/master/rdf_mds.nf
# https://docs.mdanalysis.org/1.1.1/documentation_pages/analysis/rdf.html
# https://github.com/cndaqiang/Radial_Distribution_Function_rdf/tree/main
# https://github.com/by256/rdfpy/blob/master/rdfpy/rdfpy.py
def cal_rdf_by_atom_zzy(traj_obj, species1, species2):

    import itertools

    # r_max should smaller than L/2, otherwise the RDF located in far distance will not equal to 1
    # r_max = np.min([traj[0].cell[0][0],traj[0].cell[1][1],traj[0].cell[1][1]]) / 2.0
    r_min = 0.0
    r_max = 6.0
    bin_num = 201
    bins = np.linspace(r_min, r_max, bin_num)
    r_mid = (bins[1:] + bins[:-1])/2
    bin_vol = (bins[1:]**3 - bins[:-1]**3)*4*np.pi/3

    hist = 0
    count = 0
    for frame_index in range(0, len(traj_obj)):

        print(frame_index)

        # move all atoms into one cell
        # based on my test, wrap or unwrap is no influence on the rdfs
        data = traj_obj[frame_index]

        # atom_pair_dist = data.get_all_distances(mic=True)

        # fast version
        cell = data.cell[:]
        inv_cell = np.linalg.inv(cell)
        atom_pair_dist = mic_distance_matrix(data.positions, cell, inv_cell)

        count += 1
        index_ij = ""
        if species1 == species2:
            listIndex1 = [idx for idx in range(len(data)) if data[idx].symbol == species1]
            index_ij = np.array(list(itertools.combinations(listIndex1, 2)))
        else:
            listIndex1 = [idx for idx in range(len(data)) if data[idx].symbol == species1]
            listIndex2 = [idx for idx in range(len(data)) if data[idx].symbol == species2]
            index_ij = np.array(list(itertools.product(listIndex1, listIndex2)))

        # original
        # diff = data.positions[index_ij[:, 0]] - data.positions[index_ij[:, 1]]
        # # the distance in each direction should smaller than half of cell length
        # # so np.rint is used here rather than np.trunc
        # listCellLen = [data.cell[0][0], data.cell[1][1], data.cell[2][2]]
        # diff = diff - np.rint(diff / listCellLen) * listCellLen
        # dist = np.linalg.norm(diff, axis=1)

        dist = [atom_pair_dist[item[0]][item[1]] for item in index_ij]

        h, edges = np.histogram(dist, bins)

        # normalize by the number density
        rho = len(dist) / data.get_volume()
        hist += h / bin_vol / rho

    rdf = hist/count

    rdf_file = TRAJ_DIR / f"{MD_SYSTEM}-{species1+species2}-atom-bins{bin_num}-zzy.rdf"
    if os.path.exists(rdf_file):
        os.remove(rdf_file)
    np.savetxt(rdf_file, np.stack([r_mid, rdf], axis=1), delimiter=";")

    return rdf_file

# analyse the rdf based on the species
def cal_rdf_by_species_zzy(traj_obj, traj_res, species1, species2):

    import itertools

    r_min = 0.0
    r_max = 6.0
    bin_num = 201
    bins = np.linspace(r_min,r_max,bin_num)
    r_mid = (bins[1:] + bins[:-1])/2
    bin_vol = (bins[1:]**3 - bins[:-1]**3)*4*np.pi/3

    # designed for liquid water (proton) system
    symbols = traj_obj[0].get_chemical_symbols()
    hyd_atom_list = [index for index in range(0, len(symbols)) if symbols[index] == "H"]

    hist = 0
    count = 0
    for frame_index in range(0, len(traj_obj)):

        print(frame_index)

        # move all atoms into one cell
        # based on my test, wrap or unwrap is no influence on the rdfs
        data = traj_obj[frame_index]

        # get all species
        res = traj_res[frame_index]
        hyd_atom_list_HO2 = [index for index in hyd_atom_list if index not in res.hyd_atom_index_in_H3O]

        atom_index_species1 = []
        atom_index_species2 = []
        if species1 == "O+": # hydronium oxygen
            atom_index_species1 = res.species_to_atom_index["OH3"]
        elif species1 == "O": # water oxygen
            atom_index_species1 = res.species_to_atom_index["OH2"]
        elif species1 == "H+": # hydronium hydrogen
            atom_index_species1 = res.hyd_atom_index_in_H3O
        elif species1 == "H": # water hydrogen
            atom_index_species1 = hyd_atom_list_HO2
        else:
            assert 1 == 0

        if species2 == "O+": # hydronium oxygen
            atom_index_species2 = res.species_to_atom_index["OH3"]
        elif species2 == "O": # water oxygen
            atom_index_species2 = res.species_to_atom_index["OH2"]
        elif species2 == "H+": # hydronium hydrogen
            atom_index_species2 = res.hyd_atom_index_in_H3O
        elif species2 == "H": # water hydrogen
            atom_index_species2 = hyd_atom_list_HO2
        else:
            assert 1 == 0

        count += 1
        index_ij = ""
        if species1 == species2:
            index_ij = np.array(list(itertools.combinations(atom_index_species1, 2)))
        else:
            index_ij = np.array(list(itertools.product(atom_index_species1, atom_index_species2)))

        dist = res.atom_pair_dist[index_ij[:,0], index_ij[:,1]]
        h, edges = np.histogram(dist, bins)

        # normalize by the number density
        rho = len(dist) / data.get_volume()
        hist += h / bin_vol / rho

    rdf = hist/count

    rdf_file = TRAJ_DIR / f"{MD_SYSTEM}-{species1+species2}-species-bins{bin_num}-zzy.rdf"
    if os.path.exists(rdf_file):
        os.remove(rdf_file)

    np.savetxt(rdf_file, np.stack([r_mid, rdf], axis=1), delimiter=";")

    return rdf_file

# calculate the number of concerned species
def cal_species_atom_number(traj_obj, traj_time, traj_res):

    # get total number of Oxygen atoms
    symbols = traj_obj[0].get_chemical_symbols()
    oxy_atom_list = [nAtomIndex for nAtomIndex in range(0, len(symbols)) if symbols[nAtomIndex] == "O"]

    #
    atom_num_OH3 = []
    atom_num_OH2 = []
    atom_num_OH = []
    atom_num_O = []
    for frame_index in range(0, len(traj_obj)):

        num_OH3 = 0
        num_OH2 = 0
        num_OH = 0
        num_O = 0
        species_to_atom_index = traj_res[frame_index].species_to_atom_index
        for key, value in species_to_atom_index.items():
            if key == "OH3":
                num_OH3 += len(value)  # charge 1
            elif key == "OH2":
                num_OH2 += len(value)  # charge 0
            elif key == "OH":
                num_OH += len(value)  # charge -1
            elif key == "O":
                num_O += len(value)  # charge -2
            else:
                assert 1 == 0

        atom_num_OH3.append(num_OH3)
        atom_num_OH2.append(num_OH2)
        atom_num_OH.append(num_OH)
        atom_num_O.append(num_O)

    output_file = TRAJ_DIR / f"{MD_SYSTEM}.number"
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w") as f:
        f.write("time(ps);Num(OH3);Num(OH2);Num(OH);Num(O);Sum\n")
        for frame_index in range(0, len(traj_obj)):
            total_number = atom_num_OH3[frame_index] + atom_num_OH2[frame_index]\
                           + atom_num_OH[frame_index] + atom_num_O[frame_index]
            assert total_number == len(oxy_atom_list)
            line = "%f;%d;%d;%d;%d;%d\n" % (traj_time[frame_index], atom_num_OH3[frame_index],
                    atom_num_OH2[frame_index], atom_num_OH[frame_index], atom_num_O[frame_index],
                    total_number)
            f.write(line)

    return output_file

# 
def cal_msd_by_atom_zzy(traj_obj, traj_time, species):

    print("!!!Designed for NVT and unwraped trajctory!!!")

    symbols = traj_obj[0].get_chemical_symbols()
    atom_index_list = [index for index in range(0, len(symbols)) if symbols[index] == species]

    cell_volume = []
    atom_pos_all_frame = []
    for frame_index in range(0, len(traj_obj)):

        print("Stage 1: %d"%(frame_index))
        atoms = traj_obj[frame_index]
        cell_volume.append(atoms.get_volume())

        atom_pos = atoms.get_positions(wrap=False)
        atom_pos_all_frame.append(atom_pos[atom_index_list])

    atom_pos_all_frame = np.array(atom_pos_all_frame)

    # calculate the mean squared displacement(MSD) and Nernst-Einstein conductivity
    # https://manual.gromacs.org/current/reference-manual/analysis/mean-square-displacement.html
    # $$\frac{1}{k_{\mathrm{B}} T} \sum_{\alpha} q_{\alpha}^{2} \rho_{\alpha} D_{\alpha}^{\mathrm{s}}$$
    import tidynamics

    atom_msd = []
    lag_time = []
    for atom_index in range(0, len(atom_index_list)):
        cur_msd = tidynamics.msd(atom_pos_all_frame[:, atom_index, :])
        cur_lag_time = np.arange(len(cur_msd)) * (traj_time[1]-traj_time[0])
        atom_msd.append(cur_msd)
        lag_time.append(cur_lag_time)
    species_msd = np.mean(atom_msd, axis=0) # average over all same ions, ang^2
    species_lag_time = np.mean(lag_time, axis=0) 

    msd_file = TRAJ_DIR / f"{MD_SYSTEM}-{MD_TEMPERATURE}-{species}-zzy.msd"
    if os.path.exists(msd_file):
        os.remove(msd_file)
    lag_time_msd = np.vstack([species_lag_time, species_msd])  # ps
    np.savetxt(msd_file, lag_time_msd.T, delimiter=";")

    return msd_file

# check the largest movement distance of all particles in one time steps
def cal_particle_moving_dist(traj_obj, traj_time, species):

    print("!!!Designed for NVT and unwraped trajctory!!!")

    symbols = traj_obj[0].get_chemical_symbols()
    atom_index_list = [index for index in range(0, len(symbols)) if symbols[index] == species]

    max_moving_dist = []
    for frame_index in range(1, len(traj_obj)):

        print("Stage 1: %d"%(frame_index))

        # if using Bussi, then the COM not fixed, remove the COM
        # removeCOM() is already applied in load_trajectory
        atoms1 = traj_obj[frame_index-1]
        atom_pos_1 = atoms1.get_positions(wrap=False)

        atoms2 = traj_obj[frame_index]
        atom_pos_2 = atoms2.get_positions(wrap=False)

        vec_dist = atom_pos_2[atom_index_list] - atom_pos_1[atom_index_list]
        scalar_dist = np.linalg.norm(vec_dist, axis=1)

        max_moving_dist.append(np.max(scalar_dist))

    # save
    move_file = TRAJ_DIR / f"{MD_SYSTEM}-{MD_TEMPERATURE}-{species}.move"
    if os.path.exists(move_file):
        os.remove(move_file)
    lag_time_msd = np.vstack([max_moving_dist])  # ps
    np.savetxt(move_file, lag_time_msd.T, delimiter=";")

    # it should be smaller than L/2
    max_cell_length = np.max(traj_obj[0].cell)
    if np.max(max_moving_dist) > max_cell_length/2:
        print("the moving distance of a particle (%f) is larger than L(%f)/2"%(np.max(max_moving_dist), max_cell_length))
        print("Please use a smaller span number in load_trajectory")
        assert 1 == 0

    return move_file

#
def cal_proton_coord_component(traj_obj, traj_time, traj_res):

    print("!!!Designed for NVT and unwraped trajctory!!!")
    print("!!!Suppose the system only includes one excess proton!!!")

    proton_coord_list = []
    for frame_index in range(0, len(traj_obj)):

        print("Stage 2: %d"%(frame_index))

        # removeCOM() is already applied in load_trajectory
        atom_pos = traj_obj[frame_index].get_positions(wrap=False)
        res = traj_res[frame_index]
        OH3_atom_index = res.species_to_atom_index["OH3"][0]
        # print(atom_pos[OH3_atom_index])
        proton_coord_list.append(atom_pos[OH3_atom_index])

    # save
    move_file = TRAJ_DIR / f"{MD_SYSTEM}-{MD_TEMPERATURE}-OH3-before.coord"
    if os.path.exists(move_file):
        os.remove(move_file)
    time_coord = np.column_stack((traj_time, proton_coord_list))  # ps
    np.savetxt(move_file, time_coord, delimiter=";")

# traj_obj is the ase trajectory
# traj_time is the simulation time (Delta t = 0.2 ps)
# traj_res is the analyse results of species = (species_to_atom_index, atom_pair_dist, hyd_atom_index_in_H3O, proton_atom_index)
# species_to_atom_index is a dict: {"OH2":atom_idx_water; "OH3":atom_idx_OH3}
# atom_pair_dist is the distance matrix between atom pairs
# hyd_atom_index_in_H3O: the atom idx of hydrogen atoms in OH3
# proton_atom_index: the atom idx of proton (with the longest bond length to the central oxygen in hyd_atom_index_in_H3O)
def cal_msd_proton_zzy(traj_obj, traj_time, traj_res, position_type):

    print("!!!Designed for NVT and unwraped trajctory!!!")
    print("!!!Suppose the system only includes one excess proton!!!")

    m_O = 15.999
    m_H = 1.008
    cell = traj_obj[0].cell

    # create the continuos coordinate of proton
    # vehicular_atom_pos_all_frame = [np.array([0.0, 0.0, 0.0])]
    # structural_atom_pos_all_frame = [np.array([0.0, 0.0, 0.0])]
    moving_dist = [0.0]
    total_atom_pos_all_frame = [np.array([0.0, 0.0, 0.0])]
    for frame_index in range(1, len(traj_obj)):

        print("Stage 2: %d"%(frame_index))

        # removeCOM() is already applied in load_trajectory
        atom_pos = traj_obj[frame_index].get_positions(wrap=False)
        prev_atom_pos = traj_obj[frame_index-1].get_positions(wrap=False)

        # get the atom index belonging to this species
        res = traj_res[frame_index]
        prev_res = traj_res[frame_index-1]
        assert len(res.species_to_atom_index["OH3"]) > 0
        assert len(prev_res.species_to_atom_index["OH3"]) > 0

        # get position of proton
        # we need the continuous proton coordinate, so DO NOT directly use the unwrap position of Oxygen atoms
        # \mathbf{r}_p(t+\Delta t) = \mathbf{r}_p(t) + \text{minimum_image}\Big(\mathbf{O}_{\text{new}}(t+\Delta t) - \mathbf{O}_{\text{old}}(t)\Big)
        delta_r = np.array([0.0, 0.0, 0.0])
        if position_type == "nearestOxyg":
            # based on the oxygen atom in H3O+, J. Chem. Phys. 123, 044505 (2005), https://doi.org/10.1063/1.1961443
            OH3_atom_index = res.species_to_atom_index["OH3"][0]
            prev_OH3_atom_index = prev_res.species_to_atom_index["OH3"][0]
            delta_r = atom_pos[OH3_atom_index] - prev_atom_pos[prev_OH3_atom_index]
        elif position_type == "longestDist":
            proton_atom_index = res.proton_atom_index
            prev_proton_atom_index = prev_res.proton_atom_index
            delta_r = atom_pos[proton_atom_index] - prev_atom_pos[prev_proton_atom_index]
        elif position_type == "com":
            OH3_atom_index = res.species_to_atom_index["OH3"][0]
            H_pos_mic = []
            assert len(atom_pos[res.hyd_atom_index_in_H3O]) == 3
            for h in atom_pos[res.hyd_atom_index_in_H3O]:
                dr = h - atom_pos[OH3_atom_index]
                dr -= cell.T @ np.round(np.linalg.solve(cell.T, dr)) # minimal image dist
                H_pos_mic.append(atom_pos[OH3_atom_index] + dr)
            H_pos_mic = np.array(H_pos_mic)
            com = (m_O * atom_pos[OH3_atom_index] + m_H * np.sum(H_pos_mic, axis=0))
            com /= (m_O + 3*m_H)

            prev_OH3_atom_index = prev_res.species_to_atom_index["OH3"][0]
            prev_H_pos_mic = []
            assert len(prev_atom_pos[prev_res.hyd_atom_index_in_H3O]) == 3
            for h in prev_atom_pos[prev_res.hyd_atom_index_in_H3O]:
                prev_dr = h - prev_atom_pos[prev_OH3_atom_index]
                prev_dr -= cell.T @ np.round(np.linalg.solve(cell.T, prev_dr)) # minimal image dist
                prev_H_pos_mic.append(prev_atom_pos[prev_OH3_atom_index] + prev_dr)
            prev_H_pos_mic = np.array(prev_H_pos_mic)
            prev_com = (m_O * prev_atom_pos[prev_OH3_atom_index] + m_H * np.sum(prev_H_pos_mic, axis=0))
            prev_com /= (m_O + 3*m_H)

            delta_r = com - prev_com

        delta_r -= cell.T @ np.round(np.linalg.solve(cell.T, delta_r)) # minimal image dist
        moving_dist.append(np.linalg.norm(delta_r))
        total_atom_pos_all_frame.append(total_atom_pos_all_frame[-1] + delta_r)

        # split into the vehicular diffusion and structural diffusion
        # if OH3_atom_index == prev_OH3_atom_index:
        #     vehicular_atom_pos_all_frame.append(vehicular_atom_pos_all_frame[-1] + delta_r)
        #     structural_atom_pos_all_frame.append(structural_atom_pos_all_frame[-1])
        # else:
        #     vehicular_atom_pos_all_frame.append(vehicular_atom_pos_all_frame[-1])
        #     structural_atom_pos_all_frame.append(structural_atom_pos_all_frame[-1] + delta_r) 

        # note: also can be based on the identifid proton (res.proton_atom_index), J. Chem. Phys. 123, 044505 (2005), https://doi.org/10.1063/1.1961443
        # note: based on Chatgpt, the MSD calculated by H+ includes both of structural diffusion and vibrational diffusion, which mey lead to bad defination

    # vehicular_atom_pos_all_frame = np.array(vehicular_atom_pos_all_frame, dtype=float)
    # structural_atom_pos_all_frame = np.array(structural_atom_pos_all_frame, dtype=float)
    total_atom_pos_all_frame = np.array(total_atom_pos_all_frame, dtype=float)

    # save the movement distance of OH3
    # move_file = TRAJ_DIR / f"{MD_SYSTEM}-{MD_TEMPERATURE}-OH3.move"
    # if os.path.exists(move_file):
    #     os.remove(move_file)
    # np.savetxt(move_file, moving_dist, delimiter=";")

    # save coordiantes of OH3 after minimal image correction
    # coord_file = TRAJ_DIR / f"{MD_SYSTEM}-{MD_TEMPERATURE}-OH3-after.coord"
    # if os.path.exists(coord_file):
    #     os.remove(coord_file)
    # time_coord = np.column_stack((traj_time, total_atom_pos_all_frame, moving_dist))  # ps
    # np.savetxt(coord_file, time_coord, delimiter=";")

    # calculate the mean squared displacement(MSD) and Nernst-Einstein conductivity
    # https://manual.gromacs.org/current/reference-manual/analysis/mean-square-displacement.html
    # $$\frac{1}{k_{\mathrm{B}} T} \sum_{\alpha} q_{\alpha}^{2} \rho_{\alpha} D_{\alpha}^{\mathrm{s}}$$
    import tidynamics

    # species_msd = tidynamics.msd(vehicular_atom_pos_all_frame)
    # species_lag_time = np.arange(len(species_msd)) * (traj_time[1]-traj_time[0])
    # msd_file = TRAJ_DIR / f"{MD_SYSTEM}-{MD_TEMPERATURE}-{species}-zzy-vehic.msd"
    # if os.path.exists(msd_file):
    #     os.remove(msd_file)
    # lag_time_msd = np.vstack([species_lag_time, species_msd])  # ps
    # np.savetxt(msd_file, lag_time_msd.T, delimiter=";")

    # species_msd = tidynamics.msd(structural_atom_pos_all_frame)
    # msd_file = TRAJ_DIR / f"{MD_SYSTEM}-{MD_TEMPERATURE}-{species}-zzy-struc.msd"
    # if os.path.exists(msd_file):
    #     os.remove(msd_file)
    # lag_time_msd = np.vstack([species_lag_time, species_msd])  # ps
    # np.savetxt(msd_file, lag_time_msd.T, delimiter=";")

    species_msd = tidynamics.msd(total_atom_pos_all_frame)
    species_lag_time = np.arange(len(species_msd)) * (traj_time[1]-traj_time[0])
    msd_file = TRAJ_DIR / f"{MD_SYSTEM}-{MD_TEMPERATURE}-proton-zzy-{position_type}.msd"
    if os.path.exists(msd_file):
        os.remove(msd_file)
    lag_time_msd = np.vstack([species_lag_time, species_msd])  # ps
    np.savetxt(msd_file, lag_time_msd.T, delimiter=";")

    return msd_file

#
if __name__ == '__main__':

    # cal_water_density()

    ## Step 1: load trajectory
    traj_obj, traj_time = load_trajectory()

    ## Step 2: analyze the initial MD_SYSTEM, group the atoms
    traj_res = analyse_species_traj(traj_obj)

    ## Step 3: analyze the trajectory

    # cal_rdf_by_atom_zzy(traj_obj, species1="O", species2="O")
    # cal_rdf_by_atom_zzy(traj_obj, species1="O", species2="H")

    # cal_rdf_by_species_zzy(traj_obj, traj_res, species1="H+", species2="O")
    # cal_rdf_by_species_zzy(traj_obj, traj_res, species1="O+", species2="H")
    # cal_rdf_by_species_zzy(traj_obj, traj_res, species1="O+", species2="O")
    # cal_rdf_by_species_zzy(traj_obj, traj_res, species1="O", species2="O")

    # cal_species_atom_number(traj_obj, traj_time, traj_res)

    ## Step 4: MSD and PMSD
    # cal_msd_by_atom_zzy(traj_obj, traj_time, species="O")
    cal_msd_proton_zzy(traj_obj, traj_time, traj_res, "nearestOxyg")
