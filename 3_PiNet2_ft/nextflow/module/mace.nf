nextflow.enable.dsl=2

params.publish = 'mace'

process mace {
  tag "$name"
  label 'mace'
  publishDir "$params.publish/$name"

  input:
    tuple val(name), path(input), path(geo)

  output:
    tuple val(name), path('mace.traj'), emit:logs

  script:
    """
python3 - <<- 'EOF'
from mace.calculators import mace_mp
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory

# ------------ patch ase properties to write extra cols --------------------
from ase.calculators.calculator import all_properties
all_properties+=[f'{prop}_{extra}' for prop in ['energy', 'forces', 'stress'] for extra in ['avg','std','bias']]
# --------------------------------------------------------------------------

model_name = ""
device = ""
with open("$input", "r") as f:
    list_lines = f.readlines()
    list_keys = [l.split(":")[0] for l in list_lines]
    list_values = [l.split(":")[1] for l in list_lines]
    name_index = list_keys.index("model_name")
    model_name = list_values[name_index]
    device_index = list_keys.index("device")
    device = list_values[device_index]
model_name = model_name.strip("\\n")
device = device.strip("\\n")
assert model_name != "" and device != ""
    
macemp = mace_mp(model=model_name, device=device)
    
atoms = read("$geo")
atoms.set_calculator(macemp)

energy = atoms.get_potential_energy() # eV
forces = atoms.get_forces() # eV/Ang

aseTraj = Trajectory("mace.traj", mode='w')
aseTraj.write(atoms)
aseTraj.close()
EOF
   """
}

workflow sp {
  take:
    ch // [name, inp, geo]

  main:
    ch | mace

  emit:
    mace.out.logs
}
