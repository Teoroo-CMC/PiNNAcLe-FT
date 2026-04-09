#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Input files
params.md_init = "./geo/*.vasp"

// Other parameters
params.md_ps = 1000         // length of NVT simulation

model_config = Channel.fromPath(params.md_init)
               .combine(Channel.of(330,400,500))

workflow {
    pinn_nvt(model_config)
}

def shorten(x) {sprintf('%.5E', (double) x).replaceAll(/\.?0*E/, 'E').replaceAll(/E\+0*/, 'E')}

process pinn_nvt {
    publishDir "mds", mode: 'link'
    label 'pinn'

    input:
    tuple (file(md_geo),val(temp))

    output:
    file "NVT-${temp}K-${params.md_ps}ps-${md_geo.simpleName}.log"
    file "NVT-${temp}K-${params.md_ps}ps-${md_geo.simpleName}.traj"

    """
    #!/usr/bin/env python3
    from mace.calculators import mace_mp
    from ase import units
    from ase.io import read
    from ase.io.trajectory import Trajectory
    from ase.md import MDLogger
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.bussi import Bussi

    atoms = read("$md_geo", format='vasp')
    macemp = mace_mp(model="mace-matpes-r2scan-0", device="cuda")
    atoms.set_calculator(macemp)

    dTemp = $temp
    MaxwellBoltzmannDistribution(atoms, dTemp*units.kB)
    dt = 0.5 * units.fs
    # 0.2 ps was selected based on Table 1 in https://arxiv.org/pdf/0803.4060
    dyn = Bussi(atoms, timestep=dt, temperature_K=dTemp, taut=dt*400)

    dyn.attach(
        MDLogger(dyn, atoms, 'NVT-${temp}K-${params.md_ps}ps-${md_geo.simpleName}.log',stress=True, mode="w"),
        interval=int(200*units.fs/dt))
    dyn.attach(
        Trajectory('NVT-${temp}K-${params.md_ps}ps-${md_geo.simpleName}.traj', 'w', atoms).write,
        interval=int(200*units.fs/dt))
    for i in range($params.md_ps):
        dyn.run(int(1e3*units.fs/dt))
 """
}
