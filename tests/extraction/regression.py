#!/usr/bin/env python3
"""Integration regression for the real TIPS convert process (no research data)."""
import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile


def generate(path):
    import numpy as np
    from ase import Atoms
    from ase.io.trajectory import Trajectory
    from ase.calculators.singlepoint import SinglePointCalculator
    with Trajectory(str(path), 'w') as out:
        for i in range(60):
            atoms = Atoms('OH2', positions=[[2, 2, 2], [2.96, 2, 2],
                                           [1.76, 2.93, 2]], cell=[12]*3, pbc=True)
            atoms.positions += np.array([i * .001, 0, 0])
            # Synthetic metadata required by the TIPS ASE reader; not physical labels.
            atoms.calc = SinglePointCalculator(atoms, energy=float(i), forces=np.zeros((3, 3)))
            out.write(atoms)


def trace(path, expected):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    assert len(rows) == 1 and rows[0]['status'] == expected, rows


def outputs(directory):
    files = sorted((directory / 'output').rglob('*.xyz'))
    assert len(files) == 5, files
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate-only', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--nextflow', default='nextflow')
    parser.add_argument('--config', type=Path, help='Container/environment configuration')
    parser.add_argument('--override', type=Path, help='Optional scheduled-extraction override')
    parser.add_argument('--output', type=Path, required=False)
    args = parser.parse_args()
    if args.generate_only:
        generate(args.generate_only)
        return
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='extraction-test-'))
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('--output must be empty; use a fresh directory')
    fixture = out / 'fixture.traj'
    if args.fixture:
        shutil.copyfile(args.fixture, fixture)
    else:
        generate(fixture)
    source = (root / 'nextflow/module/tips.nf').read_text()
    assert source.count("  label 'trajectory_extract'") == 1
    # Historical baseline differs only by the new label; all scientific code is shared.
    baseline = source.replace("  label 'trajectory_extract'\n", '')
    results = {}
    for name, module, override in [('baseline', baseline, None),
                                    ('default', source, None),
                                    ('override', source, args.override)]:
        if name == 'override' and override is None:
            continue
        run = out / name
        run.mkdir()
        (run / 'tips.nf').write_text(module)
        (run / 'main.nf').write_text('''nextflow.enable.dsl=2
include { convert } from './tips.nf'
workflow {
 convert(Channel.of(tuple('fixture', file(params.input),
 '-f asetraj --subsample uniform --nsample 5 --start-idx 10 -of idx.xyz -o ds --filter "mindist>0.3"')))
}
''')
        (run / 'base.config').write_text("process {\n executor = 'local'\n withLabel: 'tips' { executor = 'local' }\n}\n")
        configs = [str(run / 'base.config')]
        if args.config:
            configs.append(str(args.config.resolve()))
        if override:
            configs.append(str(override.resolve()))
        command = [args.nextflow, '-C', ','.join(configs), 'run', 'main.nf',
                   '--input', str(fixture), '--publish', str(run / 'output'),
                   '-work-dir', 'work', '-with-trace', 'trace.txt']
        with (run / 'launch.log').open('w') as log:
            subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT, check=True)
        trace(run / 'trace.txt', 'COMPLETED')
        results[name] = outputs(run)
        assert results[name] == results['baseline'], results
        resume = command[:-1] + ['resume-trace.txt', '-resume']
        with (run / 'resume.log').open('w') as log:
            subprocess.run(resume, cwd=run, stdout=log, stderr=subprocess.STDOUT, check=True)
        trace(run / 'resume-trace.txt', 'CACHED')
        assert outputs(run) == results[name]
    import json
    (out / 'verification.json').write_text(json.dumps({'passed': True, 'outputs': results}, indent=2))
    print('PASS: identical extracted files; resume cached. Evidence:', out)


if __name__ == '__main__':
    main()
