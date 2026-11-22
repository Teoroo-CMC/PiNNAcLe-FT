"""Exercise the actual MD module and Slurm script generator without submitting jobs."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


SBATCH = r'''#!/usr/bin/env python3
import json, os
from pathlib import Path
import sys
work = Path.cwd()
script = Path(sys.argv[-1]).read_text()
record = {'headers': [line for line in script.splitlines() if line.startswith('#SBATCH')],
          'command': (work / '.command.sh').read_text(), 'work': str(work)}
(Path(os.environ['WALLTIME_CAPTURE']) / (str(os.getpid()) + '.json')).write_text(json.dumps(record))
# Submission-only test: do not run the job, container or MD. Satisfy output collection.
for name in ['asemd.traj', 'asemd.log', 'training-probe', 'labeling-probe', 'mace-probe', '.command.log', '.command.begin']:
    (work / name).touch()
(work / '.exitcode').write_text('0\n')
print('Submitted batch job', os.getpid())
'''


def minutes(headers):
    values = []
    for line in headers:
        tokens = shlex.split(line.removeprefix('#SBATCH'))
        for index, token in enumerate(tokens):
            if token in ('-t', '--time'):
                values.append(tokens[index + 1])
            elif token.startswith('--time='):
                values.append(token.split('=', 1)[1])
    assert len(values) == 1, f'Expected exactly one time directive: {headers}'
    value = values[0]
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    hours, mins, seconds = map(int, clock.split(':'))
    assert seconds == 0
    return int(days) * 1440 + hours * 60 + mins


def run_tests(root, nextflow, model):
    repo = Path(__file__).resolve().parents[2]
    fake_bin = root / 'bin'
    fake_bin.mkdir()
    for name, script in [('sbatch', SBATCH), ('squeue', '#!/bin/sh\nexit 0\n'),
                         ('scancel', '#!/bin/sh\nexit 0\n')]:
        path = fake_bin / name
        path.write_text(script)
        path.chmod(0o755)
    if model is None:
        model = root / 'model-placeholder'
        model.mkdir()
        (model / 'params.yml').write_text('# Not loaded: submission-only test.\n')
    override = root / 'capture.config'
    override.write_text("executor.pollInterval = '1 sec'\nexecutor.queueStatInterval = '1 sec'\n"
                        "singularity.enabled = false\nprocess.cache = false\n")
    cases = [
        ('alvis-static', 'alvis', '5,1000', [], {5: 5760, 1000: 5760}),
        ('arrhen-static', 'arrhen', '5,1000', [], {5: 1440, 1000: 1440}),
        ('alvis-adaptive', 'alvis', '5,40,160,640,1000', ['--adaptive_md_walltime'],
         {5: 10, 40: 30, 160: 110, 640: 430, 1000: 670}),
        ('arrhen-adaptive', 'arrhen', '5,40,160,640,1000', ['--adaptive_md_walltime'],
         {5: 10, 40: 30, 160: 110, 640: 430, 1000: 670}),
        ('custom', 'arrhen', '1,5,40', ['--adaptive_md_walltime', '--md_seconds_per_ps', '60',
         '--md_walltime_overhead_minutes', '2', '--md_walltime_min_minutes', '4'],
         {1: 4, 5: 7, 40: 42}),
        ('last-duration', 'arrhen', '5', ['--adaptive_md_walltime', '--md_extra_flags', '--t 999'], {5: 10}),
        ('invalid-duration', 'arrhen', '0', ['--adaptive_md_walltime'], None),
        ('nonfinite-duration', 'arrhen', 'NaN', ['--adaptive_md_walltime'], None),
        ('invalid-rate', 'arrhen', '5', ['--adaptive_md_walltime', '--md_seconds_per_ps', '0'], None),
    ]
    summary = []
    for name, profile, durations, options, expected in cases:
        case = root / name
        case.mkdir()
        capture = case / 'captured'
        capture.mkdir()
        env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ['PATH'],
                   WALLTIME_CAPTURE=str(capture), NXF_ANSI_LOG='false')
        command = [nextflow, '-C', f'{repo / "nextflow.config"},{override}', 'run',
                   str(repo / 'tests/md_walltime/smoke.nf'), '-profile', profile,
                   '--model', str(model), '--durations', durations, '--publish', str(case / 'output')]
        if expected is not None:
            command += ['--resource_probes', '--mace_probe', str(profile == 'arrhen').lower()]
        result = subprocess.run(command + options, cwd=case, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        (case / 'nextflow.log').write_text(result.stdout)
        records = [json.loads(p.read_text()) for p in capture.glob('*.json')]
        if expected is None:
            assert result.returncode != 0 and not records, result.stdout
            assert 'must be positive' in result.stdout, result.stdout
            summary.append({'case': name, 'invalid_input_rejected_before_submission': True})
        else:
            assert result.returncode == 0, result.stdout
            md = {}
            other = {}
            for record in records:
                limit = minutes(record['headers'])
                script = record['command']
                if 'import pinn' in script:
                    duration = float(re.findall(r'--t[\s,=]+([0-9.eE+-]+)', script)[-1])
                    assert duration not in md
                    md[duration] = limit
                else:
                    probe = next(p for p in ['training', 'labeling', 'mace'] if f'touch {p}-probe' in script)
                    other[probe] = limit
            assert md == expected, (name, md, expected)
            other_expected = {'training': 5760 if profile == 'alvis' else 1440, 'labeling': 1440}
            if profile == 'arrhen':
                other_expected['mace'] = 60
            assert other == other_expected, (name, other)
            summary.append({'case': name, 'md_minutes': md, 'other_minutes': other,
                            'one_time_directive_per_job': True})
        print(f'PASS {name}', flush=True)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nextflow', default='nextflow')
    parser.add_argument('--model', type=Path, help='Optional real checkpoint to stage; no inference is run')
    parser.add_argument('--output', type=Path, help='Keep generated submission evidence in this new directory')
    args = parser.parse_args()
    model = args.model.resolve() if args.model else None
    if args.output:
        root = args.output.resolve()
        root.mkdir(parents=True)
        run_tests(root, args.nextflow, model)
    else:
        with tempfile.TemporaryDirectory(prefix='md-walltime-') as directory:
            run_tests(Path(directory), args.nextflow, model)
