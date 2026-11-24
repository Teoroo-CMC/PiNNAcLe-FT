#!/usr/bin/env python3
"""Run and validate one single-point attempt; never accept unconverged forces."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


def validate(text, natoms):
    """Validate one ENERGY_FORCE result, including both CP2K force print formats."""
    if 'SCF run NOT converged' in text or 'SCF run converged' not in text:
        return False, 'SCF not converged'
    if 'PROGRAM ENDED AT' not in text:
        return False, 'CP2K did not terminate normally'
    energies = re.findall(r'^\s*ENERGY\| Total FORCE_EVAL[^\n]*?\s+(\S+)\s*$', text, re.M)
    blocks = re.findall(r'ATOMIC FORCES in.*?\n(.*?)(?:SUM OF ATOMIC FORCES|\n\s*\n)', text, re.S)
    rows = []
    if blocks:
        # Legacy: atom, kind, element, Fx, Fy, Fz.
        rows = re.findall(r'^\s*(\d+)\s+\d+\s+[A-Za-z]+\s+(\S+)\s+(\S+)\s+(\S+)\s*$', blocks[-1], re.M)
    else:
        # CP2K 2026: FORCES| atom Fx Fy Fz |F|.
        rows = re.findall(r'^\s*FORCES\|\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+\s*$', text, re.M)
    if len(energies) != 1 or len(rows) != natoms or [int(r[0]) for r in rows] != list(range(1, natoms + 1)):
        return False, 'Expected one energy and one complete, ordered force table'
    try:
        values = [float(v.replace('D', 'E')) for v in energies + [v for row in rows for v in row[1:]]]
    except ValueError:
        return False, 'Invalid energy or force value'
    if not all(math.isfinite(v) for v in values):
        return False, 'Non-finite energy or forces'
    return True, 'Validated energy and forces'


def wavefunction(directory):
    # Never silently choose between different projects or use another label's WFN.
    for pattern in ('*-RESTART.wfn', '*-RESTART.wfn.bak-1'):
        candidates = [p for p in directory.glob(pattern) if p.is_file() and p.stat().st_size]
        if len(candidates) > 1:
            raise ValueError('Ambiguous saved wavefunctions')
        if candidates:
            return candidates[0]
    return None


def restart_input(text):
    text, n = re.subn(r'(?mi)^(\s*SCF_GUESS)\s+[^\n]+$', r'\1 RESTART', text)
    if n != 1:
        raise ValueError('Retry requires exactly one SCF_GUESS keyword')
    text, n = re.subn(r'(?mi)^(\s*WFN_RESTART_FILE_NAME)\s+[^\n]+$', r'\1 retry-source.wfn', text)
    if n > 1:
        raise ValueError('Ambiguous WFN_RESTART_FILE_NAME')
    if n == 0:
        text, n = re.subn(r'(?mi)^(\s*&DFT)\s*(?:[!#][^\n]*)?$', r'\1\n  WFN_RESTART_FILE_NAME retry-source.wfn', text)
        if n != 1:
            raise ValueError('Retry requires exactly one DFT section')
    return text


def run(source, geometry, attempt, command):
    if attempt not in (1, 2):
        raise ValueError('Only one retry is allowed')
    source, geometry = source.resolve(), geometry.resolve()
    natoms = int(geometry.read_text().splitlines()[0])
    if natoms <= 0:
        raise ValueError('Empty geometry')
    out = Path('label')
    out.mkdir()
    original = source if attempt == 1 else source / 'original.inp'
    shutil.copyfile(original, out / 'original.inp')
    record = dict(attempt=attempt)
    text = original.read_text()
    if attempt == 2:
        previous = json.loads((source / 'label-outcome.json').read_text())
        if previous['attempt'] != 1 or not previous['retryable']:
            raise ValueError('Retry provenance mismatch')
        wfn = source / previous['wavefunction']
        shutil.copyfile(wfn, out / 'retry-source.wfn')
        text = restart_input(text)
    (out / 'cp2k-attempt.inp').write_text(text)
    with (out / 'cp2k.log').open('w') as log:
        status = subprocess.run(command + ' -i cp2k-attempt.inp', shell=True,
                                executable='/bin/bash', cwd=out, stdout=log).returncode
    log = (out / 'cp2k.log').read_text()
    # Executable/launcher failures, killed jobs and truncated output are not SCF labels.
    scf_failed = 'SCF run NOT converged' in log
    if (status != 0 or 'PROGRAM ENDED AT' not in log) and not scf_failed:
        raise RuntimeError(f'CP2K execution failed (exit {status}); see cp2k.log')
    valid, reason = validate(log, natoms)
    valid = valid and status == 0
    record.update(valid=valid, retryable=False, reason=reason, exit_status=status)
    if not valid and scf_failed and attempt == 1:
        wfn = wavefunction(out)
        if wfn:
            record.update(retryable=True, wavefunction=wfn.name)
        else:
            record['reason'] += '; no saved wavefunction for retry'
    (out / 'label-outcome.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--geometry', required=True, type=Path)
    parser.add_argument('--attempt', required=True, type=int)
    parser.add_argument('--command', required=True)
    args = parser.parse_args()
    run(args.source, args.geometry, args.attempt, args.command)
