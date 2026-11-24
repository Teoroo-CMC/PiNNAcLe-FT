"""Run with: python3 -m unittest discover -s tests -p 'test_cp2k_label.py'."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('label', Path(__file__).resolve().parents[1] / 'bin/cp2k_label.py')
label = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label)
GOOD = '''SCF run converged
ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -1.0
FORCES| 1 0.0 0.0 0.0 0.0
PROGRAM ENDED AT
'''
INPUT = '&DFT\n &SCF\n SCF_GUESS ATOMIC\n EPS_SCF 1e-6\n &END SCF\n&END DFT\n'


class Labels(unittest.TestCase):
    def test_validation(self):
        self.assertTrue(label.validate(GOOD, 1)[0])
        for bad in [GOOD.replace('converged', 'NOT converged'), GOOD.replace('-1.0', 'nan'),
                    GOOD.replace('PROGRAM ENDED AT', ''), GOOD.replace('FORCES| 1', 'FORCES| 2')]:
            self.assertFalse(label.validate(bad, 1)[0])
        self.assertFalse(label.validate(GOOD, 2)[0])

    def test_retry(self):
        for retry_succeeds in (True, False):
            with self.subTest(retry_succeeds=retry_succeeds), tempfile.TemporaryDirectory() as tmp:
                root, old = Path(tmp), Path.cwd()
                (root/'input.inp').write_text(INPUT)
                (root/'geo.xyz').write_text('1\n\nH 0 0 0\n')
                def cp2k(command, **kwargs):
                    attempt = 'RESTART' in (kwargs['cwd']/'cp2k-attempt.inp').read_text()
                    kwargs['stdout'].write(GOOD if attempt and retry_succeeds else 'SCF run NOT converged\nPROGRAM ENDED AT\n')
                    (kwargs['cwd']/'test-RESTART.wfn').write_text('saved WFN')
                    return type('Result', (), {'returncode': 0})()
                try:
                    for attempt in (1, 2):
                        work = root/str(attempt); work.mkdir(); os.chdir(work)
                        source = root/'input.inp' if attempt == 1 else root/'1/label'
                        with patch.object(label.subprocess, 'run', side_effect=cp2k):
                            label.run(source, root/'geo.xyz', attempt, 'cp2k')
                        outcome = json.loads(Path('label/label-outcome.json').read_text())
                        self.assertEqual(outcome['retryable'], attempt == 1)
                    self.assertEqual(outcome['valid'], retry_succeeds)
                    self.assertEqual(Path('label/retry-source.wfn').read_text(), 'saved WFN')
                    self.assertIn('EPS_SCF 1e-6', Path('label/cp2k-attempt.inp').read_text())
                    with self.assertRaises(ValueError):
                        label.run(source, root/'geo.xyz', 3, 'cp2k')
                finally:
                    os.chdir(old)
