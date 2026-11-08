"""Check information isolation and the paired statistical decision, without physics reruns."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'policy'))
import probe_feedback as experiment


class ExperimentTests(unittest.TestCase):
    def test_geometry_cannot_change_feedback_action(self):
        policy = {'direct':{'force_N':10.},'probe_blind':{'force_N':14.},
                  'probe_feedback':{'k':1,'features':[[0.,3.],[3.,1.]],
                                    'scale':[1.,1.],'rewards':[[1.,0.,0.,0.],[0.,0.,1.,0.]]}}
        self.assertEqual(experiment.choose(policy,'probe_feedback',[.2,.06,.05],[0.,3.]),6.)
        self.assertEqual(experiment.choose(policy,'probe_feedback',[.4,.08,.09],[0.,3.]),6.)
        self.assertEqual(experiment.choose(policy,'probe_feedback',[.2,.06,.05],[20.,1.]),14.)
        self.assertEqual(experiment.choose(policy,'probe_blind',[.2,.06,.05],[0.,3.]),14.)

    def summarize_fixture(self, *, mismatch=False, simfail=False):
        rows=[]
        for group in range(8):
            for category in ('hard','bendable','compressible'):
                for seed in (0,1):
                    for arm in experiment.ARMS:
                        outcome='success' if arm=='probe_feedback' else 'grasp_fail'
                        if simfail and group==0 and category=='hard' and seed==0 and arm=='probe_feedback':
                            outcome='sim_fail'
                        rows.append(dict(object_id=f'{group}_{category}',group=group,seed=seed,arm=arm,
                            category=category,outcome=outcome,sim_s=10.,selected_force_N=10.,
                            probe_digest='different' if mismatch and arm=='probe_feedback' else 'same',
                            probe_raw=[1.,2.],feedback=[1.,2.]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            experiment.save(path/'policy.json',{})
            (path/'test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                experiment.statistics(path)
            return json.loads((path/'summary.json').read_text())

    def test_paired_group_bootstrap_and_invalid_pairs(self):
        result=self.summarize_fixture()
        comparison=result['comparisons']['probe_feedback_minus_probe_blind']
        self.assertEqual(comparison['ci95_cluster'],[1.,1.])
        self.assertEqual(comparison['wins'],48)
        self.assertTrue(result['supports_feedback_benefit'])
        result=self.summarize_fixture(simfail=True)
        comparison=result['comparisons']['probe_feedback_minus_probe_blind']
        self.assertEqual(comparison['excluded_simfail_pairs'],1)
        self.assertEqual(comparison['valid_pairs'],47)
        self.assertEqual(comparison['delta_valid_pairs'],1.)
        self.assertAlmostEqual(comparison['delta'],47/48)

    def test_probe_prefix_mismatch_invalidates_claim(self):
        result=self.summarize_fixture(mismatch=True)
        self.assertEqual(len(result['probe_prefix_mismatches']),48)
        self.assertFalse(result['supports_feedback_benefit'])


if __name__=='__main__':
    unittest.main()
