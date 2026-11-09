"""Run on the server: python3 -m unittest discover -s experiments/comparison -p 'test_*.py'."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from cornet_rt import CORnetRT, ConditionalCORnetRT, NormalizedModel
from evaluate_comparison import best_pgd
from jobs import evaluation_jobs, training_jobs
from main_conditional import sample_profile
from noises import NoisyConv2d


class ProtocolTests(unittest.TestCase):
    def test_manifest(self):
        jobs = training_jobs()
        self.assertEqual(len(jobs), 42)
        self.assertEqual(len({(j['name'], j['seed']) for j in jobs}), len(jobs))
        self.assertEqual(len(evaluation_jobs()), 396)
        self.assertEqual({j['epsilon'] for j in evaluation_jobs()}, {2, 4, 8})

    def test_fixed_ramp(self):
        args = SimpleNamespace(fixed_profile=[0.5] * 4, max_sigma=2.5)
        for maximum, expected in ((0, 0), (1.25, 0.25), (2.5, 0.5)):
            profile, _ = sample_profile(args, maximum, 'cpu')
            torch.testing.assert_close(profile, torch.full((4,), float(expected)))

    def test_noise_eval_switch(self):
        conv = NoisyConv2d(1, 1, 1, sigma_axon=0.5).eval()
        x = torch.ones(1, 1, 2, 2)
        torch.testing.assert_close(conv(x), conv(x))
        conv.noise_in_eval = True
        torch.manual_seed(1)
        first = conv(x)
        torch.manual_seed(2)
        self.assertFalse(torch.equal(first, conv(x)))

    def test_conditioner_initialization_preserves_baseline(self):
        plain = NormalizedModel(CORnetRT(times=5)).eval()
        conditional = NormalizedModel(ConditionalCORnetRT(times=5, max_sigma=2.5)).eval()
        missing, unexpected = conditional.load_state_dict(plain.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(k.startswith('model.conditioner.') for k in missing))
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            torch.testing.assert_close(plain(x), conditional(x, [0.0] * 4))

    def test_pgd_budget(self):
        class Toy(torch.nn.Module):
            def forward(self, inputs, profile=None):
                value = inputs.flatten(1).mean(1)
                return torch.stack((value, -value), dim=1)
        x, labels = torch.full((2, 1, 2, 2), 0.5), torch.zeros(2, dtype=torch.long)
        attacked = best_pgd(Toy(), x, labels, None, 8 / 255, 3, 1, 1, 2)
        self.assertLessEqual(float((attacked - x).abs().max()), 8 / 255 + 1e-6)
        self.assertGreaterEqual(float(attacked.min()), 0)
        self.assertLessEqual(float(attacked.max()), 1)


if __name__ == '__main__':
    unittest.main()
