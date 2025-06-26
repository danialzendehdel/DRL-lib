import unittest
import torch
import numpy as np
from drl_lib.agents.SAC.policy_nn import Actor

class TestSACActor(unittest.TestCase):

    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dims = [64, 64]
        self.device = 'cpu'
        self.test_state_np = np.random.randn(self.state_dim).astype(np.float32)
        self.test_state_torch = torch.tensor(self.test_state_np, dtype=torch.float32, device=self.device)

    def test_actor_initialization(self):
        actor_tanh = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device, use_tanh=True)
        self.assertTrue(actor_tanh.use_tanh)

        actor_no_tanh = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device, use_tanh=False)
        self.assertFalse(actor_no_tanh.use_tanh)

    def test_actor_forward_pass(self):
        actor = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device)
        mu, log_std = actor.forward(self.test_state_torch)
        self.assertEqual(mu.shape, (1, self.action_dim))
        self.assertEqual(log_std.shape, (1, self.action_dim))

    def test_sample_with_tanh_deterministic(self):
        actor = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device, use_tanh=True)
        action, log_prob = actor.sample(self.test_state_torch, deterministic=True)
        self.assertIsNone(log_prob)
        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertTrue(torch.all(action >= -1.0) and torch.all(action <= 1.0))

    def test_sample_with_tanh_stochastic(self):
        actor = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device, use_tanh=True)
        action, log_prob = actor.sample(self.test_state_torch, deterministic=False)
        self.assertIsNotNone(log_prob)
        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertEqual(log_prob.shape, (1, 1))
        self.assertTrue(torch.all(action >= -1.0) and torch.all(action <= 1.0))

    def test_sample_without_tanh_deterministic(self):
        actor = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device, use_tanh=False, action_bound=None)
        # Get mu directly to compare
        mu, _ = actor.forward(self.test_state_torch)
        action, log_prob = actor.sample(self.test_state_torch, deterministic=True)
        self.assertIsNone(log_prob)
        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertTrue(torch.allclose(action, mu)) # Action should be mu directly

    def test_sample_without_tanh_stochastic_unbounded(self):
        actor = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device, use_tanh=False, action_bound=None)
        action, log_prob = actor.sample(self.test_state_torch, deterministic=False)
        self.assertIsNotNone(log_prob)
        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertEqual(log_prob.shape, (1, 1))
        # We can't guarantee it's outside [-1,1] for a single sample, but it's not explicitly squashed
        # A more robust check would be statistical over many samples, or check if tanh was called (mocking)

    def test_sample_with_tanh_action_bounds(self):
        action_bound = (-0.5, 0.5)
        actor = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device, use_tanh=True, action_bound=action_bound)
        for _ in range(10): # Sample multiple times
            action, _ = actor.sample(self.test_state_torch, deterministic=False)
            self.assertTrue(torch.all(action >= action_bound[0]) and torch.all(action <= action_bound[1]))

            action_det, _ = actor.sample(self.test_state_torch, deterministic=True)
            self.assertTrue(torch.all(action_det >= action_bound[0]) and torch.all(action_det <= action_bound[1]))


    def test_sample_without_tanh_with_action_bounds(self):
        action_bound = (-2.0, 2.0)
        actor = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device, use_tanh=False, action_bound=action_bound)

        # Test stochastic sampling
        # It's hard to guarantee a raw Gaussian sample will exceed (-1,1) but be within bounds after clipping
        # So we check if it's clipped to the bounds correctly
        for _ in range(20): # Sample a few times
            action, log_prob = actor.sample(self.test_state_torch, deterministic=False)
            self.assertIsNotNone(log_prob)
            self.assertEqual(action.shape, (1, self.action_dim))
            self.assertTrue(torch.all(action >= action_bound[0]) and torch.all(action <= action_bound[1]))

        # Test deterministic sampling
        # Create a state that likely produces mu outside of (-1,1) but within action_bound
        # This is tricky without knowing the weights, so we rely on clamping
        mu, _ = actor.forward(self.test_state_torch) # mu could be anything
        action_det, _ = actor.sample(self.test_state_torch, deterministic=True)
        self.assertTrue(torch.all(action_det >= action_bound[0]) and torch.all(action_det <= action_bound[1]))
        # If mu was outside action_bound, action_det should be clamped version of mu
        clamped_mu = torch.clamp(mu, action_bound[0], action_bound[1])
        self.assertTrue(torch.allclose(action_det, clamped_mu))


    def test_log_prob_calculation_with_tanh(self):
        actor = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device, use_tanh=True)
        mu, log_std = actor.forward(self.test_state_torch)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        # Manually re-sample u and calculate action and log_prob for comparison
        # This is a bit redundant with the actor's internal logic but ensures components are used as expected
        for _ in range(5): # Repeat a few times for robustness
            u = dist.rsample()
            action_expected = torch.tanh(u)
            log_prob_expected = dist.log_prob(u) - torch.log(1 - action_expected.pow(2) + 1e-6)
            log_prob_expected = log_prob_expected.sum(dim=-1, keepdim=True)

            # Get from actor.sample (need to ensure same u, which is hard without setting seed inside sample)
            # Instead, we check if the values are reasonable.
            # A better test would be to mock dist.rsample() to return a fixed u.
            action_sampled, log_prob_sampled = actor.sample(self.test_state_torch, deterministic=False)
            self.assertIsNotNone(log_prob_sampled)
            # Cannot directly compare log_prob_sampled with log_prob_expected due to different u
            # But we can check its presence and shape, done in other tests.

    def test_log_prob_calculation_without_tanh(self):
        actor = Actor(self.state_dim, self.action_dim, self.hidden_dims, device=self.device, use_tanh=False, action_bound=None)
        mu, log_std = actor.forward(self.test_state_torch)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        for _ in range(5):
            u = dist.rsample()
            # action_expected = u (if no bound)
            log_prob_expected = dist.log_prob(u).sum(dim=-1, keepdim=True)

            action_sampled, log_prob_sampled = actor.sample(self.test_state_torch, deterministic=False)
            self.assertIsNotNone(log_prob_sampled)
            # As above, direct comparison is tricky. Shape and presence are key.
            # We can check that the log_prob does NOT contain the tanh correction term.
            # This is implicitly tested by comparing against a manually calculated Gaussian log_prob.
            # A rough check: if use_tanh=False, log_prob should be closer to dist.log_prob(action_sampled)
            # (if action_sampled was not clamped).
            # For non-deterministic, action_sampled is `u`
            manual_log_prob = dist.log_prob(action_sampled).sum(dim=-1, keepdim=True)
            self.assertTrue(torch.allclose(log_prob_sampled, manual_log_prob))


if __name__ == '__main__':
    unittest.main()
