import numpy as np
import pytest
import torch

from franka_diffusion.policy import DiffusionPolicy

B, OBS_DIM, ACT_DIM, HORIZON, N_OBS, N_ACT = 4, 17, 4, 16, 2, 8


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    return DiffusionPolicy(obs_dim=OBS_DIM, action_dims=ACT_DIM, horizon=HORIZON,
                           n_obs_steps=N_OBS, n_action_steps=N_ACT)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "obs": torch.rand(B, N_OBS, OBS_DIM, generator=g) * 2 - 1,
        "action": torch.rand(B, HORIZON, ACT_DIM, generator=g) * 2 - 1,
    }


def test_horizon_not_divisible_by_four_is_rejected():
    with pytest.raises(AssertionError):
        DiffusionPolicy(horizon=15)


def test_ddim_uses_the_training_noise_schedule(policy):
    train = policy.noise_scheduler
    infer = policy.inference_scheduler
    assert infer.config.num_train_timesteps == train.config.num_train_timesteps
    np.testing.assert_allclose(infer.alphas_cumprod.numpy(), train.alphas_cumprod.numpy())


def test_ddim_visits_num_inference_steps_within_the_training_range(policy):
    policy.inference_scheduler.set_timesteps(policy.num_inference_steps)
    steps = policy.inference_scheduler.timesteps
    assert len(steps) == policy.num_inference_steps
    assert steps.min() >= 0
    assert steps.max() < policy.noise_scheduler.config.num_train_timesteps


def test_loss_is_a_finite_scalar_with_gradients(policy):
    loss = policy.compute_loss(_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in policy.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    policy.zero_grad()


def test_predict_action_returns_executed_steps_in_action_range(policy):
    policy.eval()
    action = policy.predict_action(_batch()["obs"])
    assert action.shape == (B, N_ACT, ACT_DIM)
    assert action.min() >= -1.0 and action.max() <= 1.0
    assert not action.requires_grad


def test_predict_action_works_for_a_single_observation(policy):
    # Evaluation queries one env at a time - batch of 1 must not collapse
    policy.eval()
    obs = _batch()["obs"][:1]
    assert policy.predict_action(obs).shape == (1, N_ACT, ACT_DIM)


def test_overfitting_one_batch_reduces_the_loss_and_reproduces_its_actions():
    """ Overfitting a single batch checks the whole gradient path (noising, U-Net,
        loss) before spending an hour on a real run. Reproducing the batch also checks
        that predict_action returns the executed slice of the horizon: the targets are
        independent per step, so a shifted slice would score like random actions. """
    torch.manual_seed(0)
    policy = DiffusionPolicy(obs_dim=OBS_DIM, action_dims=ACT_DIM, horizon=HORIZON,
                             n_obs_steps=N_OBS, n_action_steps=N_ACT)
    batch = _batch()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    def mean_loss():
        # Average over fixed noise draws so the comparison is not dominated by sampling
        torch.manual_seed(123)
        with torch.no_grad():
            return np.mean([policy.compute_loss(batch).item() for _ in range(8)])

    before = mean_loss()
    torch.manual_seed(1)
    for _ in range(150):
        optimizer.zero_grad()
        policy.compute_loss(batch).backward()
        optimizer.step()
    after = mean_loss()

    assert after < 0.5 * before, f"loss {before:.3f} -> {after:.3f}"

    # Measured: ~0.28 after 200 steps vs ~0.67 for uniform random actions
    policy.eval()
    torch.manual_seed(5)
    predicted = policy.predict_action(batch["obs"])
    executed = batch["action"][:, N_OBS - 1:N_OBS - 1 + N_ACT]
    mae = (predicted - executed).abs().mean().item()
    assert mae < 0.45, f"MAE {mae:.3f}"
