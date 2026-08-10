"""Open-loop evaluation of a GR00T policy, returning raw trajectories.

`gr00t.eval.open_loop_eval` reports one aggregate MSE per trajectory and draws a
plot every time it runs. This study needs the underlying arrays instead, because
it reports per-dimension error and trajectory drift, and it repeats the whole
evaluation dozens of times -- once per seed for the noise floor, once per point
on the quantization grid. So this module re-uses GR00T's data loading, drops the
plotting, and hands back the numbers.

The policy is stochastic: the action head is a denoiser driven by sampled noise.
Every entry point here takes an explicit seed for that reason.
"""

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.utils import parse_observation_gr00t
import numpy as np
import torch


# ------------------------------------------------------------------- rollouts

def rollout(policy, loader, traj_id, embodiment_tag, steps=200, execution_horizon=16):
    """Run the policy open-loop over one episode.

    Open-loop means the policy never sees the consequences of its own actions:
    every `execution_horizon` steps it re-observes the *ground truth* state and
    predicts the next chunk. Errors therefore accumulate within a chunk but are
    reset at each chunk boundary.

    Returns (ground_truth, predicted), both (T, action_dim).
    """
    episode = loader[traj_id]
    n_steps = min(steps, len(episode))

    action_keys = loader.modality_configs["action"].modality_keys
    observation_configs = {k: v for k, v in loader.modality_configs.items() if k != "action"}
    language_keys = loader.modality_configs["language"].modality_keys

    predicted = []
    for start in range(0, n_steps, execution_horizon):
        step = extract_step_data(episode, start, observation_configs, embodiment_tag)
        observation = {f"state.{k}": v for k, v in step.states.items()}
        observation |= {f"video.{k}": np.array(v) for k, v in step.images.items()}
        observation |= {k: step.text for k in language_keys}

        chunk, _ = policy.get_action(parse_observation_gr00t(observation, loader.modality_configs))
        for offset in range(execution_horizon):
            predicted.append(np.concatenate(
                [np.atleast_1d(np.atleast_1d(chunk[key][0])[offset]) for key in action_keys]
            ))

    ground_truth = np.concatenate(
        [np.vstack(list(episode[f"action.{key}"])) for key in action_keys], axis=-1
    )[:n_steps]
    return ground_truth, np.array(predicted)[:n_steps]


def rollout_many(policy, loader, traj_ids, embodiment_tag, seed, **kwargs):
    """Rollout over several episodes under one sampling seed.

    Returns {traj_id: (ground_truth, predicted)}.
    """
    torch.manual_seed(seed)
    return {i: rollout(policy, loader, i, embodiment_tag, **kwargs) for i in traj_ids}


# -------------------------------------------------------------------- metrics

def dimension_scales(trajectories):
    """Per-dimension standard deviation of the ground truth, as a normalizer.

    Joint angles and the gripper differ in range by enough that an unnormalized
    aggregate MSE just reports whichever dimension has the widest swing.
    """
    stacked = np.concatenate([gt for gt, _ in trajectories.values()], axis=0)
    return np.maximum(stacked.std(axis=0), 1e-8)


def metrics(trajectories, scales):
    """Summarise a set of rollouts into the numbers the study reports.

    - mse_per_dim:  normalized squared error, one entry per action dimension
    - mse:          mean of those, the single headline number
    - terminal_l2:  normalized L2 error at the last step of each episode, averaged.
                    Per-step MSE averages drift away; the endpoint keeps it.
    - drift_ratio:  mean error over the final quarter of an episode divided by
                    the first quarter. > 1 means error compounds over time.
    """
    per_dim, terminal, early, late = [], [], [], []
    for ground_truth, predicted in trajectories.values():
        error = (ground_truth - predicted) / scales
        per_dim.append((error ** 2).mean(axis=0))
        terminal.append(np.linalg.norm(error[-1]))
        quarter = max(len(error) // 4, 1)
        early.append(np.abs(error[:quarter]).mean())
        late.append(np.abs(error[-quarter:]).mean())

    per_dim = np.mean(per_dim, axis=0)
    return {
        "mse_per_dim": per_dim.tolist(),
        "mse": float(per_dim.mean()),
        "terminal_l2": float(np.mean(terminal)),
        "drift_ratio": float(np.mean(late) / max(np.mean(early), 1e-12)),
        "n_episodes": len(trajectories),
    }


# ------------------------------------------------------------------- plumbing

def load_eval_loader(policy, dataset_path):
    """A loader whose modality config matches whatever the policy expects."""
    return LeRobotEpisodeLoader(
        dataset_path=dataset_path, modality_configs=policy.get_modality_config()
    )


def evaluate(policy, loader, traj_ids, embodiment_tag, seed, scales=None, **kwargs):
    """One full evaluation pass: rollouts under `seed`, then metrics.

    Pass `scales` from a reference run so every point on the sweep is normalized
    the same way; otherwise it is derived from this run's own ground truth,
    which is identical anyway since ground truth does not depend on the policy.
    """
    trajectories = rollout_many(policy, loader, traj_ids, embodiment_tag, seed, **kwargs)
    scales = dimension_scales(trajectories) if scales is None else scales
    return metrics(trajectories, scales) | {"seed": seed}
