#!/usr/bin/env python
"""Phase 7: quantize the finetuned GR00T policy and measure what it costs.

    source scripts/env.sh
    python $GROOT_PROJECT/scripts/quant_sweep.py --out $GROOT_PROJECT/results/quant_sweep.jsonl

One row per grid point, written incrementally so a crash keeps what ran.

Two things worth knowing about the numbers this produces:

  - Quantization here is *simulated*: apexquant reconstructs each weight and
    writes it back in the module's own dtype. Nothing gets packed, so latency
    and resident memory do not change. `bytes_quantized` reports the size the
    weights *would* occupy packed, which is the real axis to plot accuracy
    against; measured latency is reported only to show it is flat.

  - The action head is a denoiser driven by sampled noise, so a policy
    evaluated twice under different seeds gives different numbers. Every grid
    point runs under the same seed as the FP reference, which makes the
    sampling noise common-mode; the FP multi-seed spread is measured separately
    and is the bar a quantization delta has to clear to mean anything.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import torch

from discover import load_policy
import eval_lib
from quant_lib import fp_weights, quantize_tower, restore

PROJECT = Path(__file__).resolve().parent.parent


def quantizable_param_count(model, tower):
    """(params in Linear/Conv2d weights, rows) for the tower the sweep touches."""
    target = model if tower == "both" else getattr(model, tower)
    params = rows = 0
    for m in target.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            params += m.weight.numel()
            rows += m.weight.shape[0]
    return params, rows


def packed_bytes(model, tower, bits):
    """Size of the weights if the codes were actually stored at `bits`.

    Quantized layers cost `bits` per coefficient plus one FP32 scale per output
    row; everything else stays at the model's own dtype.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    q_params, q_rows = quantizable_param_count(model, tower)
    q_bytes_fp = q_params * 2  # bf16
    return total_bytes - q_bytes_fp + q_params * bits / 8 + q_rows * 4


def measure_latency(policy, loader, embodiment_tag, repeats=5):
    """Seconds per get_action call, after a warmup call."""
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t

    episode = loader[0]
    configs = {k: v for k, v in loader.modality_configs.items() if k != "action"}
    language_keys = loader.modality_configs["language"].modality_keys
    step = extract_step_data(episode, 0, configs, embodiment_tag)
    observation = {f"state.{k}": v for k, v in step.states.items()}
    observation |= {f"video.{k}": np.array(v) for k, v in step.images.items()}
    observation |= {k: step.text for k in language_keys}
    parsed = parse_observation_gr00t(observation, loader.modality_configs)

    policy.get_action(parsed)
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(repeats):
        policy.get_action(parsed)
    torch.cuda.synchronize()
    return (time.time() - started) / repeats


def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  -> {row['label']}: mse={row['mse']:.4f} terminal_l2={row['terminal_l2']:.3f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(PROJECT / "checkpoints/so101_ft"))
    ap.add_argument("--dataset", default="demo_data/so101-table-cleanup-eval")
    ap.add_argument("--modality-config", default="examples/SO100/so100_config.py")
    ap.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 2])
    ap.add_argument("--methods", nargs="+", default=["apexquant", "quarot", "rtn_absmax"])
    ap.add_argument("--towers", nargs="+", default=["both"],
                    help="'both', or individual towers for the ablation.")
    ap.add_argument("--ablation-bits", type=int, default=4,
                    help="Bit width for the per-tower ablation.")
    ap.add_argument("--ablation-towers", nargs="*", default=["backbone", "action_head"])
    ap.add_argument("--ablation-method", default="apexquant")
    ap.add_argument("--fp-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Seeds for the FP noise floor. The first is the reference.")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=None, help="Limit episode count.")
    ap.add_argument("--out", default=str(PROJECT / "results/quant_sweep.jsonl"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    print(f"loading {args.model}", flush=True)
    policy = load_policy(args.model, args.modality_config, args.embodiment_tag, device="cuda")
    model = policy.model
    loader = eval_lib.load_eval_loader(policy, args.dataset)

    traj_ids = list(range(len(loader)))[: args.episodes]
    embodiment = args.embodiment_tag.lower()
    ref_seed = args.fp_seeds[0]
    eval_kwargs = dict(steps=args.steps)
    print(f"{len(traj_ids)} eval episodes, {args.steps} steps each", flush=True)

    # --------------------------------------------------------- FP reference
    started = time.time()
    ref_trajectories = eval_lib.rollout_many(policy, loader, traj_ids, embodiment,
                                             ref_seed, **eval_kwargs)
    eval_seconds = time.time() - started
    # Ground truth does not depend on the policy, so these scales normalize every
    # later grid point identically.
    scales = eval_lib.dimension_scales(ref_trajectories)
    fp_ref = eval_lib.metrics(ref_trajectories, scales) | {"seed": ref_seed}
    fp_latency = measure_latency(policy, loader, embodiment)
    print(f"FP eval pass: {eval_seconds:.0f}s, get_action {1000*fp_latency:.0f} ms", flush=True)

    base = dict(method="fp", bits=16, tower="none", quant_seconds=0.0,
                weight_mse=0.0, n_layers=0, latency_s=fp_latency,
                packed_bytes=sum(p.numel() * p.element_size() for p in model.parameters()))
    append(out, base | fp_ref | {"label": f"fp/seed{ref_seed}"})

    for seed in args.fp_seeds[1:]:
        row = eval_lib.evaluate(policy, loader, traj_ids, embodiment, seed,
                                scales=scales, **eval_kwargs)
        append(out, base | row | {"label": f"fp/seed{seed}"})

    # ------------------------------------------------------------ the grid
    snapshot = fp_weights(model)
    print(f"FP snapshot cached on CPU ({sum(v.numel()*v.element_size() for v in snapshot.values())/1e9:.1f} GB)",
          flush=True)

    grid = [(t, m, b) for t in args.towers for m in args.methods for b in args.bits]
    grid += [(t, args.ablation_method, args.ablation_bits) for t in args.ablation_towers]

    for i, (tower, method, bits) in enumerate(grid, 1):
        label = f"{method}/{bits}b/{tower}"
        print(f"[{i}/{len(grid)}] {label}", flush=True)
        restore(model, snapshot)
        stats, quant_seconds = quantize_tower(model, tower, bits, method)
        weight_mse = float(np.mean([s.mse for s in stats])) if stats else 0.0
        print(f"  quantized {len(stats)} layers in {quant_seconds:.0f}s, "
              f"mean weight mse {weight_mse:.3e}", flush=True)

        row = eval_lib.evaluate(policy, loader, traj_ids, embodiment, ref_seed,
                                scales=scales, **eval_kwargs)
        append(out, row | dict(
            label=label, method=method, bits=bits, tower=tower,
            quant_seconds=quant_seconds, weight_mse=weight_mse, n_layers=len(stats),
            latency_s=measure_latency(policy, loader, embodiment),
            packed_bytes=packed_bytes(model, tower, bits),
        ))

    restore(model, snapshot)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
