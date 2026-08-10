"""Quantize one tower of a GR00T policy, and put the full-precision weights back.

Two things here differ from the obvious approach, both for memory reasons. A 3B
model is ~12 GB in bf16, so:

  - Quantization runs `inplace=True`. `quantize_model` otherwise deep-copies the
    module it is given and returns the copy; quantizing `getattr(model, tower)`
    and discarding that return value would leave the model untouched and report
    a "quantized" result identical to FP.

  - Restoring uses a single cached CPU copy of the original weights rather than
    keeping a second model alive. Only one model is ever on the GPU.
"""

import time

from apexquant import quantize_model
import torch


def fp_weights(model):
    """A CPU snapshot of every weight, to restore between grid points."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def restore(model, snapshot):
    """Undo quantization by copying the snapshot back."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(snapshot[name].to(p.device, p.dtype))


def quantize_tower(model, tower, bits, method, exclude=None, seed=0):
    """Quantize `model.<tower>` in place. `tower='both'` quantizes everything.

    Preflight is off because the exclusion list is chosen deliberately from the
    layer inventory; the audit is reported separately rather than used as a gate.
    """
    target = model if tower == "both" else getattr(model, tower)
    started = time.time()
    _, stats = quantize_model(
        target, bits=bits, method=method, exclude=exclude,
        inplace=True, preflight=False, rotation_seed=seed,
    )
    return stats, time.time() - started


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from discover import load_policy

    parser = argparse.ArgumentParser(description="Time quantization of each tower.")
    parser.add_argument("model_path", nargs="?", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--embodiment-tag", default="REAL_G1")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--method", default="apexquant")
    args = parser.parse_args()

    policy = load_policy(args.model_path, "examples/SO100/so100_config.py", args.embodiment_tag)
    model = policy.model

    for tower in ["backbone", "action_head"]:
        stats, seconds = quantize_tower(model, tower, args.bits, args.method)
        params = sum(s.d for s in stats)
        print(f"{tower:14s} {len(stats):4d} layers  {seconds:7.1f} s  ({args.method}, {args.bits}-bit)")
