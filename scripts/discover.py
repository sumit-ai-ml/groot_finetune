"""Phase 6a/6b/6c: find out what the model is actually made of.

Nothing here assumes a module layout. GR00T's submodule names differ between
releases, and the whole quantization study is a statement about specific towers,
so the tower names are discovered and printed rather than hardcoded.

Usage:
    python scripts/discover.py [model-path] [--out runs/phase6]
"""

import argparse
import importlib
from pathlib import Path
import sys

from apexquant import audit
import torch


def load_policy(model_path, modality_config, embodiment_tag="NEW_EMBODIMENT", device="cpu"):
    """Load a GR00T policy, registering the custom embodiment's modality config first.

    On CPU the backbone must not use FlashAttention2, which is CUDA-only. GR00T
    picks the attention implementation by trying `import flash_attn` and falling
    back to sdpa when it fails, so poisoning that import is the supported-path
    way to get an sdpa model -- and it leaves the GPU free for training.
    """
    if device == "cpu":
        sys.modules["flash_attn"] = None  # makes `import flash_attn` raise ImportError

    config_path = Path(modality_config)
    sys.path.append(str(config_path.parent))
    importlib.import_module(config_path.stem)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    return Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(embodiment_tag),
        model_path=model_path,
        device=device,
    )


def linear_inventory(model):
    """One line per Linear/Conv2d: name, shape, fan-in, parameter count."""
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            rows.append((name, f"in={m.in_features}", f"out={m.out_features}", m.weight.numel()))
        elif isinstance(m, torch.nn.Conv2d):
            fan_in = m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
            rows.append((name, f"in={fan_in}", f"out={m.out_channels}", m.weight.numel()))
    return rows


def is_power_of_two(n):
    return n > 0 and n & (n - 1) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--modality-config", default="examples/SO100/so100_config.py")
    parser.add_argument("--out", default="runs/phase6")
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT",
                        help="Base checkpoints reject NEW_EMBODIMENT; use e.g. REAL_G1 for those.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.model_path, args.modality_config, args.embodiment_tag)
    model = policy.model

    print("=" * 70)
    print("6a. TOP-LEVEL CHILDREN")
    print("=" * 70)
    children = [name for name, _ in model.named_children()]
    for name in children:
        child = getattr(model, name)
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:30s} {type(child).__name__:35s} {n/1e6:9.1f} M params")
    print(f"  {'TOTAL':30s} {'':35s} {sum(p.numel() for p in model.parameters())/1e6:9.1f} M params")

    print()
    print("=" * 70)
    print("6b. FUSED ATTENTION CHECK")
    print("=" * 70)
    fused = [n for n, m in model.named_modules() if isinstance(m, torch.nn.MultiheadAttention)]
    print(f"  nn.MultiheadAttention modules: {fused if fused else 'none -- safe'}")

    rows = linear_inventory(model)
    (out / "layer_inventory.txt").write_text(
        "".join(f"{n:75s} {i:>12s} {o:>12s} {p:>12d}\n" for n, i, o, p in rows)
    )
    print(f"\n  {len(rows)} quantizable layers -> {out / 'layer_inventory.txt'}")

    print()
    print("=" * 70)
    print("6e. ROTATION COST: fan-in dimensions")
    print("=" * 70)
    print("  SRHT needs a power-of-2 fan-in; make_rotation() silently falls back")
    print("  to a dense d x d QR otherwise, which is O(d^3) to build.")
    fan_ins = {}
    for name, i, _, params in rows:
        d = int(i.split("=")[1])
        entry = fan_ins.setdefault(d, [0, 0])
        entry[0] += 1
        entry[1] += params
    for d in sorted(fan_ins):
        count, params = fan_ins[d]
        flag = "SRHT" if is_power_of_two(d) else "DENSE <-- O(d^3)"
        print(f"  d={d:<8d} {count:4d} layers  {params/1e6:8.1f} M params   {flag}")
    dense_params = sum(p for d, (_, p) in fan_ins.items() if not is_power_of_two(d))
    total_params = sum(p for _, (_, p) in fan_ins.items())
    print(f"\n  parameter mass needing DENSE rotation: {100*dense_params/total_params:.1f}%")

    print()
    print("=" * 70)
    print("6c. FAN-IN AUDIT")
    print("=" * 70)
    for label, module in [("WHOLE MODEL", model)] + [(c, getattr(model, c)) for c in children]:
        if not any(True for _ in module.parameters()):
            continue
        report = audit(module, verbose=False)
        if not report.layers:
            continue
        print(f"\n  --- {label} ---")
        print(f"  verdict={report.overall_verdict}  layers={len(report.layers)}  "
              f"good={report.n_layers_good} marginal={report.n_layers_marginal} bad={report.n_layers_bad}")
        for layer in report.layers:
            if layer.verdict != "good":
                print(f"    {layer.verdict.upper():9s} {layer.name:55s} fan_in={layer.fan_in:<7d} {layer.shape}")


if __name__ == "__main__":
    main()
