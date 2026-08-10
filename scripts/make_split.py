"""Split a LeRobot dataset into disjoint train / eval datasets at episode level.

Whole episodes go to one side or the other -- never individual frames -- because
a policy that has seen part of a trajectory has effectively memorised the rest.

The split is stratified by task, so every task is represented on both sides.
Episode files are hardlinked, so the two copies cost no extra disk.

Usage:
    python scripts/make_split.py <source-dataset> <out-dir> [--eval-fraction 0.2] [--seed 0]
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import shutil

# ---------------------------------------------------------------- tiny helpers

read_jsonl = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
write_jsonl = lambda p, rows: Path(p).write_text("".join(json.dumps(r) + "\n" for r in rows))
read_json = lambda p: json.loads(Path(p).read_text())
write_json = lambda p, obj: Path(p).write_text(json.dumps(obj, indent=4))

video_keys = lambda info: [k for k, v in info["features"].items() if v["dtype"] == "video"]


# ------------------------------------------------------------------- splitting

def choose_eval_ids(episodes, eval_fraction, seed):
    """Pick `eval_fraction` of each task's episodes, so eval covers every task."""
    by_task = defaultdict(list)
    for ep in episodes:
        by_task[ep["tasks"][0]].append(ep["episode_index"])

    rng = random.Random(seed)
    chosen = []
    for task in sorted(by_task):
        ids = sorted(by_task[task])
        chosen += rng.sample(ids, round(len(ids) * eval_fraction))
    return sorted(chosen)


def episode_files(info, episode_id):
    """Relative paths of every file belonging to one episode: parquet + one video per camera."""
    chunk = episode_id // info["chunks_size"]
    parquet = info["data_path"].format(episode_chunk=chunk, episode_index=episode_id)
    videos = [
        info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=episode_id)
        for key in video_keys(info)
    ]
    return [parquet] + videos


def build_dataset(source, dest, keep_ids, modality_json):
    """Write a standalone LeRobot dataset at `dest` holding only `keep_ids`.

    Episode indices keep their original values (with gaps). The loader resolves
    file paths through each episode's own "episode_index" field, so gaps are fine
    and an episode id means the same thing in both splits.
    """
    source, dest = Path(source), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)

    info = read_json(source / "meta/info.json")
    keep = set(keep_ids)
    episodes = [ep for ep in read_jsonl(source / "meta/episodes.jsonl") if ep["episode_index"] in keep]

    (dest / "meta").mkdir(parents=True)
    write_jsonl(dest / "meta/episodes.jsonl", episodes)
    write_jsonl(
        dest / "meta/episodes_stats.jsonl",
        [s for s in read_jsonl(source / "meta/episodes_stats.jsonl") if s["episode_index"] in keep],
    )
    shutil.copy(source / "meta/tasks.jsonl", dest / "meta/tasks.jsonl")
    shutil.copy(modality_json, dest / "meta/modality.json")

    total_frames = sum(ep["length"] for ep in episodes)
    write_json(dest / "meta/info.json", info | {
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_videos": len(episodes) * len(video_keys(info)),
        "splits": {"train": f"0:{len(episodes)}"},
    })

    for episode_id in keep_ids:
        for relative in episode_files(info, episode_id):
            (dest / relative).parent.mkdir(parents=True, exist_ok=True)
            os.link(source / relative, dest / relative)  # same filesystem: no extra bytes

    return {"episodes": len(episodes), "frames": total_frames}


# ------------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Source LeRobot dataset directory")
    parser.add_argument("out_dir", help="Where to write <name>-train and <name>-eval")
    parser.add_argument("--splits-dir", default="splits", help="Where to write the episode id lists")
    parser.add_argument("--modality-json", required=True, help="modality.json to install into both datasets")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source)
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    all_ids = sorted(ep["episode_index"] for ep in episodes)

    eval_ids = choose_eval_ids(episodes, args.eval_fraction, args.seed)
    train_ids = [i for i in all_ids if i not in set(eval_ids)]

    assert not (set(train_ids) & set(eval_ids)), "train and eval episodes overlap"
    assert sorted(train_ids + eval_ids) == all_ids, "split lost or duplicated episodes"

    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"source": str(source), "seed": args.seed, "eval_fraction": args.eval_fraction}
    write_json(splits_dir / "train_episodes.json", provenance | {"episodes": train_ids})
    write_json(splits_dir / "eval_episodes.json", provenance | {"episodes": eval_ids})

    out_dir = Path(args.out_dir)
    for name, ids in [("train", train_ids), ("eval", eval_ids)]:
        dest = out_dir / f"{source.name}-{name}"
        counts = build_dataset(source, dest, ids, args.modality_json)
        print(f"{name:5s} {counts['episodes']:3d} episodes  {counts['frames']:6d} frames  -> {dest}")

    print(f"\nGate 3: {len(train_ids)} train + {len(eval_ids)} eval = {len(all_ids)} total, "
          f"intersection empty: {not (set(train_ids) & set(eval_ids))}")


if __name__ == "__main__":
    main()
