"""Report episodes whose declared length exceeds what the files actually hold.

The loader trusts `min(len(parquet), meta["length"])` and then asks the video
decoder for exactly that many frames. If a video is shorter than that, decoding
raises IndexError partway into training. This finds those episodes up front.

With --fix, each short episode's declared length is clamped to what the files
actually hold. That is only sound when the video is a clean prefix of the
episode -- verify the frame rate and final timestamp line up before trusting it.

Usage:
    python scripts/check_frames.py <dataset-dir> [--fix]
"""

import json
from pathlib import Path
import sys

import pandas as pd
from torchcodec.decoders import VideoDecoder

read_jsonl = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
video_keys = lambda info: [k for k, v in info["features"].items() if v["dtype"] == "video"]


def episode_frame_counts(dataset, info, episode_id):
    """Frames available per source for one episode: the parquet and each video."""
    chunk = episode_id // info["chunks_size"]
    parquet = dataset / info["data_path"].format(episode_chunk=chunk, episode_index=episode_id)
    counts = {"parquet": len(pd.read_parquet(parquet))}
    for key in video_keys(info):
        path = dataset / info["video_path"].format(
            episode_chunk=chunk, video_key=key, episode_index=episode_id
        )
        counts[key] = VideoDecoder(str(path), device="cpu").metadata.num_frames
    return counts


def main(dataset, fix=False):
    dataset = Path(dataset)
    info = json.loads((dataset / "meta/info.json").read_text())
    episodes = read_jsonl(dataset / "meta/episodes.jsonl")

    short = {}
    for episode in episodes:
        episode_id, declared = episode["episode_index"], episode["length"]
        counts = episode_frame_counts(dataset, info, episode_id)
        usable = min(counts.values())
        if usable < declared:
            short[episode_id] = usable
            print(f"episode {episode_id:3d}  declared={declared:5d}  usable={usable:5d}  {counts}")

    print(f"\n{len(short)} of {len(episodes)} episodes are short.")

    if fix and short:
        for episode in episodes:
            episode["length"] = min(episode["length"], short.get(episode["episode_index"], 1 << 30))
        (dataset / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in episodes)
        )
        info["total_frames"] = sum(e["length"] for e in episodes)
        (dataset / "meta/info.json").write_text(json.dumps(info, indent=4))
        print(f"fixed: clamped {len(short)} episode(s); total_frames now {info['total_frames']}")
    return short


if __name__ == "__main__":
    main(sys.argv[1], fix="--fix" in sys.argv)
