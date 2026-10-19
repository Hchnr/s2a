"""Export actual sampled simulator cameras, never the model's imagined video."""

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

from s2a.artifacts import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_wam_effect_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    root = Path(c["output_dir"])
    dest = root / "videos"
    dest.mkdir(exist_ok=True)
    rows = []
    for case in ("wam_affine", "wam_response"):
        source = root / "sync" / case / "trial0"
        result = json.loads((source / "result.json").read_text())
        if result["error"] is not None or result["steps"] != c["max_steps"]:
            raise ValueError("source task is incomplete")
        inputs = sorted(source.glob("input_*.png"))
        if len(inputs) != result["chunks"]:
            raise ValueError("missing actual camera frames")
        path = dest / f"{case}_trial0_actual_cameras.mp4"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-framerate",
                str(c["control_fps"] / c["execution_horizon"]),
                "-i",
                str(source / "input_%04d.png"),
                "-vf",
                "scale=320:480",
                "-c:v",
                "libx264",
                "-crf",
                "24",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )
        rows.append(
            dict(
                path=str(path),
                sha256=sha256(path),
                frames=len(inputs),
                physical_frame_interval_s=c["execution_horizon"] / c["control_fps"],
                inputs_sha256={str(p): sha256(p) for p in inputs},
            )
        )
    write_json(
        dest / "manifest.json",
        dict(
            cases=rows,
            evidence="Actual simulator cameras captured before WAM requests at 0.4s physical intervals, encoded at 2.5 fps. Not model-generated video. Last physical interval is represented by the step log, not an additional camera frame.",
        ),
    )
    print(dest)


if __name__ == "__main__":
    main()
