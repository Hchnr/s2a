"""Compare frozen native RGB with cached published RGB under preregistered hypotheses.

Run with outputs/libero_env311/bin/python. This stage reads images only; it does
not import a simulator, decode video, infer WAM actions or select a true offset.
"""

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORIENTATIONS = ["identity", "vertical_flip", "horizontal_flip", "both_flip"]
OFFSETS = [0, -1, 1, -2, 2]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def resolve_path(value):
    path = Path(value)
    return (REPO / path).resolve() if not path.is_absolute() else path.resolve()


def verify_inputs(config):
    paths = {}
    for name, spec in config["inputs"].items():
        path = resolve_path(spec["path"])
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"frozen input changed: {name}: {path}")
        paths[name] = path
    return paths


def validate_protocol(config, seed):
    if seed != 42 or config["seed"] != 42:
        raise ValueError("this diagnostic requires seed 42")
    if Path(sys.prefix).resolve() != resolve_path(Path(config["reader_python"]).parent.parent):
        raise ValueError("run with the configured outputs/libero_env311/bin/python HDF5 reader")
    if config["orientation_hypotheses"] != ORIENTATIONS or config["primary_orientation"] != "identity":
        raise ValueError("orientation hypotheses differ from the preregistered four")
    if config["published_index_minus_native_index_offsets"] != OFFSETS:
        raise ValueError("offset hypotheses differ from the preregistered five")
    if (
        config["native_demo"] != "demo_4"
        or config["native_record"] != "task1_init2"
        or config["r1_episode"] != 683
        or config["r2_episode"] != 1139
        or config["frame_count"] != 158
        or config["comparison_panel_frames"] != [13, 73, 90]
        or config["native_image_shape"] != [128, 128, 3]
        or config["published_image_shape"] != [256, 256, 3]
        or config["expected_native_image_convention"] != "opengl"
    ):
        raise ValueError("source, shapes or panel selection differs from the fixed study")
    expected_roles = [
        dict(role="auxiliary_0", native_rgb_dataset="agentview_rgb", r2_source_key="observation.images.image"),
        dict(role="left_wrist", native_rgb_dataset="eye_in_hand_rgb", r2_source_key="observation.images.wrist_image"),
    ]
    if config["roles"] != expected_roles:
        raise ValueError("nominal camera-role correspondence differs")
    if config["resize"] != dict(
        method="Pillow.Image.Resampling.BILINEAR",
        target_width=256,
        target_height=256,
        purpose="Deterministic comparison resize; does not claim historical R1 video encoding resize was reproduced.",
    ):
        raise ValueError("comparison resize differs")
    destination = resolve_path(config["output_dir"])
    if destination != REPO / "outputs/wam_video_pose_timing_20261020":
        raise ValueError("unexpected diagnostic output directory")
    if destination.exists():
        raise FileExistsError(destination)
    return destination


def validate_identity(config, paths):
    source = read_json(paths["source_audit"])
    matches = [row for row in source["cases"] if row["record"] == config["native_record"]]
    if (
        len(matches) != 1
        or matches[0]["demo"] != config["native_demo"]
        or len(matches[0]["r1_matches"]) != 1
        or matches[0]["r1_matches"][0]["r1_episode"] != config["r1_episode"]
        or [row["r2_episode"] for row in matches[0]["r2"]] != [config["r2_episode"]]
    ):
        raise ValueError("exact native-command source identity is not the frozen mapping")
    samples = [row for row in read_json(paths["sample_manifest"])["cases"] if row["episode"] == 1139]
    if len(samples) != 4 or any(
        resolve_path(row["arrays_path"]) != paths["published_npz"]
        or row["arrays_sha256"] != config["inputs"]["published_npz"]["sha256"]
        for row in samples
    ):
        raise ValueError("cached arrays are not the four frozen episode1139 phase samples")
    snapshot = config["r2_metadata_snapshot"]
    if (
        snapshot["episode_index"] != 1139
        or snapshot["fps"] != 20
        or snapshot["source_row_start"] != 0
        or snapshot["source_row_end"] != 158
        or json.loads(snapshot["source_episode_id"])["source_id"].split(":")[-1] != "683"
        or snapshot["view_mapping"] != {row["role"]: row["r2_source_key"] for row in config["roles"]}
    ):
        raise ValueError("frozen metadata snapshot does not describe the mapped episode")
    for role in config["roles"]:
        views = [view for view in snapshot["video_views"] if view["role"] == role["role"]]
        if len(views) != 1 or any(
            views[0][key] != expected
            for key, expected in dict(source_key=role["r2_source_key"], width=256, height=256, fps=20).items()
        ):
            raise ValueError("frozen metadata camera role or dimensions differ")
    with paths["r2_processing"].open() as stream:
        processing = json.loads(next(itertools.islice(stream, 1139, 1140)))
    native = processing["source_processing"]["source"]
    if processing["episode_index"] != 1139 or native["episode_index"] != 683 or native["row_range"] != [0, 158]:
        raise ValueError("R2 processing record differs from the source mapping")
    return dict(source_match=matches[0], sample_names=[row["name"] for row in samples])


def orient(image, hypothesis):
    if hypothesis == "identity":
        return image
    if hypothesis == "vertical_flip":
        return image[::-1]
    if hypothesis == "horizontal_flip":
        return image[:, ::-1]
    if hypothesis == "both_flip":
        return image[::-1, ::-1]
    raise ValueError(hypothesis)


def load_images(config, paths):
    import h5py
    import numpy as np
    from PIL import Image

    native, published = {}, {}
    with h5py.File(paths["native_hdf5"], "r") as source, np.load(paths["published_npz"], allow_pickle=False) as cache:
        convention = source["data"].attrs["macros_image_convention"]
        if isinstance(convention, bytes):
            convention = convention.decode()
        if convention != config["expected_native_image_convention"]:
            raise ValueError("native HDF image convention differs")
        observations = source[f"data/{config['native_demo']}/obs"]
        for role in config["roles"]:
            name = role["role"]
            raw = observations[role["native_rgb_dataset"]][:]
            released = cache[name]
            if raw.shape != (158, 128, 128, 3) or released.shape != (158, 256, 256, 3):
                raise ValueError(f"incomplete or unexpected image array: {name}")
            if raw.dtype != np.uint8 or released.dtype != np.uint8:
                raise ValueError(f"requires RGB uint8 images: {name}")
            native[name] = np.stack(
                [np.asarray(Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR)) for image in raw]
            )
            published[name] = released.copy()
    return native, published


def compare(config, native, published):
    import numpy as np

    records, summaries = [], []
    for role in config["roles"]:
        name = role["role"]
        for hypothesis in ORIENTATIONS:
            for offset in OFFSETS:
                values = []
                for published_index in range(config["frame_count"]):
                    native_index = published_index - offset
                    row = dict(
                        role=name,
                        orientation_hypothesis=hypothesis,
                        published_index_minus_native_index=offset,
                        published_index=published_index,
                        native_index=native_index,
                        status="out_of_range",
                        mae_rgb_0_255=None,
                        mse_rgb_0_255=None,
                        psnr_db=None,
                        pixel_identical=None,
                    )
                    if 0 <= native_index < config["frame_count"]:
                        difference = orient(native[name][native_index], hypothesis).astype(np.float64)
                        difference -= published[name][published_index]
                        mae = float(np.mean(abs(difference)))
                        mse = float(np.mean(difference * difference))
                        row.update(
                            status="compared",
                            mae_rgb_0_255=mae,
                            mse_rgb_0_255=mse,
                            psnr_db=None if mse == 0 else 10 * math.log10(255**2 / mse),
                            pixel_identical=mse == 0,
                        )
                        values.append(row)
                    records.append(row)
                summaries.append(
                    dict(
                        role=name,
                        orientation_hypothesis=hypothesis,
                        published_index_minus_native_index=offset,
                        expected_records=config["frame_count"],
                        compared_frames=len(values),
                        out_of_range_frames=config["frame_count"] - len(values),
                        mean_mae_rgb_0_255=float(np.mean([row["mae_rgb_0_255"] for row in values])),
                        mean_mse_rgb_0_255=float(np.mean([row["mse_rgb_0_255"] for row in values])),
                        pixel_identical_frames=sum(row["pixel_identical"] for row in values),
                    )
                )
    if len(records) != 6320 or sum(row["status"] == "compared" for row in records) != 6272:
        raise ValueError("predeclared comparison matrix coverage differs")
    return records, summaries


def save_panels(config, native, published, destination):
    from PIL import Image, ImageDraw

    paths = []
    for frame in config["comparison_panel_frames"]:
        panel = Image.new("RGB", (5 * 256, 2 * (256 + 24)), "white")
        draw = ImageDraw.Draw(panel)
        for row, role in enumerate(config["roles"]):
            name, top = role["role"], row * (256 + 24)
            images = [published[name][frame]] + [orient(native[name][frame], hypothesis) for hypothesis in ORIENTATIONS]
            titles = [f"published {name} [{frame}]"] + [f"native [{frame}] {value}" for value in ORIENTATIONS]
            for column, (pixels, title) in enumerate(zip(images, titles, strict=True)):
                draw.text((column * 256 + 4, top + 4), title, fill="black")
                panel.paste(Image.fromarray(pixels), (column * 256, top + 24))
        path = destination / f"same_index_frame{frame:04d}.png"
        with path.open("xb") as stream:
            panel.save(stream, format="PNG")
        paths.append(path)
    return paths


def main():
    # This entry never imports GPU packages. Set the environment before optional
    # image/HDF imports and record it in the output manifest.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["LD_LIBRARY_PATH"] = ""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_video_pose_timing_20261020.json")
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = read_json(config_path)
    destination = validate_protocol(config, args.seed)
    initial_code_sha, initial_config_sha = sha256(__file__), sha256(config_path)
    paths = verify_inputs(config)
    identity = validate_identity(config, paths)
    native, published = load_images(config, paths)
    records, summaries = compare(config, native, published)
    verify_inputs(config)
    if sha256(__file__) != initial_code_sha or sha256(config_path) != initial_config_sha:
        raise ValueError("diagnostic code or configuration changed while reading inputs")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "comparisons.json", records)
    with (destination / "comparisons.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    panels = save_panels(config, native, published, destination)
    summary = dict(
        complete_comparison_matrix=True,
        physical_image_pose_timing="unresolved",
        native_to_published_orientation_lineage="unresolved",
        nominal_native_to_published_role_lineage="unresolved",
        automatic_true_offset_or_orientation_selection=False,
        index_relation="published_index - native_index = configured offset",
        primary_table=[row for row in summaries if row["orientation_hypothesis"] == "identity"],
        descriptive_competing_hypotheses=[row for row in summaries if row["orientation_hypothesis"] != "identity"],
        coverage=dict(total_records=6320, compared_records=6272, out_of_range_records=48),
        psnr_convention="10*log10(255^2/MSE); psnr_db=null with pixel_identical=true represents positive infinity.",
        limitations=config["limitations"],
        interpretation="All predeclared scores are descriptive; a minimum pixel error does not establish physical time or direction.",
    )
    write_json(destination / "summary.json", summary)
    import h5py
    import numpy as np
    import PIL

    artifacts = [destination / name for name in ("comparisons.json", "comparisons.csv", "summary.json")] + panels
    write_json(
        destination / "manifest.json",
        dict(
            complete=True,
            seed=args.seed,
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            code_sha256=initial_code_sha,
            config_path=str(config_path),
            config_sha256=initial_config_sha,
            config=config,
            input_files_sha256={str(path): config["inputs"][name]["sha256"] for name, path in paths.items()},
            identity=identity,
            metadata_note="R2 metadata snapshot was explicitly frozen in config; raw Parquet SHA is checked without importing PyArrow into the HDF reader.",
            runtime=dict(
                python=sys.version,
                python_prefix=sys.prefix,
                numpy=np.__version__,
                h5py=h5py.__version__,
                pillow=PIL.__version__,
                CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
                LD_LIBRARY_PATH=os.environ["LD_LIBRARY_PATH"],
            ),
            artifact_files_sha256={str(path): sha256(path) for path in artifacts},
            model_loaded=False,
            simulator_started=False,
            new_video_decoding=False,
            physical_image_pose_timing="unresolved",
        ),
    )
    print(json.dumps(dict(output_dir=str(destination), **summary["coverage"], physical_image_pose_timing="unresolved")))


if __name__ == "__main__":
    main()
