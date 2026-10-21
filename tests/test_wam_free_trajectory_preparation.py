"""029 preparation keeps source observations closed and exact sampling pairs."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prepare_wam_free_trajectory")


def roster():
    anchors = [
        dict(
            name=f"ep{ep}_a{a}",
            episode=ep,
            split="validation",
            valid_future_steps=40,
            image_path=f"/never-open/ep{ep}_a{a}.png",
        )
        for ep in range(6)
        for a in range(4)
    ]
    rows = [
        dict(
            name=f"{a['name']}_s{s}",
            anchor_name=a["name"],
            sampling_seed=s,
            variant="raw",
            generation_index=i * 2,
            request=dict(seed=s, vision_path=a["image_path"]),
            **{
                k: dict(path=f"/{i}_{k}.json", sha256="h")
                for k in ("response", "conditioning", "initial", "actual_sigma_trace")
            },
        )
        for i, (a, s) in enumerate((a, s) for a in anchors for s in (42, 43))
    ]
    config = dict(
        validation_episodes=list(range(6)),
        selected_step_indices=[19, 29, 33],
        selected_actual_sigmas=[0.8, 0.5, 0.2],
        precision_step_indices=[0, 19, 29, 33, 34],
    )

    class Evidence:
        def ref(self, ref):
            seed = rows[int(Path(ref["path"]).name.split("_")[0])]["sampling_seed"]
            steps = [dict(solver_sigma=1 - i / 35, solver_timestep=999 - i) for i in range(35)]
            for i, s in zip(config["selected_step_indices"], config["selected_actual_sigmas"], strict=True):
                steps[i] = dict(solver_sigma=s, solver_timestep=int(s * 1000))
            return dict(complete=True, steps=steps, actual_sampler_arguments=dict(num_steps=35, shift=5.0, seed=[seed]))

    return dict(anchors=anchors), dict(generations=rows), config, Evidence()


def test_preparation_never_opens_declared_episode_or_png(module, tmp_path):
    episode, png, source = tmp_path / "episode.npz", tmp_path / "current.png", tmp_path / "source.py"
    source.write_text("valid source")
    evidence = module.Evidence()
    skipped = module.safe_hashes(
        {str(episode): "declared-only", str(png): "declared-image", str(source): module.sha256(source)},
        {str(episode)},
        evidence,
    )
    assert skipped == {str(episode): "declared-only", str(png): "declared-image"}
    assert set(evidence.hashes) == {str(source)}
    assert not episode.exists() and not png.exists()


def test_raw48_and_teacher144_keep_both_seeds_exactly(module):
    anchors, free, teacher, nodes = module.rosters(*roster())
    assert len(anchors) == 24 and len(free) == 48 and len(teacher) == 144 and len(nodes) == 5
    assert free[3]["generation_index"] == 3 and free[3]["source_generation_index"] == 6
    assert [(t["sampling_seed"], t["step_index"]) for t in teacher[:6]] == [
        (42, 19),
        (42, 29),
        (42, 33),
        (43, 19),
        (43, 29),
        (43, 33),
    ]


@pytest.mark.parametrize("change", ["seed", "order", "train"])
def test_sampling_roster_rejects_seed_order_or_training_leak(module, change):
    source, sampling, config, evidence = roster()
    if change == "seed":
        sampling["generations"][0]["request"]["seed"] = 43
    elif change == "order":
        sampling["generations"][0], sampling["generations"][1] = sampling["generations"][1], sampling["generations"][0]
    else:
        source["anchors"][0]["split"] = "train"
    with pytest.raises(ValueError):
        module.rosters(source, sampling, config, evidence)
