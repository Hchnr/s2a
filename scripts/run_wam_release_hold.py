"""CPU fixed-reference release hold, with causal contact and exact baseline gates.

Only --stage prepare reads source evidence. Physics starts only in explicit
workers dispatched by baseline/counterfactual/all. No WAM query is made.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from verify_wam_acceptance import verify_case

from s2a.artifacts import code_fingerprint, load_config, sha256
from s2a.geometry import pose_error
from s2a.gripper import OpeningIntent, opening_position_target
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.task_acceptance import task_outcome

REPO = Path(__file__).resolve().parents[1]
MODES = {"baseline": 0, "hold20": 20, "hold40": 40}
CASE_RECORDS = ["task0_init0", "task0_init1", "task1_init0", "task1_init2"]
CASE_NAMES = ["8800_bowl_init0_s43", "8800_bowl_init1_s43", "8800_wine_init0_s43", "8800_wine_init2_s43"]
NUMERIC_FIELDS = (
    "measured_source",
    "desired_source",
    "after_source",
    "opening_target",
    "opening_after",
    "command",
    "actual_dt",
)
PRE_FIELDS = (
    "measured_source",
    "opening_before",
    "normal_command",
    "contact_before",
    "physical_state_before_sha256",
    "intent_before",
    "intent_after",
    "measured_velocity",
)
IMPLEMENTATION = [
    "scripts/run_wam_release_hold.py",
    "scripts/verify_wam_acceptance.py",
    "src/s2a/artifacts.py",
    "src/s2a/checkpoint_simulation.py",
    "src/s2a/libero_simulation.py",
    "src/s2a/cameras.py",
    "src/s2a/gripper.py",
    "src/s2a/osc.py",
    "src/s2a/geometry.py",
    "src/s2a/task_acceptance.py",
    "src/s2a/action_reference.py",
    "pyproject.toml",
]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    with Path(path).open("x") as stream:
        json.dump(value, stream, default=convert, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_steps(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify_hashes(files):
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError(f"frozen evidence changed: {path}")


def validate_protocol(c, seed):
    if seed != 42 or c["seed"] != 42:
        raise ValueError("study seed must be 42; it does not replace source simulation seeds")
    if (c["workers"], c["steps"], c["stable_steps"], c["control_fps"]) != (2, 600, 40, 20):
        raise ValueError("release-hold budget, concurrency or success tail differs")
    if c["gripper_modes"] != list(MODES) or c["hold_control_steps"] != {"hold20": 20, "hold40": 40}:
        raise ValueError("release-hold modes differ")
    expected_cases = [
        dict(name=n, checkpoint="8800_masked", record=r, sampling_seed=43)
        for n, r in zip(CASE_NAMES, CASE_RECORDS, strict=True)
    ]
    if c["cases"] != expected_cases:
        raise ValueError("release-hold source cases differ")
    expected_trigger = dict(
        normal_open_command=-1.0,
        forced_close_command=1.0,
        contact="any_single_object_native_two_finger_grasp_contact",
        contact_time="current_initial_or_previous_completed_endpoint_before_command",
        first_only=True,
        include_trigger_step=True,
        intent_update="every_step_including_hold",
    )
    if c["trigger"] != expected_trigger:
        raise ValueError("release-hold trigger differs")
    for key, expected in {
        "output_dir": "outputs/wam_release_hold_20261020",
        "report_path": "reports/wam_release_hold_20261020.md",
        "source_config": "configs/wam_independent_20261020.json",
        "simulation_python": "outputs/libero_env311/bin/python",
    }.items():
        if (REPO / c[key]).resolve() != (REPO / expected).resolve():
            raise ValueError(f"release-hold path differs: {key}")


def prepared_protocol(c, root, config_path):
    prepared = read_json(root / "manifest.json")
    if prepared["config"] != c or sha256(config_path) != prepared["config_sha256"]:
        raise ValueError("release-hold protocol changed after preparation")
    verify_hashes(prepared["implementation_sha256"])
    return prepared


def prepare(c, root, config_path):
    if any(root.iterdir()):
        raise FileExistsError("release-hold output must be empty before preparation")
    source_c, source_root = load_config(c["source_config"])
    fixture_path = source_root / "simulation/fixtures.json"
    fixture_data = read_json(fixture_path)
    if fixture_data["config"] != source_c:
        raise ValueError("source fixture protocol changed")
    fixtures = {r["name"]: r for r in fixture_data["records"]}
    variants = {r["name"]: r for r in source_c["checkpoints"]}
    cases = []
    for spec in c["cases"]:
        path = source_root / f"simulation/{spec['checkpoint']}/{spec['record']}_s{spec['sampling_seed']}/result.json"
        integrity = verify_case(path, source_c, fixtures, variants)
        manifest = read_json(path.with_name("manifest.json"))
        sim = manifest["config"]
        if (integrity["steps"], sim["control_fps"], sim["execution_horizon"], sim["seed"]) != (600, 20, 8, 43):
            raise ValueError("source simulation budget/seed differs")
        if sim.get("gripper_mode", "intent") != "intent" or not sim["refresh_controller_after_restore"]:
            raise ValueError("source controller initialization differs")
        files = dict(integrity["files"])
        for p in (Path(sim["tool_contract"]), fixture_path, Path(c["source_config"])):
            files[str(p)] = sha256(p)
        cases.append(
            dict(spec, source_result=str(path), record=manifest["record"], simulation_config=sim, inputs_sha256=files)
        )
    write_json(
        root / "manifest.json",
        dict(
            code=code_fingerprint(),
            config=c,
            config_sha256=sha256(config_path),
            implementation_sha256={p: sha256(REPO / p) for p in IMPLEMENTATION},
            cases=cases,
        ),
    )


def exact_rows(left, right):
    if len(left) != len(right):
        raise ValueError("trajectory comparison coverage differs")
    for key in NUMERIC_FIELDS:
        np.testing.assert_array_equal([r[key] for r in left], [r[key] for r in right])
    for a, b in zip(left, right, strict=True):
        for key in ("success", "saturated", "rate_limited", "objects"):
            if a[key] != b[key]:
                raise ValueError(f"exact replay differs: {key}, step={a['step']}")
        for key in ("physical_state_before_sha256", "physical_state_after_sha256"):
            if key in b and a[key] != b[key]:
                raise ValueError(f"exact replay physical state differs: step={a['step']}")


def trigger_event(row):
    return dict(
        step=row["step"],
        command_start_time_s=row["command_start_time_s"],
        endpoint_time_s=row["endpoint_time_s"],
        contact_observed_at_s=row["command_start_time_s"],
        contact_observation="initial_endpoint" if row["step"] == 0 else "previous_completed_endpoint",
        contact_before=row["contact_before"],
        normal_gripper_command=row["normal_command"][-1],
        opening_target=row["opening_target"],
        opening_before=row["opening_before"],
        desired_vs_current_tcp_error=row["desired_vs_current_tcp_error"],
        physical_state_before_sha256=row["physical_state_before_sha256"],
    )


def audit_rows(rows, original, mode, baseline=None):
    if len(rows) != 600 or [r["step"] for r in rows] != list(range(600)):
        raise ValueError("release-hold replay budget differs")
    np.testing.assert_allclose([r["actual_dt"] for r in rows], 0.05, atol=1e-10, rtol=0)
    for key in ("desired_source", "opening_target"):
        np.testing.assert_array_equal([r[key] for r in rows], [r[key] for r in original])
    np.testing.assert_array_equal([r["normal_command"][-1] for r in rows], [r["command"][-1] for r in original])
    first = next((r["step"] for r in rows if r["normal_command"][-1] == -1 and any(r["contact_before"].values())), None)
    start = first if mode != "baseline" else None
    for i, row in enumerate(rows):
        if i and row["contact_before"] != {n: o["grasp_contact"] for n, o in rows[i - 1]["objects"].items()}:
            raise ValueError("trigger contact is not the previous endpoint")
        active = start is not None and start <= i < start + MODES[mode]
        remaining = max(0, start + MODES[mode] - i) if start is not None and i >= start else 0
        np.testing.assert_array_equal(row["command"][:6], row["normal_command"][:6])
        expected = 1.0 if active else row["normal_command"][-1]
        if (row["command"][-1], row["hold_active"], row["hold_remaining_before"], row["hold_remaining_after"]) != (
            expected,
            active,
            remaining,
            max(0, remaining - 1),
        ):
            raise ValueError("actual command is not the registered single fixed hold")
    if mode == "baseline":
        exact_rows(rows, original)
    else:
        if baseline is None:
            raise ValueError("counterfactual requires a verified baseline")
        baseline_first = next(
            (r["step"] for r in baseline if r["normal_command"][-1] == -1 and any(r["contact_before"].values())), None
        )
        if first != baseline_first:
            raise ValueError("candidate trigger differs before any intervention")
        end = 600 if start is None else start
        exact_rows(rows[:end], baseline[:end])
        if start is not None:
            for key in PRE_FIELDS:
                if rows[start][key] != baseline[start][key]:
                    raise ValueError(f"trigger command-start state differs: {key}")
    return dict(
        fixed_world_reference_exact=True,
        normal_gripper_commands_source_exact=True,
        baseline_source_exact=mode == "baseline",
        prefix_and_trigger_prestate_exact=mode != "baseline",
        no_trigger_full_baseline_exact=mode != "baseline" and start is None,
        first_eligible_open_with_contact=trigger_event(rows[first]) if first is not None else None,
        intervention_trigger=trigger_event(rows[start]) if start is not None else None,
        hold_control_steps=sum(r["hold_active"] for r in rows),
        changed_gripper_command_steps=sum(r["command"][-1] != r["normal_command"][-1] for r in rows),
    )


def checked_run(root, case, mode):
    dest = root / "physics" / case["name"] / mode
    integrity = read_json(dest / "integrity.json")
    verify_hashes(integrity["files_sha256"])
    result = read_json(dest / "result.json")
    if result["error"] is not None or result["steps"] != result["expected_steps"] or result["steps"] != 600:
        raise ValueError("incomplete or invalid release-hold run")
    return result, read_steps(dest / "steps.jsonl"), {str(dest / "integrity.json"): sha256(dest / "integrity.json")}


def baseline_gate(c, root, prepared):
    files = {}
    for case in prepared["cases"]:
        verify_hashes(case["inputs_sha256"])
        result, rows, hashes = checked_run(root, case, "baseline")
        audit = audit_rows(rows, read_steps(Path(case["source_result"]).with_name("steps.jsonl")), "baseline")
        if audit != result["verification"]:
            raise ValueError("baseline exact-reproduction summary differs")
        files.update(hashes)
    return files


def run_worker(c, root, prepared, name, mode):
    from PIL import Image

    from s2a.cameras import render_current_cameras
    from s2a.checkpoint_simulation import object_progress
    from s2a.libero_simulation import (
        aperture,
        canvas,
        controller_contract,
        current_observation,
        make_environment,
        measured_twist,
        source_pose,
    )

    if Path(sys.prefix).resolve() != (REPO / c["simulation_python"]).parent.parent.resolve():
        raise ValueError("physics worker requires the registered isolated simulation Python")
    case = next(x for x in prepared["cases"] if x["name"] == name)
    verify_hashes(case["inputs_sha256"])
    baseline_inputs = baseline_gate(c, root, prepared) if mode != "baseline" else {}
    dest = root / "physics" / name / mode
    dest.mkdir(parents=True, exist_ok=False)
    source = Path(case["source_result"])
    original = read_steps(source.with_name("steps.jsonl"))
    source_chunks = read_json(source.with_name("chunks.json"))
    sim = case["simulation_config"]  # Original seed43 and every controller/environment option are retained.
    h = np.asarray(read_json(sim["tool_contract"])["source_from_tcp"])
    baseline = read_steps(root / "physics" / name / "baseline/steps.jsonl") if mode != "baseline" else None
    write_json(
        dest / "manifest.json",
        dict(
            code=code_fingerprint(),
            source=case,
            config=c,
            mode=mode,
            implementation_sha256=prepared["implementation_sha256"],
            baseline_inputs_sha256=baseline_inputs,
            contact_privilege="simulator any-single-object native two-finger contact, available before command",
            intermediate_state_restore=False,
            model_query=False,
            reference_reanchor=False,
        ),
    )
    env, rows, renders, error, initial_objects = None, [], [], None, None
    first_eligible, start = None, None
    try:
        env, _ = make_environment(sim, dest, case["record"], 0)
        obs = current_observation(env)
        initial_opening = aperture(obs)
        previous = object_progress(env)
        initial_objects = previous["objects"]
        _, contract = controller_contract(env)
        expected = read_json(source.with_name("controller.json"))
        if set(contract) != set(expected):
            raise ValueError("controller contract keys differ")
        for key in contract:
            np.testing.assert_array_equal(contract[key], expected[key])
        if not np.isclose(env.env.control_timestep, 1 / sim["control_fps"]):
            raise ValueError("control period differs")
        converter = OSCStateToAction(
            OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max"))),
            contract["kp"],
            contract["kd"],
            1 / sim["control_fps"],
            compensate=True,
            inverse_weight=sim["inverse_weight"],
            max_command_rate=sim["max_command_rate"],
        )
        intent = OpeningIntent(initial_opening, band=sim["opening_hysteresis"])
        env.robots[0].gripper.current_action = opening_position_target(initial_opening)
        write_json(dest / "controller.json", contract)
        with (dest / "steps.jsonl").open("x") as stream:
            for step, reference in enumerate(original):
                state_before = hashlib.sha256(np.asarray(env.sim.get_state().flatten()).tobytes()).hexdigest()
                if step % sim["execution_horizon"] == 0:
                    images = render_current_cameras(env, sim["render_resolution"])
                    pixels = canvas(
                        [images[k][::-1, ::-1].copy() for k in ("agentview_image", "robot0_eye_in_hand_image")]
                    )
                    pixel_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
                    if mode == "baseline" and (
                        pixel_sha != source_chunks[step // 8]["image_sha256"]
                        or state_before != source_chunks[step // 8]["physics_sha256"]
                    ):
                        raise ValueError("baseline chunk physical state or rendered source pixels differ")
                    renders.append(
                        dict(
                            step=step,
                            command_start_time_s=step / 20,
                            canvas_pixels_sha256=pixel_sha,
                            raw_camera_pixels_sha256={
                                k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in images.items()
                            },
                        )
                    )
                    if step in (0, 592):
                        Image.fromarray(pixels).save(dest / ("initial_input.png" if step == 0 else "last_input.png"))
                measured = source_pose(obs)
                opening_before = aperture(obs)
                desired, opening = np.asarray(reference["desired_source"]), float(reference["opening_target"])
                converted = converter.convert(measured, desired, measured_twist(env))
                intent_before = dict(band=intent.band, extreme=intent.extreme, direction=intent.direction)
                normal_grip = intent.update(opening)  # Always update, including every overridden hold step.
                intent_after = dict(band=intent.band, extreme=intent.extreme, direction=intent.direction)
                if normal_grip != reference["command"][-1]:
                    raise ValueError("normal intent gripper command differs from source")
                contact_before = {n: o["grasp_contact"] for n, o in previous["objects"].items()}
                eligible = normal_grip == -1.0 and any(contact_before.values())
                if first_eligible is None and eligible:
                    first_eligible = step
                    if mode != "baseline":
                        start = step
                active = start is not None and start <= step < start + MODES[mode]
                remaining = max(0, start + MODES[mode] - step) if start is not None else 0
                normal_command = np.r_[converted["command"], normal_grip]
                command = normal_command.copy()
                if active:
                    command[-1] = 1.0
                before_time = float(env.sim.data.time)
                env.step(command)
                obs = current_observation(env)
                after = source_pose(obs)
                progress = object_progress(env)
                for key, obj in progress["objects"].items():
                    obj["height_gain_m"] = float(obj["position"][2] - initial_objects[key]["position"][2])
                    obj["displacement_m"] = float(np.linalg.norm(obj["position"] - initial_objects[key]["position"]))
                row = dict(
                    step=step,
                    command_start_time_s=step / 20,
                    endpoint_time_s=(step + 1) / 20,
                    measured_source=measured,
                    desired_source=desired,
                    after_source=after,
                    opening_target=opening,
                    opening_before=opening_before,
                    opening_after=aperture(obs),
                    normal_command=normal_command,
                    command=command,
                    normal_gripper_command=normal_grip,
                    intent_before=intent_before,
                    intent_after=intent_after,
                    contact_before=contact_before,
                    hold_active=active,
                    hold_remaining_before=remaining,
                    hold_remaining_after=max(0, remaining - 1),
                    measured_velocity=converted["measured_velocity"],
                    rotation_inverse_enabled=converted["rotation_inverse_enabled"],
                    desired_vs_current_tcp_error=pose_error(desired @ h, measured @ h),
                    tracking_tcp_error=pose_error(desired @ h, after @ h),
                    actual_dt=float(env.sim.data.time) - before_time,
                    saturated=converted["saturated"],
                    rate_limited=converted["rate_limited"],
                    physical_state_before_sha256=state_before,
                    physical_state_after_sha256=hashlib.sha256(
                        np.asarray(env.sim.get_state().flatten()).tobytes()
                    ).hexdigest(),
                    **progress,
                )
                rows.append(json.loads(json.dumps(row, default=lambda v: v.tolist(), allow_nan=False)))
                stream.write(json.dumps(rows[-1], allow_nan=False) + "\n")
                stream.flush()
                previous = progress  # Only this candidate's completed endpoint can trigger its next command.
        images = render_current_cameras(env, sim["render_resolution"])
        Image.fromarray(
            canvas([images[k][::-1, ::-1].copy() for k in ("agentview_image", "robot0_eye_in_hand_image")])
        ).save(dest / "final_input.png")
    except Exception as exc:
        error = repr(exc)
    finally:
        if env is not None:
            env.close()
    result = dict(
        name=name,
        mode=mode,
        steps=len(rows),
        expected_steps=600,
        error=error,
        simulation_seed=sim["seed"],
        study_seed=c["seed"],
        requested_hold_steps=MODES[mode],
    )
    if error is None:
        try:
            result["verification"] = audit_rows(rows, original, mode, baseline)
            errors = pose_error(
                np.asarray([r["desired_source"] for r in rows]) @ h, np.asarray([r["after_source"] for r in rows]) @ h
            )
            result.update(
                outcome=task_outcome([r["success"] for r in rows], expected_steps=600, stable_steps=40),
                position_rmse_tcp_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
                rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
                object_progress={
                    n: dict(
                        grasp_contact_steps=sum(r["objects"][n]["grasp_contact"] for r in rows),
                        max_height_gain_m=max(r["objects"][n]["height_gain_m"] for r in rows),
                        max_displacement_m=max(r["objects"][n]["displacement_m"] for r in rows),
                    )
                    for n in initial_objects
                },
            )
            verify_hashes(case["inputs_sha256"])
            verify_hashes(prepared["implementation_sha256"])
            verify_hashes(baseline_inputs)
        except Exception as exc:
            result["error"] = repr(exc)
    write_json(dest / "render_hashes.json", renders)
    write_json(dest / "result.json", result)
    write_json(
        dest / "integrity.json", dict(files_sha256={str(p): sha256(p) for p in sorted(dest.rglob("*")) if p.is_file()})
    )
    if result["error"] is not None:
        raise RuntimeError(result["error"])
    return result


def execute(c, root, config_path, prepared, modes):
    if any(mode != "baseline" for mode in modes):
        baseline_gate(c, root, prepared)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
        LP_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
    )

    def worker(job):
        case, mode = job
        logfile = root / f"{case['name']}_{mode}.log"
        cmd = [
            str((REPO / c["simulation_python"]).absolute()),
            __file__,
            "--config",
            str(Path(config_path).resolve()),
            "--seed",
            "42",
            "--worker",
            case["name"],
            "--mode",
            mode,
        ]
        with logfile.open("x") as stream:
            process = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=REPO)
        if process.returncode:
            raise RuntimeError(f"release-hold worker failed: {logfile}")
        print(json.dumps(dict(case=case["name"], mode=mode, complete=True)), flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, [(case, mode) for case in c["cases"] for mode in modes]))


def report(c, root, prepared):
    destination, document = root / "summary.json", REPO / c["report_path"]
    if destination.exists() or document.exists():
        raise FileExistsError("release-hold report already exists")
    baseline_gate(c, root, prepared)
    results, files = [], {str(root / "manifest.json"): sha256(root / "manifest.json")}
    for case in prepared["cases"]:
        original = read_steps(Path(case["source_result"]).with_name("steps.jsonl"))
        baseline = checked_run(root, case, "baseline")[1]
        paired = {}
        for mode in MODES:
            result, rows, hashes = checked_run(root, case, mode)
            audit = audit_rows(rows, original, mode, baseline if mode != "baseline" else None)
            if audit != result["verification"] or result["outcome"] != task_outcome(
                [r["success"] for r in rows], expected_steps=600, stable_steps=40
            ):
                raise ValueError("release-hold report summary differs from raw evidence")
            results.append(result)
            paired[mode] = rows
            files.update(hashes)
            files.update(read_json(root / "physics" / case["name"] / mode / "integrity.json")["files_sha256"])
            log = root / f"{case['name']}_{mode}.log"
            files[str(log)] = sha256(log)
        first = next(
            (r["step"] for r in baseline if r["normal_command"][-1] == -1 and any(r["contact_before"].values())), None
        )
        end = 600 if first is None else min(600, first + 20)
        exact_rows(paired["hold20"][:end], paired["hold40"][:end])
    lines = [
        "# 固定WAM参考的首次接触后释放 hold 诊断 — 2026-10-20",
        "",
        "4条事后选择的保存world参考×baseline/hold20/hold40，共12条CPU物理重放，各600步。触发使用sim privileged的任意单个物体双指contact，不读取目标身份。",
        "",
        "原OpeningIntent每步持续更新；首次正常open且命令前当前/上一端点已有contact时，仅将实际夹爪强制close20或40控制步（含触发步），之后恢复正常命令，不重触发。",
        "",
        "| source | mode | trigger command-start s | hold steps | changed commands | ever | terminal | stable | TCP RMSE mm | SO(3) rad |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        v, o = row["verification"], row["outcome"]
        trigger = v["intervention_trigger"]
        t = f"{trigger['command_start_time_s']:.2f}" if trigger else "none"
        lines.append(
            f"| {row['name']} | {row['mode']} | {t} | {v['hold_control_steps']} | {v['changed_gripper_command_steps']} | {int(o['ever'])} | {int(o['terminal'])} | {int(o['stable'])} | {row['position_rmse_tcp_m'] * 1000:.3f} | {row['rotation_rmse_rad']:.5f} |"
        )
    lines += [
        "",
        "全部baseline须精确复现原命令、pose、开度、物体、接触、任务谓词与chunk物理/像素SHA；干预前缀含触发步命令前状态但不含被覆盖命令。无触发全轨迹精确一致；hold20/40触发及最初20个干预步一致。",
        "",
        "固定原desired_source/opening_target及arm算法/参数/限速，每个反事实使用自己的当前反馈，所以干预后arm命令数值允许变化。原source seed43不被study seed42覆盖。没有WAM查询、重规划、重新锚定、途中state恢复或额外控制步。",
        "",
        "这是局部sim privileged执行反事实，不是新WAM闭环或独立成功率。后续参考仍来自原轨迹，不能证明反事实重新观测后模型会生成这些参考；只检验有限时长强制保持的总影响，不能推广至所有失败、部署传感器或唯一权重/释放因果。flattened physics SHA也不认证所有未保存的controller隐藏状态。",
        "",
        "完整normal/actual command、命令前contact、触发时刻、开度、canonical TCP误差、原始对象/任务日志、源及产物SHA见outputs/wam_release_hold_20261020/。",
        "",
    ]
    with document.open("x") as stream:
        stream.write("\n".join(lines))
    write_json(
        destination,
        dict(
            code=code_fingerprint(),
            config=c,
            cases=results,
            files_sha256=files,
            hold20_hold40_common_prefix_exact=True,
            report_sha256=sha256(document),
            independent_wam_success_rate=False,
        ),
    )
    print(document, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_release_hold_20261020.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("prepare", "baseline", "counterfactual", "report", "all"), default="all")
    parser.add_argument("--worker", choices=CASE_NAMES)
    parser.add_argument("--mode", choices=list(MODES))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("explicit CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' are required")
    c, root = load_config(args.config)
    validate_protocol(c, args.seed)
    if args.worker:
        if args.mode is None:
            parser.error("--worker requires --mode")
        prepared = prepared_protocol(c, root, args.config)
        print(json.dumps(run_worker(c, root, prepared, args.worker, args.mode)), flush=True)
        return
    if args.mode is not None:
        parser.error("--mode is worker-only")
    if args.stage in ("prepare", "all"):
        prepare(c, root, args.config)
    prepared = prepared_protocol(c, root, args.config)
    if args.stage in ("baseline", "all"):
        execute(c, root, args.config, prepared, ["baseline"])
    if args.stage in ("counterfactual", "all"):
        execute(c, root, args.config, prepared, ["hold20", "hold40"])
    if args.stage in ("report", "all"):
        report(c, root, prepared)


if __name__ == "__main__":
    main()
