"""029 synthetic audit boundaries and independently specified numerical gold."""

import copy
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_wam_free_trajectory.py"
    spec = importlib.util.spec_from_file_location("independent_free_trajectory_checker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def precision_fixture(checker):
    sigmas, times = checker.FLOW.solver_schedule()
    nodes = [
        dict(step_index=i, sigma_actual=float(sigmas[i]), network_timestep=int(times[i])) for i in (0, 19, 29, 33, 34)
    ]
    arrays, cases = {}, []
    for node in nodes:
        for dtype in ("torch.bfloat16", "torch.float32"):
            for payload in ("signed_normal", "boundaries_subnormal"):
                name = f"step{node['step_index']}_{dtype.removeprefix('torch.')}_{payload}"
                sample, uncast = checker.precision_payload(payload)
                velocity = checker.OLD.quantize_bfloat16(uncast) if dtype == "torch.bfloat16" else uncast
                sigma = np.asarray(node["sigma_actual"], np.float32)
                arrays.update({f"{name}_sample": sample.copy(), f"{name}_velocity_input": uncast.copy()})
                case = dict(
                    name=name,
                    node=node,
                    payload=payload,
                    velocity_dtype=dtype,
                    sample_dtype="torch.float32",
                    source_conversion_calls_per_device=1,
                )
                for device in ("cpu", "cuda"):
                    product, result = checker.mixed_convert(sample, velocity, sigma, dtype, device)
                    for key, value in dict(velocity=velocity, product=product, x0=result, sigma=sigma).items():
                        arrays[f"{name}_{device}_{key}"] = value.copy()
                    case[device] = dict(
                        product_dtype=dtype,
                        x0_dtype="torch.float32",
                        sigma_dtype="torch.float32",
                        sigma_device="cpu",
                        actual_input_unchanged=True,
                    )
                case["cpu_cuda_bits_equal"] = {
                    key: arrays[f"{name}_cpu_{key}"].tobytes() == arrays[f"{name}_cuda_{key}"].tobytes()
                    for key in ("velocity", "product", "x0", "sigma")
                }
                cases.append(case)
    return arrays, cases, nodes


def test_precision_full128_bits_cpu_sigma_and_actual_node_coverage(checker):
    arrays, cases, nodes = precision_fixture(checker)
    differences = checker.precision_arrays(arrays, cases, nodes)
    assert len(differences["ideal_fp64_difference"]) == 40
    assert differences["cpu_cuda_full128_bitwise_equal"] is False
    assert differences["cross_backend_equality_is_admission_gate"] is False
    assert nodes[0]["sigma_actual"] != float(np.float32(0.999))
    assert nodes[1]["sigma_actual"] != float(np.float32(0.8))
    key = f"{cases[3]['name']}_cuda_x0"
    arrays[key].view(np.uint32)[1, 127] ^= np.uint32(1)
    with pytest.raises(ValueError, match="full128 bits"):
        checker.precision_arrays(arrays, cases, nodes)
    arrays, cases, nodes = precision_fixture(checker)
    cases[0]["cuda"]["sigma_device"] = "cuda:0"
    with pytest.raises(ValueError, match="dtype differs"):
        checker.precision_arrays(arrays, cases, nodes)
    with pytest.raises(ValueError, match="conversion20 coverage"):
        checker.precision_arrays(arrays, cases[:-1], nodes)


def test_fp32_subtraction_is_separate_from_real_number_inverse(checker):
    sample = np.asarray([[1.0, -1.0, 0.0, -0.0]], np.float32)
    velocity = np.asarray([[1.0, -1.0, 0.0, -0.0]], np.float32)
    sigma = np.asarray(0.9997998476028442, np.float32)
    product, x0 = checker.mixed_convert(sample, velocity, sigma, "torch.float32")
    np.testing.assert_array_equal(product.view(np.uint32), (velocity * sigma).view(np.uint32))
    np.testing.assert_array_equal(
        x0.view(np.uint32),
        np.asarray([[0.00020015239715576172, -0.00020015239715576172, 0.0, 0.0]], np.float32).view(np.uint32),
    )
    _, bf16_x0 = checker.mixed_convert(sample, velocity, sigma, "torch.bfloat16")
    assert bf16_x0[0, 0] == 0 and x0[0, 0] != 0
    with pytest.raises(ValueError, match="positive sigma"):
        checker.mixed_convert(sample, velocity, np.asarray(0, np.float32), "torch.float32")


def test_fixed_cpu_scalar_gold_matches_actual_cpu_primitives_and_differs_from_cuda_gold(checker):
    torch = pytest.importorskip("torch")
    different = 0
    for sigma_value in (0.8077949285507202, 0.5081728100776672, 0.2323688119649887):
        sigma = np.asarray(sigma_value, np.float32)
        for payload in ("signed_normal", "boundaries_subnormal"):
            sample, uncast = checker.precision_payload(payload)
            velocity = checker.OLD.quantize_bfloat16(uncast)
            product, x0 = checker.mixed_convert(sample, velocity, sigma, "torch.bfloat16", "cpu")
            actual_product = torch.tensor(sigma) * torch.tensor(uncast).to(torch.bfloat16)
            actual_x0 = torch.tensor(sample) - actual_product
            np.testing.assert_array_equal(product.view(np.uint32), actual_product.float().numpy().view(np.uint32))
            np.testing.assert_array_equal(x0.view(np.uint32), actual_x0.numpy().view(np.uint32))
            cuda_product, _ = checker.mixed_convert(sample, velocity, sigma, "torch.bfloat16", "cuda")
            different += product.tobytes() != cuda_product.tobytes()
    assert different > 0


def test_source111_sorted_snapshots_survive_json_sort_and_require_observer_entrypoint(checker, tmp_path):
    inherited = {
        str(checker.HELPER): checker.HELPER_SHA,
        **{str(tmp_path / f"f{i}.py"): f"{i:064x}" for i in range(95)},
    }
    sources = dict(inherited)
    sources.update({str(checker.REPO / p): "a" * 64 for p in checker.NEW_SOURCES})
    sources.update({str(checker.CONFIG): checker.CONFIG_SHA, str(checker.PLAN): checker.PLAN_SHA})
    sources[str(checker.ADDENDUM)] = checker.ADDENDUM_SHA
    sources[str(checker.SOURCE_ADDENDUM)] = checker.SOURCE_ADDENDUM_SHA
    path = tmp_path / "preparation/prepared.json"
    snapshots = {
        str(path.parent / "source_snapshot" / f"{i:03d}_{Path(p).name}"): sources[p]
        for i, p in enumerate(sorted(sources))
    }
    prepared = json.loads(json.dumps(dict(source_sha256=sources, snapshot_sha256=snapshots), sort_keys=True))
    checker.source_closure(prepared, dict(source_sha256=inherited), path)
    changed = copy.deepcopy(prepared)
    changed["source_sha256"].pop(str(checker.REPO / "src/s2a/wam_trajectory_capture.py"))
    with pytest.raises(ValueError, match="exact111 source"):
        checker.source_closure(changed, dict(source_sha256=inherited), path)
    changed = copy.deepcopy(prepared)
    first, second = list(changed["snapshot_sha256"])[:2]
    changed["snapshot_sha256"][first] = changed["snapshot_sha256"][second]
    with pytest.raises(ValueError, match="source snapshots"):
        checker.source_closure(changed, dict(source_sha256=inherited), path)


def test_real_program_entry_binds_scoring_functions_before_main_and_old_order_fails(checker, tmp_path):
    """Execute the real __main__ load order, stopping before argparse or data IO."""

    class MainEntryObserved(Exception):
        pass

    def run_to_main(path):
        def trace(frame, event, argument):
            if event == "call" and frame.f_code.co_filename == str(path) and frame.f_code.co_name == "main":
                assert all(callable(frame.f_globals.get(name)) for name in ("statistic", "point_diagnostics")), (
                    "scoring functions unbound at actual main entry"
                )
                raise MainEntryObserved
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            runpy.run_path(str(path), run_name="__main__")
        finally:
            sys.settrace(previous)

    real = Path(checker.__file__).resolve()
    with pytest.raises(MainEntryObserved):
        run_to_main(real)
    guard = 'if __name__ == "__main__":\n    main()\n'
    source = real.read_text()
    assert source.count(guard) == 1
    old_order = source.replace(guard, "").replace("def statistic(", guard + "\n\ndef statistic(", 1)
    old_path = tmp_path / "former_entry_order.py"
    old_path.write_text(old_order)
    with pytest.raises(AssertionError, match="scoring functions unbound"):
        run_to_main(old_path)


def test_source_revision_accepts_only_declared_v2_root_and_rejects_old_prepare_before_admission(checker, monkeypatch):
    config = json.loads(checker.CONFIG.read_text())
    prepared = dict(
        schema_version="s2a.free-trajectory-preparation.v1",
        config=config,
        config_ref=checker.reference(checker.CONFIG),
        registration=checker.reference(checker.PLAN),
        source_revision_ref=dict(path=str(checker.SOURCE_ADDENDUM), sha256=checker.SOURCE_ADDENDUM_SHA),
        output_root=str(checker.REPO / (config["output_dir"] + "_source_v2")),
    )

    class RegistrationAccepted(Exception):
        pass

    def stop_after_registration(*args):
        raise RegistrationAccepted

    evidence = SimpleNamespace(
        json=lambda path, *args: config if path == checker.CONFIG else prepared, check=lambda *args: None
    )
    monkeypatch.setattr(checker, "arithmetic_admission", stop_after_registration)
    with pytest.raises(RegistrationAccepted):
        checker.verify_preparation(Path("prepared.json"), evidence)
    for root in (config["output_dir"], config["output_dir"] + "_source_v3"):
        prepared["output_root"] = str(checker.REPO / root)
        with pytest.raises(ValueError, match="registration/config/output changed"):
            checker.verify_preparation(Path("prepared.json"), evidence)
    prepared["output_root"] = str(checker.REPO / (config["output_dir"] + "_source_v2"))
    prepared.pop("source_revision_ref")
    with pytest.raises(ValueError, match="registration/config/output changed"):
        checker.verify_preparation(Path("prepared.json"), evidence)


def test_preparation_never_hashes_ep_or_any_png_in_inherited_maps(checker):
    class NoRead:
        def check(self, *args):
            raise AssertionError("observation was opened before collection admission")

    mapping = {"/raw.npz": "a" * 64, "/new-camera.PNG": "b" * 64}
    assert checker.metadata_hashes(mapping, {"/raw.npz"}, NoRead()) == mapping


def collection_fixture(checker, tmp_path):
    preparation = tmp_path / "preparation/prepared.json"
    preparation.parent.mkdir()
    preparation.write_text("{}")
    prepared = dict(output_root=str(tmp_path), checkpoint="original8400")
    labels = [f"free:{i}" for i in range(48) for _ in range(35)] + [f"teacher:{i}" for i in range(144)]
    actual = dict(
        schema_version="s2a.free-trajectory-collection.v1",
        complete=True,
        passed=True,
        prepared_ref=checker.prepared_ref(prepared),
        all_original_parameters_frozen=True,
        original_parameters_require_grad_false=True,
        wam_model_initializations=1,
        denoise_calls=1824,
        free_generations=[{}] * 48,
        teacher_calls=[{}] * 144,
        fit_groups=0,
        train_episode_reads=0,
        physics_steps=0,
        fresh_cases=0,
        denoise_ledger=[dict(index=i, label=label, completed=True) for i, label in enumerate(labels)],
        head_call_counts=dict(free=1680, teacher=144),
        runtime=dict(checkpoint_path="original8400", ema_weights=False),
    )
    return prepared, actual


def test_collection_rejects_partial_repeated_reordered_or_new_fit_budget(checker, tmp_path):
    prepared, actual = collection_fixture(checker, tmp_path)
    checker.collection_contract(actual, prepared)
    for key, value in (("complete", False), ("denoise_calls", 1823), ("fit_groups", 1), ("train_episode_reads", 1)):
        with pytest.raises(ValueError, match="collection|budget"):
            checker.collection_contract(dict(actual, **{key: value}), prepared)
    changed = copy.deepcopy(actual)
    changed["denoise_ledger"][1680]["label"] = "free:47"
    with pytest.raises(ValueError, match="boundary ledger"):
        checker.collection_contract(changed, prepared)


def test_protocol_and_release_must_precede_source_label_loading(checker, monkeypatch):
    actual = dict(precision_ref="p", precision_verification_ref="v")
    evidence = SimpleNamespace(json=lambda path: actual)
    calls = []
    monkeypatch.setattr(checker, "precision_admission", lambda *args: ("p", "v"))
    monkeypatch.setattr(checker, "collection_contract", lambda *args: calls.append("complete"))
    monkeypatch.setattr(checker.STATE, "public_lifecycle", lambda *args: calls.append("release"))

    def rejected(*args):
        calls.append("protocol")
        raise ValueError("raw exact replay rejected")

    monkeypatch.setattr(checker, "collection_records", rejected)
    monkeypatch.setattr(checker, "verified_labels", lambda *args: pytest.fail("labels opened after rejected protocol"))
    with pytest.raises(ValueError, match="raw exact replay"):
        checker.verify_collection({}, {}, "p", "v", "collection", evidence)
    assert calls == ["complete", "release", "protocol"]


def test_free_point_uses_original_fp32_sample_actual_casts_and_unmasked_head_tail(checker):
    vision_shape = (1, 48, 11, 60, 40)
    offset = int(np.prod(vision_shape))
    sample = np.zeros(offset + 5120, np.float32)
    base = np.zeros((40, 128), np.float32)
    base[:, 10:] = 0.5  # The original head tail is allowed; final velocity is masked.
    arrays = dict(
        sample=sample,
        sample_action=sample[offset:].reshape(40, 128),
        packed_vision=sample[:offset].reshape(vision_shape),
        packed_action=np.zeros((40, 128), np.float32),
        hidden=np.zeros((40, 1280), np.float32),
        base=base,
        projection_ids=np.zeros(40, np.int64),
        pred_action=np.zeros((40, 128), np.float32),
        network_timestep=np.asarray([[807]], np.int64),
        time_vision=np.full(6000, 807, np.float32),
        time_action=np.full(40, 807, np.float32),
        condition_action=np.zeros((40, 1), np.float32),
        velocity=np.zeros_like(sample),
        velocity_action=np.zeros((40, 128), np.float32),
        converted_action=np.zeros((40, 128), np.float32),
    )
    dtypes = {
        key: "torch.bfloat16" for key in ("hidden", "base", "packed_action", "packed_vision", "pred_action", "velocity")
    }
    dtypes.update(sample="torch.float32", time_vision="torch.float32", time_action="torch.float32")
    metadata = dict(
        original_dtypes=dtypes,
        action_offset=offset,
        vision_shape=list(vision_shape),
        network_timestep=807,
        sigma_actual=float(np.float32(0.8077949285507202)),
        context=dict(
            domain=checker.OLD.DOMAIN, normalization_id=31, projection_id=0, action_dim_mask=checker.OLD.ACTIVE.tolist()
        ),
        conversion_sample_is_original_storage_view=True,
        conversion_returned_original_object=True,
        conversion_calls=1,
    )
    checker.tensor_contract(metadata, arrays)
    arrays["time_action"][0] = 808
    with pytest.raises(ValueError, match="packed network time"):
        checker.tensor_contract(metadata, arrays)
    arrays["time_action"][0] = 807
    arrays["velocity_action"][0, 127] = 0.125
    with pytest.raises(ValueError, match="returned velocity action view"):
        checker.tensor_contract(metadata, arrays)


def test_aggregation_gives_equal_episode_weight_instead_of_point_weight(checker):
    rows = []
    for episode in range(6):
        for index in range(4 if episode == 0 else 28):
            rows.append(
                dict(
                    episode=episode,
                    task=episode // 2,
                    sigma=0.5,
                    seed=42 + index % 2,
                    metrics=dict(value=float(episode)),
                )
            )
    # 4 + 5*28 =144: deliberately imbalanced synthetic rows exercise the unit.
    pooled = checker.aggregate_diagnostics(rows)["pooled"]
    assert pooled["metrics"]["value"] == 2.5
    assert pooled["metrics"]["value"] != np.mean([r["metrics"]["value"] for r in rows])


def test_free_pull_is_not_teacher_target_and_q_sigma_is_separate_from_clean_error(checker):
    sigma = 0.5
    base = np.zeros((40, 128), np.float32)
    sample = np.zeros_like(base)
    sample[:, :10] = np.float32(0.2501)
    x0 = np.zeros_like(base)
    x0[:, :10] = 0.5
    hidden = np.zeros((40, 1280), np.float32)
    free = dict(
        hidden=hidden,
        base=base,
        packed_action=checker.OLD.quantize_bfloat16(sample),
        packed_vision=np.zeros((1, 1, 2, 1, 1), np.float32),
        sample_action=sample,
        velocity_action=base,
        converted_action=sample.copy(),
    )
    target = np.zeros_like(base)
    target[:, :10] = 0.5
    teacher = dict(
        hidden=hidden,
        base=base,
        pred_action=base,
        target=target,
        packed_action=np.zeros_like(base),
        packed_vision=free["packed_vision"].copy(),
    )
    weights = np.zeros((1281, 10))
    weights[-1] = 0.05
    model = dict(weights=weights, center=np.zeros(1280), scale=np.asarray(1.0))
    metrics = checker.point_diagnostics(free, teacher, x0, model, sigma)
    assert metrics["clean.active10.offline027_actual_error.mse"] < metrics["clean.active10.raw_actual_error.mse"]
    assert (
        metrics["velocity.teacher.active10.offline027_error.mse"] > metrics["velocity.teacher.active10.raw_error.mse"]
    )
    assert metrics["quantization.active10.q_over_sigma.mse"] == 4 * metrics["quantization.active10.q.mse"]
    for arithmetic in ("ideal", "actual"):
        prefix = f"decomposition.active10.{arithmetic}"
        assert metrics[f"{prefix}.two_e_delta"] < 0
        assert metrics[f"{prefix}.delta_squared"] > 0
        assert metrics[f"{prefix}.mse_change"] == pytest.approx(
            metrics[f"{prefix}.two_e_delta"] + metrics[f"{prefix}.delta_squared"], abs=2e-12
        )


def test_phase_arguments_are_explicit_and_do_not_accept_extra_partial_evidence(checker):
    for phase, present in dict(
        prepare=(False, False, False), precision=(True, False, False), collection=(True, True, True)
    ).items():
        args = SimpleNamespace(
            phase=phase,
            **dict(
                zip(
                    ("precision", "precision_verification", "collection"),
                    [Path("x") if p else None for p in present],
                    strict=True,
                )
            ),
        )
        checker.phase_arguments(args)
        args.collection = Path("unrequested") if args.collection is None else None
        with pytest.raises(ValueError, match="explicit phase"):
            checker.phase_arguments(args)
