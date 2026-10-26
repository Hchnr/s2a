"""122 branchwise temporal encoding of fully qualified116 training equations."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once
from scipy.spatial.transform import Rotation

from s2a.artifacts import sha256
from s2a.temporal_rotation_residual import decode, encode


def checked_encode(packet, twist):
    encoded = encode(packet, twist, 0.05)
    for i in range(len(packet)):
        q = Rotation.from_rotvec(0.5 * (0.05 * twist[i, 3:] + packet[i, 9:])).as_matrix()
        matrix = Rotation.from_rotvec(packet[i, 3:6]).as_matrix() @ q.T
        np.testing.assert_allclose(Rotation.from_rotvec(encoded[i, 3:6]).as_matrix(), matrix, rtol=0, atol=1e-12)
    restored = decode(encoded, twist, 0.05)
    np.testing.assert_allclose(
        Rotation.from_rotvec(restored[:, 3:6]).as_matrix(),
        Rotation.from_rotvec(packet[:, 3:6]).as_matrix(),
        rtol=0,
        atol=1e-12,
    )
    return encoded


def main():
    e = Evidence()
    for p in (
        __file__,
        "src/s2a/temporal_rotation_residual.py",
        "plans/v0.0.1/122_temporal_response_equations_20261025.md",
    ):
        e.read(p)
    root = Path("outputs/condition_span_forward_20261025")
    closed = e.json(root / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    p = root / "fit_result.json"
    e.check(p, closed["source_sha256"][str(p.resolve())])
    fit = e.json(p)
    ref = fit["cache_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        cache = {k: a[k].copy() for k in a.files}
    assert cache["target"].shape == (5063, 12)
    current = []
    for entry in fit["fit_sources"]:
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            current.append(a["x_site_twist"].copy())
    current = np.vstack(current)
    assert current.shape == (4865, 6)
    np.testing.assert_allclose(current, cache["raw"][:, 37:43], rtol=0, atol=1e-12)
    target, base = cache["target"].copy(), cache["nominal"].copy()
    target[:4865] = checked_encode(target[:4865], current)
    base[:4865] = checked_encode(base[:4865], current)
    assert len(fit["contrast_sources"]) == 198
    for j, record in enumerate(fit["contrast_sources"]):
        packets, twists = [], []
        for key in ("minus_ref", "plus_ref"):
            ref = record[key]
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as a:
                pose = a["controller_site_pose"][-2:]
                twists.append(a["source_twist"][-2].copy())
                packets.append(
                    np.r_[
                        pose[1, :3, 3] - pose[0, :3, 3],
                        Rotation.from_matrix(pose[1, :3, :3] @ pose[0, :3, :3].T).as_rotvec(),
                        0.05 * a["source_twist"][-1],
                    ]
                )
        np.testing.assert_allclose(twists[0], twists[1], rtol=0, atol=1e-9)
        delta = record["actual_normalized_delta"]
        np.testing.assert_allclose((packets[1] - packets[0]) / delta, cache["target"][4865 + j], rtol=0, atol=1e-9)
        encoded = checked_encode(np.asarray(packets), np.asarray(twists))
        target[4865 + j] = (encoded[1] - encoded[0]) / delta
    np.testing.assert_allclose(base[4865:, 3:6], 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(base[4865:, 9:12], 0, rtol=0, atol=1e-12)
    e.recheck()
    output = Path("outputs/temporal_response_equations_20261025")
    output.mkdir(exist_ok=False)
    path = output / "training.npz"
    with path.open("xb") as stream:
        np.savez(
            stream,
            features=cache["features"],
            target=target,
            nominal=base,
            weights=cache["weights"],
            groups=cache["groups"],
            row_kind=cache["row_kind"],
            current_twist=current,
        )
    write_once(
        output / "result.json",
        dict(
            complete=True,
            rows=5063,
            natural_rows=4865,
            contrast_equations=198,
            branchwise_encoding_verified=True,
            training_ref=dict(path=str(path.resolve()), sha256=sha256(path)),
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    print("122 all5063 equations encoded;198 branchwise contrasts;matrix parity and roundtrip passed;fit0")


if __name__ == "__main__":
    main()
