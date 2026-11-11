"""Probe the existing runtime audit hook with inherited socket communication."""

import hashlib
import json
import sys

from s2a.infrastructure.lifecycle.owned_socket_pair import run_owned_socket_pair
from s2a.infrastructure.repository_paths import repository_root


def test_actual_runtime_guard_allows_inherited_socket_and_current_image_but_blocks_other_writes(tmp_path):
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    image = b"guarded current image"
    (guarded / "current.png").write_bytes(image)
    root = repository_root()
    config = {
        "session": "audit-hook-cpu-probe",
        "initial": {"step": 80, "sim_time_s": 4.0, "state_sha256": "a" * 64},
        "image_root": str(guarded.relative_to(root)),
        "task": "move the object",
        "seed": 42,
        "model_sha256": "b" * 64,
        "max_requests": 1,
    }
    image_ref = {"path": "current.png", "sha256": hashlib.sha256(image).hexdigest()}
    simulation = """
import ast, inspect, json, os, socket, sys
from pathlib import Path
from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.research.native_runtime import development_prefix_runtime as runtime
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger, model_inputs
config, image = json.loads(sys.argv[1]), json.loads(sys.argv[2])
source = inspect.getsource(runtime.run_prefix_runtime)
function = ast.parse(source).body[0]
guard = next(node for node in function.body if isinstance(node, ast.FunctionDef) and node.name == 'guard')
module = ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[]))
namespace = {'active': [True], 'output': Path(sys.argv[3]), 'bootstrap': runtime.bootstrap, 'os': os, 'Path': Path}
exec(compile(module, runtime.__file__, 'exec'), namespace)
sys.addaudithook(namespace['guard'])
ledger = CausalStreamLedger(**config)
current = ledger.current
request = ledger.issue(before_render=current, after_render=current, image=image)
assert model_inputs(request, image_root=config['image_root'])['image_bytes'] == b'guarded current image'
channel = FramedJSONChannel(socket.socket(fileno=int(os.environ['S2A_CHANNEL_FD'])), max_bytes=1024, timeout_seconds=10)
channel.send({'image_sha256': request['image']['sha256']})
assert channel.receive() == {'ok': True}
channel.close()
try:
    Path(sys.argv[4]).write_text('must not be written')
except ValueError:
    pass
else:
    raise AssertionError('runtime guard admitted a write outside its output')
"""
    model = """
import os, socket
from s2a.infrastructure.framed_channel import FramedJSONChannel
channel = FramedJSONChannel(socket.socket(fileno=int(os.environ['S2A_CHANNEL_FD'])), max_bytes=1024, timeout_seconds=10)
assert len(channel.receive()['image_sha256']) == 64
channel.send({'ok': True})
channel.close()
"""
    env = {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
    specs = {
        "simulation": {
            "argv": [
                sys.executable,
                "-c",
                simulation,
                json.dumps(config),
                json.dumps(image_ref),
                str(guarded),
                str(tmp_path / "outside.txt"),
            ],
            "env": dict(env),
            "log": str((tmp_path / "simulation.log").relative_to(root)),
        },
        "model": {
            "argv": [sys.executable, "-c", model],
            "env": dict(env),
            "log": str((tmp_path / "model.log").relative_to(root)),
        },
    }
    result = run_owned_socket_pair(specs, timeout_seconds=15)
    logs = [(tmp_path / f"{name}.log").read_text() for name in ("simulation", "model")]
    assert result["passed"], (result, logs)
    assert not (tmp_path / "outside.txt").exists()
