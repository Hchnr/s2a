"""The local transport must fail closed under real stream and process behavior."""

import os
import socket
import struct
import subprocess
import sys
import threading
import time

import pytest

from s2a.infrastructure.framed_channel import FramedJSONChannel


def test_round_trip_with_an_inherited_socket_in_a_child_process():
    parent, child = socket.socketpair()
    channel = FramedJSONChannel(parent, max_bytes=1024, timeout_seconds=10)
    code = """
import socket, sys
from s2a.infrastructure.framed_channel import FramedJSONChannel
channel = FramedJSONChannel(socket.socket(fileno=int(sys.argv[1])), max_bytes=1024, timeout_seconds=10)
request = channel.receive()
channel.send({'id': request['id'], 'image_sha': request['image_sha'], 'status': 'ok'})
channel.close()
"""
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(child.fileno())],
            pass_fds=(child.fileno(),),
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        )
        child.close()
        channel.send({"id": 3, "image_sha": "a" * 64, "task": "拿起物体"})
        assert channel.receive() == {"id": 3, "image_sha": "a" * 64, "status": "ok"}
        assert process.wait(timeout=10) == 0
    finally:
        channel.close()
        child.close()
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)


def test_fragmented_header_and_body_are_read_as_one_frame():
    receiver, sender = socket.socketpair()
    channel = FramedJSONChannel(receiver, max_bytes=1024, timeout_seconds=5)
    payload = b'{"id":1,"status":"ok"}'
    parts = [struct.pack("!I", len(payload))[:2], struct.pack("!I", len(payload))[2:], payload[:4], payload[4:]]
    sender.sendall(parts[0])
    thread = threading.Thread(target=lambda: [sender.sendall(part) for part in parts[1:]])
    try:
        thread.start()
        assert channel.receive() == {"id": 1, "status": "ok"}
    finally:
        thread.join(timeout=5)
        sender.close()
        channel.close()
    assert not thread.is_alive()


@pytest.mark.parametrize(
    ("payload", "exception"),
    [
        (b'{"id":1,"id":2}', ValueError),
        (b'{"inner":{"id":1,"id":2}}', ValueError),
        (b'{"value":NaN}', ValueError),
        (b'{"value":1e999}', ValueError),
        (b'{"value":"\xff"}', UnicodeDecodeError),
        (b"[]", ValueError),
    ],
)
def test_invalid_payload_poisoned_channel_cannot_be_reused(payload, exception):
    receiver, sender = socket.socketpair()
    channel = FramedJSONChannel(receiver, max_bytes=1024, timeout_seconds=5)
    try:
        sender.sendall(struct.pack("!I", len(payload)) + payload)
        with pytest.raises(exception):
            channel.receive()
        assert channel.failed
        with pytest.raises(RuntimeError):
            channel.send({"retry": True})
    finally:
        sender.close()
        channel.close()


@pytest.mark.parametrize("size", [0, 129])
def test_invalid_length_fails_before_waiting_for_the_body(size):
    receiver, sender = socket.socketpair()
    channel = FramedJSONChannel(receiver, max_bytes=128, timeout_seconds=5)
    try:
        sender.sendall(struct.pack("!I", size))
        with pytest.raises(ValueError):
            channel.receive()
        assert channel.failed
    finally:
        sender.close()
        channel.close()


def test_peer_closing_in_the_middle_of_a_frame_is_not_a_valid_message():
    receiver, sender = socket.socketpair()
    channel = FramedJSONChannel(receiver, max_bytes=128, timeout_seconds=5)
    try:
        sender.sendall(struct.pack("!I", 8) + b"{")
        sender.close()
        with pytest.raises(EOFError):
            channel.receive()
        assert channel.failed
    finally:
        sender.close()
        channel.close()


def test_one_deadline_applies_to_a_slowly_fragmented_header_and_body():
    receiver, sender = socket.socketpair()
    channel = FramedJSONChannel(receiver, max_bytes=128, timeout_seconds=0.18)
    sender.sendall(b"\x00\x00")

    def delayed_fragments():
        time.sleep(0.11)
        sender.sendall(b"\x00\x02")
        time.sleep(0.11)
        sender.sendall(b"{}")

    thread = threading.Thread(target=delayed_fragments)
    try:
        thread.start()
        with pytest.raises(TimeoutError):
            channel.receive()
        assert channel.failed
    finally:
        thread.join(timeout=5)
        sender.close()
        channel.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("message", [[], {1: "value"}, {"inner": {2: 3}}, {"x": float("inf")}, {"x": "a" * 200}])
def test_invalid_outbound_message_poisoned_channel(message):
    sender, receiver = socket.socketpair()
    channel = FramedJSONChannel(sender, max_bytes=128, timeout_seconds=5)
    try:
        with pytest.raises(ValueError):
            channel.send(message)
        assert channel.failed
        with pytest.raises(RuntimeError):
            channel.receive()
    finally:
        receiver.close()
        channel.close()


def test_unconnected_or_nonlocal_stream_is_rejected():
    local = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    network = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        for connection in (local, network):
            with pytest.raises(ValueError):
                FramedJSONChannel(connection, max_bytes=128, timeout_seconds=5)
    finally:
        local.close()
        network.close()
