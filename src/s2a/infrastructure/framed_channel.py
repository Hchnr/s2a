"""Bounded, deadline-controlled JSON frames on one inherited local stream."""

import contextlib
import json
import math
import socket
import struct
import time


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("non-finite JSON number")


def _string_keys(value):
    if isinstance(value, dict):
        return all(isinstance(key, str) and _string_keys(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_string_keys(item) for item in value)
    return True


def _finite_numbers(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    return True


class FramedJSONChannel:
    """One sequential, fail-closed exchange on a preconnected Unix stream.

    The caller owns process admission and must not use this channel concurrently
    from multiple threads. Each send or receive has one total deadline, including
    a fragmented header or body. An I/O or validation failure poisons the channel.
    """

    def __init__(self, connection, *, max_bytes, timeout_seconds):
        if (
            not isinstance(connection, socket.socket)
            or connection.fileno() < 0
            or connection.family != socket.AF_UNIX
            or connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM
            or type(max_bytes) is not int
            or not 1 <= max_bytes <= 16 * 1024 * 1024
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("A local stream, bounded frame size and positive finite timeout are required")
        try:
            connection.getpeername()
        except OSError as error:
            raise ValueError("The local stream must already be connected") from error
        self.connection = connection
        self.max_bytes = max_bytes
        self.timeout_seconds = float(timeout_seconds)
        self.failed = False
        self.closed = False

    def _ready(self):
        if self.failed or self.closed:
            raise RuntimeError("The channel is failed or closed")

    def _timeout(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Frame deadline expired")
        self.connection.settimeout(remaining)

    def _read_exactly(self, size, deadline):
        result = bytearray()
        while len(result) < size:
            self._timeout(deadline)
            try:
                part = self.connection.recv(size - len(result))
            except socket.timeout as error:
                raise TimeoutError("Frame deadline expired") from error
            if not part:
                raise EOFError("Peer closed during a frame")
            result.extend(part)
        return bytes(result)

    def send(self, message):
        self._ready()
        original_timeout = self.connection.gettimeout()
        try:
            if not isinstance(message, dict) or not _string_keys(message):
                raise ValueError("A JSON object with string keys is required")
            body = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            if not 0 < len(body) <= self.max_bytes:
                raise ValueError("Frame size exceeds its registered bound")
            packet = memoryview(struct.pack("!I", len(body)) + body)
            deadline = time.monotonic() + self.timeout_seconds
            while packet:
                self._timeout(deadline)
                try:
                    count = self.connection.send(packet)
                except socket.timeout as error:
                    raise TimeoutError("Frame deadline expired") from error
                if count <= 0:
                    raise EOFError("Peer closed during a frame")
                packet = packet[count:]
        except BaseException:
            self.failed = True
            raise
        finally:
            with contextlib.suppress(OSError):
                self.connection.settimeout(original_timeout)

    def receive(self):
        self._ready()
        original_timeout = self.connection.gettimeout()
        try:
            deadline = time.monotonic() + self.timeout_seconds
            size = struct.unpack("!I", self._read_exactly(4, deadline))[0]
            if not 0 < size <= self.max_bytes:
                raise ValueError("Frame size exceeds its registered bound")
            data = self._read_exactly(size, deadline).decode("utf-8")
            message = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            if not isinstance(message, dict) or not _finite_numbers(message):
                raise ValueError("A JSON object containing only finite numbers is required")
            return message
        except BaseException:
            self.failed = True
            raise
        finally:
            with contextlib.suppress(OSError):
                self.connection.settimeout(original_timeout)

    def close(self):
        if not self.closed:
            self.closed = True
            self.connection.close()
