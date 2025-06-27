import pathlib
import socket
import sys
import threading
import time

import pytest

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api import gtp_interface


@pytest.fixture()
def gtp_client(monkeypatch):
    """Start the GTP server and return a helper to send commands."""
    monkeypatch.setattr(gtp_interface, "predict", lambda data: "D4", raising=False)
    thread = threading.Thread(target=gtp_interface.main, daemon=True)
    thread.start()
    # Wait for the server socket to be ready
    time.sleep(0.1)
    sock = socket.create_connection((gtp_interface.HOST, gtp_interface.PORT))
    file = sock.makefile("rw", encoding="utf-8", newline="\n")

    def send(cmd: str) -> str:
        file.write(cmd + "\n")
        file.flush()
        lines = []
        while True:
            line = file.readline()
            assert line != ""  # connection should remain open
            if line == "\n":
                break
            lines.append(line.strip())
        return "\n".join(lines)

    yield send

    try:
        try:
            file.write("quit\n")
            file.flush()
        except Exception:
            pass
    finally:
        file.close()
        sock.close()
        thread.join(timeout=2)


@pytest.fixture()
def gtp_client_real():
    """Start the GTP server without patching predict."""
    thread = threading.Thread(target=gtp_interface.main, daemon=True)
    thread.start()
    time.sleep(0.1)
    sock = socket.create_connection((gtp_interface.HOST, gtp_interface.PORT))
    file = sock.makefile("rw", encoding="utf-8", newline="\n")

    def send(cmd: str) -> str:
        file.write(cmd + "\n")
        file.flush()
        lines = []
        while True:
            line = file.readline()
            assert line != ""  # connection should remain open
            if line == "\n":
                break
            lines.append(line.strip())
        return "\n".join(lines)

    yield send

    try:
        try:
            file.write("quit\n")
            file.flush()
        except Exception:
            pass
    finally:
        file.close()
        sock.close()
        thread.join(timeout=2)


def test_gtp_interface_commands(gtp_client):
    assert gtp_client("boardsize 19") == "="
    assert gtp_client("komi 6.5") == "="
    assert gtp_client("clear_board") == "="
    assert gtp_client("play black D4") == "="

    assert gtp_client("name") == "= LightGo"
    assert gtp_client("version") == "= 0.1"
    assert gtp_client("protocol_version") == "= 2"

    resp = gtp_client("list_commands").split("\n")
    assert resp[0].startswith("=")
    cmds = [resp[0][2:]] + resp[1:]
    expected = sorted([
        "protocol_version",
        "name",
        "version",
        "list_commands",
        "boardsize",
        "clear_board",
        "komi",
        "play",
        "genmove",
        "quit",
    ])
    assert sorted(cmds) == expected

    assert gtp_client("genmove black") == "= D4"
    assert gtp_client("unknown_cmd") == "? unknown command"
    assert gtp_client("quit") == "="


def test_gtp_interface_default_move(gtp_client_real):
    assert gtp_client_real("clear_board") == "="
    resp = gtp_client_real("genmove black")
    assert resp.startswith("=")
    assert resp.strip() not in {"=", "= PASS"}
