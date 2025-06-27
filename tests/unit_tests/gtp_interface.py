import socket
import threading
import pathlib
import sys
from unittest.mock import patch

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import api.gtp_interface
from api.gtp_interface import GTPServer


def _start_server():
    server = GTPServer()
    srv_sock, cli_sock = socket.socketpair()
    thread = threading.Thread(target=server.handle_connection, args=(srv_sock,))
    thread.start()
    return server, cli_sock, thread


def _send(sock: socket.socket, cmd: str) -> str:
    sock.sendall((cmd + "\n").encode())
    data = b""
    while not data.endswith(b"\n\n"):
        data += sock.recv(1024)
    return data.decode().strip()


def test_basic_info_commands():
    server, sock, thread = _start_server()
    try:
        assert _send(sock, "name") == "= light-go"
        assert _send(sock, "version") == "= 1.0"
        assert _send(sock, "protocol_version") == "= 2"
    finally:
        _send(sock, "quit")
        thread.join()


def test_board_commands():
    server, sock, thread = _start_server()
    try:
        assert _send(sock, "boardsize 9") == "="
        assert server.board_size == 9
        assert _send(sock, "komi 6.5") == "="
        assert server.komi == 6.5
        assert _send(sock, "play black D4") == "="
        assert server.moves == [("black", "D4")]
        assert _send(sock, "clear_board") == "="
        assert server.moves == []
    finally:
        _send(sock, "quit")
        thread.join()


def test_genmove_and_list():
    server, sock, thread = _start_server()
    try:
        with patch.object(
            api.gtp_interface.core_engine,
            "predict",
            return_value="Q16",
            create=True,
        ) as predict:
            assert _send(sock, "genmove white") == "= Q16"
            predict.assert_called_once()
        resp = _send(sock, "list_commands")
        assert resp.startswith("=")
        for cmd in server.handlers.keys():
            assert cmd in resp
    finally:
        _send(sock, "quit")
        thread.join()


def test_unknown_and_quit():
    server, sock, thread = _start_server()
    try:
        assert _send(sock, "unknown") == "? unknown command"
    finally:
        _send(sock, "quit")
        thread.join()
