"""Simple GTP (Go Text Protocol) server implementation.

This module exposes a tiny TCP server speaking the Go Text Protocol (GTP).
It listens on ``localhost:6617`` and understands a handful of common
commands.  Each command is handled by a dedicated function that updates the
current session state and returns a proper GTP response.

The server assumes the existence of a prediction function
``core.engine.predict(input_dict)`` which is used by the ``genmove`` command
to generate the next move.  If it is not available, a dummy stub returning
``"pass"`` is used so that the module remains executable in lightweight
environments and during unit tests.
"""

from __future__ import annotations

import socket
from typing import Callable, Dict, List, Tuple, Optional

try:
    from core import engine as core_engine
    predict = core_engine.predict  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback for tests
    def predict(input_dict: Dict[str, object]) -> str:
        """Fallback prediction function returning ``"pass"``."""
        return "pass"


HOST = "localhost"
PORT = 6617


class GTPServer:
    """Minimal stateful GTP server."""

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        """Initialize the server listening on ``host`` and ``port``."""
        self.host = host
        self.port = port
        self.board_size = 19
        self.komi = 0.0
        self.moves: List[Tuple[str, str]] = []  # (color, coordinate)
        self._handlers: Dict[str, Callable[[List[str]], Tuple[str, bool]]] = {
            "protocol_version": self.handle_protocol_version,
            "name": self.handle_name,
            "version": self.handle_version,
            "list_commands": self.handle_list_commands,
            "boardsize": self.handle_boardsize,
            "clear_board": self.handle_clear_board,
            "komi": self.handle_komi,
            "play": self.handle_play,
            "genmove": self.handle_genmove,
            "quit": self.handle_quit,
        }

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def handle_protocol_version(self, args: List[str]) -> Tuple[str, bool]:
        """Return the supported GTP protocol version."""
        return "2", False

    def handle_name(self, args: List[str]) -> Tuple[str, bool]:
        """Return the engine name."""
        return "LightGo", False

    def handle_version(self, args: List[str]) -> Tuple[str, bool]:
        """Return the engine version."""
        return "0.1", False

    def handle_list_commands(self, args: List[str]) -> Tuple[str, bool]:
        """Return a newline separated list of supported commands."""
        cmds = sorted(self._handlers.keys())
        return "\n".join(cmds), False

    def handle_boardsize(self, args: List[str]) -> Tuple[str, bool]:
        """Set the board size."""
        if not args:
            return "missing size", False
        try:
            self.board_size = int(args[0])
        except ValueError:
            return "invalid size", False
        return "", False

    def handle_clear_board(self, args: List[str]) -> Tuple[str, bool]:
        """Clear all recorded moves."""
        self.moves.clear()
        return "", False

    def handle_komi(self, args: List[str]) -> Tuple[str, bool]:
        """Set the komi value."""
        if not args:
            return "missing komi", False
        try:
            self.komi = float(args[0])
        except ValueError:
            return "invalid komi", False
        return "", False

    def handle_play(self, args: List[str]) -> Tuple[str, bool]:
        """Record a move provided by the client."""
        if len(args) < 2:
            return "invalid play", False
        color = args[0].lower()
        move = args[1].upper()
        self.moves.append((color, move))
        return "", False

    def handle_genmove(self, args: List[str]) -> Tuple[str, bool]:
        """Generate a move using ``core.engine.predict``."""
        if not args:
            return "invalid color", False
        color = args[0].lower()
        move = predict({
            "board": list(self.moves),
            "color": color,
            "size": self.board_size,
            "komi": self.komi,
        })
        if not move:
            move = "pass"
        self.moves.append((color, str(move).upper()))
        return str(move).upper(), False

    def handle_quit(self, args: List[str]) -> Tuple[str, bool]:
        """Terminate the current session."""
        return "", True

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _dispatch(self, line: str) -> Tuple[str, bool]:
        """Parse ``line`` and dispatch to the appropriate handler."""
        tokens = line.strip().split()
        if not tokens:
            return "? syntax error", False

        ident: Optional[str] = None
        if tokens[0].isdigit():
            ident = tokens.pop(0)
        cmd = tokens.pop(0)
        handler = self._handlers.get(cmd)
        if handler is None:
            return self._format_response("?", ident, "unknown command"), False

        msg, should_quit = handler(tokens)
        response = self._format_response("=", ident, msg)
        return response, should_quit

    @staticmethod
    def _format_response(prefix: str, ident: Optional[str], msg: str) -> str:
        """Return a properly formatted GTP response line."""
        parts = [prefix]
        if ident:
            parts.append(ident)
        if msg:
            parts.append(msg)
        return " ".join(parts).rstrip()

    # ------------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------------
    def serve(self) -> None:
        """Start the TCP server and handle a single session."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen(1)
            conn, _ = sock.accept()
            with conn:
                rfile = conn.makefile("r", encoding="utf-8", newline="\n")
                wfile = conn.makefile("w", encoding="utf-8", newline="\n")
                while True:
                    line = rfile.readline()
                    if not line:
                        break
                    response, should_quit = self._dispatch(line)
                    wfile.write(response + "\n\n")
                    wfile.flush()
                    if should_quit:
                        break


def main() -> None:
    """Entry point for running the GTP server from the command line."""
    server = GTPServer()
    server.serve()


if __name__ == "__main__":  # pragma: no cover
    main()
