"""Simple GTP (Go Text Protocol) TCP service.

This module implements a lightweight GTP server that listens on
``localhost:6617`` and supports a handful of common commands. Each
incoming connection is treated as one GTP session allowing multiple
commands until ``quit`` is received. Responses follow the standard
GTP text format with ``="" for success and ``?`` for errors.

The server relies on a core prediction function
``core.engine.predict(input_dict)`` which should return the next move
for ``genmove``. The actual implementation of this function is
provided elsewhere in the project and can be monkey patched in unit
tests.
"""
from __future__ import annotations

import socket
from typing import Callable, Dict, List, Tuple

from core import engine as core_engine


class GTPServer:
    """Minimal GTP protocol server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6617) -> None:
        self.host = host
        self.port = port
        self.board_size = 19
        self.komi = 0.0
        self.moves: List[Tuple[str, str]] = []
        self.handlers: Dict[str, Callable[[List[str]], Tuple[bool, str]]] = {
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
    # Public API
    # ------------------------------------------------------------------
    def serve_forever(self) -> None:
        """Start the TCP server and handle sessions forever."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.host, self.port))
            sock.listen(1)
            while True:  # pragma: no cover - manual break via quit
                conn, _ = sock.accept()
                with conn:
                    self.handle_connection(conn)

    def handle_connection(self, conn: socket.socket) -> None:
        """Handle a single GTP session on the provided socket."""
        file = conn.makefile("rwb")
        try:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                tokens = line.split()
                cmd_id = None
                if tokens and tokens[0].isdigit():
                    cmd_id = tokens.pop(0)
                command = tokens[0]
                args = tokens[1:]

                handler = self.handlers.get(command, self.handle_unknown)
                success, message = handler(args)

                id_prefix = f" {cmd_id}" if cmd_id is not None else ""
                prefix = "=" if success else "?"
                response = f"{prefix}{id_prefix} {message}\n\n"
                file.write(response.encode("utf-8"))
                file.flush()

                if command == "quit":
                    break
        finally:
            file.close()

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def handle_protocol_version(self, _: List[str]) -> Tuple[bool, str]:
        """Return GTP protocol version."""
        return True, "2"

    def handle_name(self, _: List[str]) -> Tuple[bool, str]:
        """Return engine name."""
        return True, "light-go"

    def handle_version(self, _: List[str]) -> Tuple[bool, str]:
        """Return engine version."""
        return True, "1.0"

    def handle_list_commands(self, _: List[str]) -> Tuple[bool, str]:
        """List all supported command names separated by newlines."""
        cmds = "\n".join(self.handlers.keys())
        return True, cmds

    def handle_boardsize(self, args: List[str]) -> Tuple[bool, str]:
        """Set the board size."""
        if args:
            self.board_size = int(args[0])
        return True, ""

    def handle_clear_board(self, _: List[str]) -> Tuple[bool, str]:
        """Clear current board state."""
        self.moves.clear()
        return True, ""

    def handle_komi(self, args: List[str]) -> Tuple[bool, str]:
        """Set komi value."""
        if args:
            self.komi = float(args[0])
        return True, ""

    def handle_play(self, args: List[str]) -> Tuple[bool, str]:
        """Record a played move."""
        if len(args) < 2:
            return False, "invalid arguments"
        color = args[0].lower()
        move = args[1].upper()
        self.moves.append((color, move))
        return True, ""

    def handle_genmove(self, args: List[str]) -> Tuple[bool, str]:
        """Generate the next move using the core engine."""
        color = args[0].lower() if args else "black"
        move = core_engine.predict({"board": self.moves, "color": color})
        if move is None:
            move = "resign"
        return True, str(move)

    def handle_quit(self, _: List[str]) -> Tuple[bool, str]:
        """Terminate the current session."""
        return True, ""

    def handle_unknown(self, _: List[str]) -> Tuple[bool, str]:
        """Handle unsupported commands."""
        return False, "unknown command"


def main() -> None:
    """Run the GTP interface server."""
    server = GTPServer()
    server.serve_forever()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
