"""Unit tests for GTP Interface (api/gtp_interface.py).

This module provides lightweight tests for GTP server using real implementation.
For comprehensive GTPServer tests, see tests/components/test_gtp_interface.py.
"""
from __future__ import annotations

import pytest

from api.gtp_interface import GTPServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gtp_server() -> GTPServer:
    """Return a fresh GTPServer instance with default settings."""
    return GTPServer()


# ---------------------------------------------------------------------------
# Basic Command Tests (using real implementation)
# ---------------------------------------------------------------------------

class TestGTPBasicCommands:
    """Tests for basic GTP commands using real implementation."""

    def test_protocol_version(self, gtp_server: GTPServer):
        """protocol_version returns GTP version 2."""
        response, should_quit = gtp_server.handle_protocol_version([])

        assert response == "2"
        assert should_quit is False

    def test_name(self, gtp_server: GTPServer):
        """name returns LightGo."""
        response, should_quit = gtp_server.handle_name([])

        assert response == "LightGo"
        assert should_quit is False

    def test_version(self, gtp_server: GTPServer):
        """version returns 0.1."""
        response, should_quit = gtp_server.handle_version([])

        assert response == "0.1"
        assert should_quit is False

    def test_list_commands(self, gtp_server: GTPServer):
        """list_commands returns all supported commands."""
        response, should_quit = gtp_server.handle_list_commands([])

        expected_commands = [
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
        ]
        for cmd in expected_commands:
            assert cmd in response
        assert should_quit is False


class TestGTPBoardConfiguration:
    """Tests for board configuration commands."""

    @pytest.mark.parametrize("size", [9, 13, 19], ids=["9x9", "13x13", "19x19"])
    def test_boardsize(self, gtp_server: GTPServer, size: int):
        """boardsize sets board size correctly."""
        response, should_quit = gtp_server.handle_boardsize([str(size)])

        assert gtp_server.board_size == size
        assert should_quit is False

    @pytest.mark.parametrize("komi", [0.0, 6.5, 7.5], ids=["0.0", "6.5", "7.5"])
    def test_komi(self, gtp_server: GTPServer, komi: float):
        """komi sets komi value correctly."""
        response, should_quit = gtp_server.handle_komi([str(komi)])

        assert gtp_server.komi == komi
        assert should_quit is False

    def test_clear_board(self, gtp_server: GTPServer):
        """clear_board resets move history."""
        gtp_server.handle_play(["black", "D4"])
        gtp_server.handle_play(["white", "Q16"])

        response, should_quit = gtp_server.handle_clear_board([])

        assert gtp_server.moves == []
        assert should_quit is False


class TestGTPGameplay:
    """Tests for gameplay commands using real implementation."""

    @pytest.mark.parametrize("color,move", [
        ("black", "D4"),
        ("white", "Q16"),
        ("black", "PASS"),
    ], ids=["black_d4", "white_q16", "black_pass"])
    def test_play(self, gtp_server: GTPServer, color: str, move: str):
        """play records moves correctly."""
        response, should_quit = gtp_server.handle_play([color, move])

        assert (color, move) in gtp_server.moves
        assert should_quit is False

    @pytest.mark.parametrize("color", ["black", "white"], ids=["black", "white"])
    def test_genmove(self, gtp_server: GTPServer, color: str):
        """genmove generates a valid move using real engine."""
        gtp_server.handle_boardsize(["9"])
        gtp_server.handle_clear_board([])

        response, should_quit = gtp_server.handle_genmove([color])

        assert response is not None
        assert should_quit is False
        # Move should be recorded
        assert len(gtp_server.moves) == 1


class TestGTPDispatch:
    """Tests for command dispatch mechanism."""

    def test_dispatch_known_command(self, gtp_server: GTPServer):
        """Dispatch routes to correct handler."""
        response, _ = gtp_server._dispatch("name")

        assert "LightGo" in response

    def test_dispatch_unknown_command(self, gtp_server: GTPServer):
        """Dispatch returns error for unknown command."""
        response, _ = gtp_server._dispatch("unknown_cmd")

        assert "unknown" in response.lower()

    def test_dispatch_quit(self, gtp_server: GTPServer):
        """Dispatch quit signals termination."""
        response, should_quit = gtp_server._dispatch("quit")

        assert should_quit is True


class TestGTPIntegration:
    """Integration tests simulating real game sequences."""

    def test_full_game_sequence(self, gtp_server: GTPServer):
        """Simulate a complete game setup and play sequence."""
        # Setup
        gtp_server.handle_boardsize(["19"])
        gtp_server.handle_komi(["6.5"])
        gtp_server.handle_clear_board([])

        # Play opening moves
        gtp_server.handle_play(["black", "D4"])
        gtp_server.handle_play(["white", "Q16"])
        gtp_server.handle_play(["black", "D16"])
        gtp_server.handle_play(["white", "Q4"])

        assert gtp_server.board_size == 19
        assert gtp_server.komi == 6.5
        assert len(gtp_server.moves) == 4

    def test_genmove_after_play(self, gtp_server: GTPServer):
        """Generate move after some manual plays."""
        gtp_server.handle_boardsize(["9"])
        gtp_server.handle_clear_board([])
        gtp_server.handle_play(["black", "E5"])
        gtp_server.handle_play(["white", "E4"])

        response, _ = gtp_server.handle_genmove(["black"])

        assert response is not None
        assert len(gtp_server.moves) == 3
