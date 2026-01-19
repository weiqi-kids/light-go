"""Component tests for GTP Interface (api/gtp_interface.py).

This module tests the GTP (Go Text Protocol) server:
- GTPServer: Minimal GTP server implementation
- Command handlers: protocol_version, name, boardsize, komi, play, genmove, etc.
- Session state management
"""
from __future__ import annotations

import pytest

from api.gtp_interface import GTPServer


@pytest.fixture
def gtp_server():
    """Return a fresh GTPServer instance."""
    return GTPServer()


class TestGTPServerInstantiation:
    """Tests for GTPServer instantiation."""

    def test_create_server(self):
        """GTPServer can be created with defaults."""
        server = GTPServer()

        assert server is not None
        assert server.board_size == 19
        assert server.moves == []
        assert server.komi == 0.0

    def test_create_server_custom_port(self):
        """GTPServer with custom host and port."""
        server = GTPServer(host="localhost", port=7777)

        assert server.port == 7777


class TestGTPBasicCommands:
    """Tests for simple GTP commands that return fixed values."""

    def test_protocol_version(self, gtp_server):
        """protocol_version returns '2'."""
        response, should_quit = gtp_server.handle_protocol_version([])

        assert response == "2"
        assert should_quit is False

    def test_name(self, gtp_server):
        """name returns engine name."""
        response, should_quit = gtp_server.handle_name([])

        assert "LightGo" in response
        assert should_quit is False

    def test_version(self, gtp_server):
        """version returns version string."""
        response, should_quit = gtp_server.handle_version([])

        assert response is not None
        assert should_quit is False

    def test_list_commands(self, gtp_server):
        """list_commands returns supported commands."""
        response, should_quit = gtp_server.handle_list_commands([])

        assert "protocol_version" in response
        assert "name" in response
        assert "play" in response
        assert "genmove" in response
        assert should_quit is False

    def test_clear_board(self, gtp_server):
        """clear_board clears moves."""
        gtp_server.moves = [("black", "D4"), ("white", "Q16")]

        response, should_quit = gtp_server.handle_clear_board([])

        assert gtp_server.moves == []
        assert should_quit is False

    def test_quit(self, gtp_server):
        """quit signals termination."""
        response, should_quit = gtp_server.handle_quit([])

        assert should_quit is True


class TestGTPBoardsize:
    """Tests for boardsize command."""

    @pytest.mark.parametrize("size", [9, 13, 19])
    def test_boardsize_valid(self, gtp_server, size):
        """boardsize sets board size correctly."""
        response, should_quit = gtp_server.handle_boardsize([str(size)])

        assert gtp_server.board_size == size
        assert should_quit is False

    def test_boardsize_missing_arg(self, gtp_server):
        """boardsize without argument returns error."""
        response, _ = gtp_server.handle_boardsize([])

        assert "missing" in response.lower()

    def test_boardsize_invalid(self, gtp_server):
        """boardsize with invalid argument returns error."""
        response, _ = gtp_server.handle_boardsize(["abc"])

        assert "invalid" in response.lower()


class TestGTPKomi:
    """Tests for komi command."""

    @pytest.mark.parametrize("komi_str,expected", [("7.5", 7.5), ("6.5", 6.5), ("0", 0.0)])
    def test_komi_valid(self, gtp_server, komi_str, expected):
        """komi sets komi value correctly."""
        response, should_quit = gtp_server.handle_komi([komi_str])

        assert gtp_server.komi == expected
        assert should_quit is False

    def test_komi_missing_arg(self, gtp_server):
        """komi without argument returns error."""
        response, _ = gtp_server.handle_komi([])

        assert "missing" in response.lower()

    def test_komi_invalid(self, gtp_server):
        """komi with invalid argument returns error."""
        response, _ = gtp_server.handle_komi(["abc"])

        assert "invalid" in response.lower()


class TestGTPPlay:
    """Tests for play command."""

    @pytest.mark.parametrize("color,move", [("black", "D4"), ("white", "Q16"), ("black", "PASS")])
    def test_play_valid(self, gtp_server, color, move):
        """play records move correctly."""
        response, should_quit = gtp_server.handle_play([color, move])

        assert (color, move) in gtp_server.moves
        assert should_quit is False

    def test_play_multiple_moves(self, gtp_server):
        """play records multiple moves in sequence."""
        gtp_server.handle_play(["black", "D4"])
        gtp_server.handle_play(["white", "Q16"])
        gtp_server.handle_play(["black", "D16"])

        assert len(gtp_server.moves) == 3

    def test_play_missing_args(self, gtp_server):
        """play without required arguments returns error."""
        response, _ = gtp_server.handle_play(["black"])

        assert "invalid" in response.lower()


class TestGTPGenmove:
    """Tests for genmove command."""

    @pytest.mark.parametrize("color", ["black", "white"])
    def test_genmove_valid(self, gtp_server, color):
        """genmove generates move for given color."""
        response, should_quit = gtp_server.handle_genmove([color])

        assert response is not None
        assert should_quit is False
        assert len(gtp_server.moves) == 1

    def test_genmove_missing_color(self, gtp_server):
        """genmove without color returns error."""
        response, _ = gtp_server.handle_genmove([])

        assert "invalid" in response.lower()


class TestGTPDispatch:
    """Tests for command dispatch mechanism."""

    def test_dispatch_known_command(self, gtp_server):
        """Dispatch routes to correct handler."""
        response, _ = gtp_server._dispatch("name")

        assert "LightGo" in response

    def test_dispatch_unknown_command(self, gtp_server):
        """Dispatch returns error for unknown command."""
        response, _ = gtp_server._dispatch("unknown_cmd")

        assert "?" in response or "unknown" in response.lower()

    def test_dispatch_with_id(self, gtp_server):
        """Dispatch includes command ID in response."""
        response, _ = gtp_server._dispatch("123 name")

        assert "123" in response

    def test_dispatch_empty_line(self, gtp_server):
        """Dispatch returns error for empty input."""
        response, _ = gtp_server._dispatch("")

        assert "?" in response or "error" in response.lower()


class TestGTPResponseFormat:
    """Tests for GTP response formatting."""

    @pytest.mark.parametrize("prefix,ident,msg,expected_start", [
        ("=", None, "result", "="),
        ("?", None, "error msg", "?"),
        ("=", "42", "result", "="),
        ("=", None, "", "="),
    ])
    def test_format_response(self, prefix, ident, msg, expected_start):
        """Response formatting follows GTP spec."""
        response = GTPServer._format_response(prefix, ident, msg)

        assert response.startswith(expected_start)
        if ident:
            assert ident in response
        if msg:
            assert msg in response


class TestGTPIntegration:
    """Integration tests for GTP server workflows."""

    def test_full_game_sequence(self, gtp_server):
        """Simulate a short game sequence."""
        gtp_server.handle_boardsize(["9"])
        gtp_server.handle_komi(["7.5"])
        gtp_server.handle_clear_board([])
        gtp_server.handle_play(["black", "E5"])
        gtp_server.handle_play(["white", "E4"])
        gtp_server.handle_genmove(["black"])

        assert len(gtp_server.moves) == 3
        assert gtp_server.board_size == 9
        assert gtp_server.komi == 7.5

    def test_reset_between_games(self, gtp_server):
        """clear_board resets state for new game."""
        gtp_server.handle_play(["black", "D4"])
        gtp_server.handle_play(["white", "Q16"])

        gtp_server.handle_clear_board([])

        assert gtp_server.moves == []


class TestEdgeCases:
    """Edge case tests for GTP server."""

    @pytest.mark.parametrize("color", ["BLACK", "White", "WHITE", "black"])
    def test_case_insensitive_color(self, gtp_server, color):
        """Colors are case insensitive."""
        gtp_server.handle_play([color, "D4"])

        assert len(gtp_server.moves) == 1

    def test_uppercase_move_normalization(self, gtp_server):
        """Moves are normalized to uppercase."""
        gtp_server.handle_play(["black", "d4"])

        assert any("D4" in move[1] for move in gtp_server.moves)

    def test_many_moves(self, gtp_server):
        """Handle many moves in sequence."""
        for i in range(50):
            color = "black" if i % 2 == 0 else "white"
            gtp_server.handle_play([color, "A1"])

        assert len(gtp_server.moves) == 50

    def test_concurrent_instances(self):
        """Multiple server instances are independent."""
        server1 = GTPServer()
        server2 = GTPServer()

        server1.handle_boardsize(["9"])
        server2.handle_boardsize(["19"])

        assert server1.board_size == 9
        assert server2.board_size == 19
