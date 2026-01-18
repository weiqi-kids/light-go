"""Component tests for Self-Play Engine / GTP Interface (api/gtp_interface.py).

This module tests the GTP (Go Text Protocol) server:
- GTPServer: Minimal GTP server implementation
- Command handlers: protocol_version, name, boardsize, komi, play, genmove, etc.
- Session state management
"""
from __future__ import annotations

import pytest
from typing import List, Tuple

from api.gtp_interface import GTPServer


class TestGTPServerInstantiation:
    """Tests for GTPServer instantiation."""

    def test_create_server(self):
        """GTPServer can be created."""
        server = GTPServer()

        assert server is not None
        assert server.board_size == 19  # Default

    def test_create_server_custom_port(self):
        """GTPServer with custom port."""
        server = GTPServer(host="localhost", port=7777)

        assert server.port == 7777

    def test_initial_state(self):
        """Server starts with clean state."""
        server = GTPServer()

        assert server.moves == []
        assert server.komi == 0.0
        assert server.board_size == 19


class TestGTPProtocolVersion:
    """Tests for protocol_version command."""

    def test_protocol_version(self):
        """protocol_version returns '2'."""
        server = GTPServer()

        response, should_quit = server.handle_protocol_version([])

        assert response == "2"
        assert should_quit is False


class TestGTPName:
    """Tests for name command."""

    def test_name(self):
        """name returns engine name."""
        server = GTPServer()

        response, should_quit = server.handle_name([])

        assert "LightGo" in response
        assert should_quit is False


class TestGTPVersion:
    """Tests for version command."""

    def test_version(self):
        """version returns version string."""
        server = GTPServer()

        response, should_quit = server.handle_version([])

        assert response is not None
        assert should_quit is False


class TestGTPListCommands:
    """Tests for list_commands command."""

    def test_list_commands(self):
        """list_commands returns supported commands."""
        server = GTPServer()

        response, should_quit = server.handle_list_commands([])

        # Check some expected commands
        assert "protocol_version" in response
        assert "name" in response
        assert "play" in response
        assert "genmove" in response
        assert should_quit is False


class TestGTPBoardsize:
    """Tests for boardsize command."""

    def test_boardsize(self):
        """boardsize sets board size."""
        server = GTPServer()

        response, should_quit = server.handle_boardsize(["9"])

        assert server.board_size == 9
        assert should_quit is False

    def test_boardsize_19(self):
        """boardsize 19 (standard)."""
        server = GTPServer()

        server.handle_boardsize(["19"])

        assert server.board_size == 19

    def test_boardsize_missing_arg(self):
        """boardsize without argument."""
        server = GTPServer()

        response, _ = server.handle_boardsize([])

        assert "missing" in response.lower()

    def test_boardsize_invalid(self):
        """boardsize with invalid argument."""
        server = GTPServer()

        response, _ = server.handle_boardsize(["abc"])

        assert "invalid" in response.lower()


class TestGTPClearBoard:
    """Tests for clear_board command."""

    def test_clear_board(self):
        """clear_board clears moves."""
        server = GTPServer()
        server.moves = [("black", "D4"), ("white", "Q16")]

        response, should_quit = server.handle_clear_board([])

        assert server.moves == []
        assert should_quit is False


class TestGTPKomi:
    """Tests for komi command."""

    def test_komi(self):
        """komi sets komi value."""
        server = GTPServer()

        response, should_quit = server.handle_komi(["7.5"])

        assert server.komi == 7.5
        assert should_quit is False

    def test_komi_zero(self):
        """komi 0 (no komi)."""
        server = GTPServer()

        server.handle_komi(["0"])

        assert server.komi == 0.0

    def test_komi_missing_arg(self):
        """komi without argument."""
        server = GTPServer()

        response, _ = server.handle_komi([])

        assert "missing" in response.lower()

    def test_komi_invalid(self):
        """komi with invalid argument."""
        server = GTPServer()

        response, _ = server.handle_komi(["abc"])

        assert "invalid" in response.lower()


class TestGTPPlay:
    """Tests for play command."""

    def test_play_black(self):
        """play black move."""
        server = GTPServer()

        response, should_quit = server.handle_play(["black", "D4"])

        assert ("black", "D4") in server.moves
        assert should_quit is False

    def test_play_white(self):
        """play white move."""
        server = GTPServer()

        response, should_quit = server.handle_play(["white", "Q16"])

        assert ("white", "Q16") in server.moves

    def test_play_multiple_moves(self):
        """play multiple moves."""
        server = GTPServer()

        server.handle_play(["black", "D4"])
        server.handle_play(["white", "Q16"])
        server.handle_play(["black", "D16"])

        assert len(server.moves) == 3

    def test_play_missing_args(self):
        """play without required arguments."""
        server = GTPServer()

        response, _ = server.handle_play(["black"])

        assert "invalid" in response.lower()

    def test_play_pass(self):
        """play pass move."""
        server = GTPServer()

        response, _ = server.handle_play(["black", "PASS"])

        assert ("black", "PASS") in server.moves


class TestGTPGenmove:
    """Tests for genmove command."""

    def test_genmove_black(self):
        """genmove for black."""
        server = GTPServer()

        response, should_quit = server.handle_genmove(["black"])

        assert response is not None
        assert should_quit is False

    def test_genmove_white(self):
        """genmove for white."""
        server = GTPServer()

        response, should_quit = server.handle_genmove(["white"])

        assert response is not None

    def test_genmove_adds_to_moves(self):
        """genmove adds move to history."""
        server = GTPServer()
        initial_count = len(server.moves)

        server.handle_genmove(["black"])

        assert len(server.moves) == initial_count + 1

    def test_genmove_missing_color(self):
        """genmove without color."""
        server = GTPServer()

        response, _ = server.handle_genmove([])

        assert "invalid" in response.lower()


class TestGTPQuit:
    """Tests for quit command."""

    def test_quit(self):
        """quit signals termination."""
        server = GTPServer()

        response, should_quit = server.handle_quit([])

        assert should_quit is True


class TestGTPDispatch:
    """Tests for command dispatch."""

    def test_dispatch_known_command(self):
        """Dispatch known command."""
        server = GTPServer()

        response, _ = server._dispatch("name")

        assert "LightGo" in response

    def test_dispatch_unknown_command(self):
        """Dispatch unknown command."""
        server = GTPServer()

        response, _ = server._dispatch("unknown_cmd")

        assert "?" in response or "unknown" in response.lower()

    def test_dispatch_with_id(self):
        """Dispatch command with ID."""
        server = GTPServer()

        response, _ = server._dispatch("123 name")

        assert "123" in response

    def test_dispatch_empty_line(self):
        """Dispatch empty line."""
        server = GTPServer()

        response, _ = server._dispatch("")

        assert "?" in response or "error" in response.lower()


class TestGTPResponseFormat:
    """Tests for GTP response formatting."""

    def test_format_success_response(self):
        """Success response starts with '='."""
        response = GTPServer._format_response("=", None, "result")

        assert response.startswith("=")
        assert "result" in response

    def test_format_error_response(self):
        """Error response starts with '?'."""
        response = GTPServer._format_response("?", None, "error msg")

        assert response.startswith("?")

    def test_format_with_id(self):
        """Response includes ID when provided."""
        response = GTPServer._format_response("=", "42", "result")

        assert "42" in response

    def test_format_empty_message(self):
        """Response with empty message."""
        response = GTPServer._format_response("=", None, "")

        assert response.startswith("=")


class TestGTPIntegration:
    """Integration tests for GTP server."""

    def test_full_game_sequence(self):
        """Simulate a short game sequence."""
        server = GTPServer()

        # Setup
        server.handle_boardsize(["9"])
        server.handle_komi(["7.5"])
        server.handle_clear_board([])

        # Play some moves
        server.handle_play(["black", "E5"])
        server.handle_play(["white", "E4"])

        # Generate a move
        response, _ = server.handle_genmove(["black"])

        assert len(server.moves) == 3  # 2 plays + 1 genmove
        assert server.board_size == 9
        assert server.komi == 7.5

    def test_reset_between_games(self):
        """clear_board resets for new game."""
        server = GTPServer()

        # First game
        server.handle_play(["black", "D4"])
        server.handle_play(["white", "Q16"])

        # Clear and check
        server.handle_clear_board([])

        assert server.moves == []


class TestGTPCapabilities:
    """Tests for GTP capability checking."""

    def test_has_genmove(self):
        """Server has genmove capability."""
        server = GTPServer()

        assert hasattr(server, "handle_genmove")
        assert callable(server.handle_genmove)

    def test_has_play(self):
        """Server has play capability."""
        server = GTPServer()

        assert hasattr(server, "handle_play")
        assert callable(server.handle_play)

    def test_has_boardsize(self):
        """Server has boardsize capability."""
        server = GTPServer()

        assert hasattr(server, "handle_boardsize")
        assert callable(server.handle_boardsize)


class TestEdgeCases:
    """Edge case tests for GTP server."""

    def test_case_insensitive_color(self):
        """Colors are case insensitive."""
        server = GTPServer()

        server.handle_play(["BLACK", "D4"])
        server.handle_play(["White", "Q16"])

        assert len(server.moves) == 2

    def test_uppercase_move(self):
        """Moves are stored uppercase."""
        server = GTPServer()

        server.handle_play(["black", "d4"])

        # Check move was normalized
        assert any("D4" in move[1] for move in server.moves)

    def test_many_moves(self):
        """Handle many moves in sequence."""
        server = GTPServer()

        for i in range(50):
            color = "black" if i % 2 == 0 else "white"
            server.handle_play([color, "A1"])

        assert len(server.moves) == 50

    def test_concurrent_instances(self):
        """Multiple server instances are independent."""
        server1 = GTPServer()
        server2 = GTPServer()

        server1.handle_boardsize(["9"])
        server2.handle_boardsize(["19"])

        assert server1.board_size == 9
        assert server2.board_size == 19
