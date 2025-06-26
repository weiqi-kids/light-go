import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from input import sgf_to_input

REPORT_FILE = pathlib.Path(__file__).resolve().parents[2] / ".github" / "test_report.md"


def _append_report(test_name: str, context: str, expected, actual) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().isoformat()
    with REPORT_FILE.open("a", encoding="utf-8") as f:
        f.write(f"### {test_name}\n")
        f.write(f"{context}\n\n")
        f.write("**預期值**:\n```")
        f.write(f"{expected}\n")
        f.write("```\n")
        f.write("**實際值**:\n```")
        f.write(f"{actual}\n")
        f.write("```\n")
        f.write(f"**時間**: {ts}\n\n---\n")


def _assert_eq(test_name: str, context: str, expected, actual) -> None:
    if expected != actual:
        print(f"{test_name} failed: {context}\nexpected: {expected}\nactual: {actual}")
        _append_report(test_name, context, expected, actual)
        raise AssertionError(f"{context}: expected {expected}, got {actual}")


SGF_CONTENT = "(;FF[4]SZ[5];B[aa];W[bb];B[cc];W[dd])"
ALL_STEPS = [
    ("black", (0, 0)),
    ("white", (1, 1)),
    ("black", (2, 2)),
    ("white", (3, 3)),
]


def _create_sgf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "game.sgf"
    path.write_text(SGF_CONTENT)
    return path


def test_liberty(tmp_path: pathlib.Path) -> None:
    sgf_file = _create_sgf(tmp_path)
    expected = {
        1: {(0, 0, 2)},
        2: {(0, 0, 2), (1, 1, -4)},
        3: {(0, 0, 2), (1, 1, -4), (2, 2, 4)},
        4: {(0, 0, 2), (1, 1, -4), (2, 2, 4), (3, 3, -4)},
    }
    for step in range(1, 5):
        result = sgf_to_input.convert(str(sgf_file), step=step)
        actual = {tuple(item) for item in result["liberty"]}
        _assert_eq("test_liberty", f"SGF step {step} liberty", expected[step], actual)


def test_forbidden(tmp_path: pathlib.Path) -> None:
    sgf_file = _create_sgf(tmp_path)
    expected = {1: [], 2: [], 3: [], 4: []}
    for step in range(1, 5):
        result = sgf_to_input.convert(str(sgf_file), step=step)
        _assert_eq("test_forbidden", f"SGF step {step} forbidden", expected[step], result["forbidden"])


def test_metadata(tmp_path: pathlib.Path) -> None:
    sgf_file = _create_sgf(tmp_path)
    for step in range(1, 5):
        result = sgf_to_input.convert(str(sgf_file), step=step)
        md = result["metadata"]
        expected_steps = ALL_STEPS[:step]
        expected_next = "white" if step % 2 == 1 else "black"
        _assert_eq("test_metadata", f"SGF step {step} board_size", 5, md["rules"]["board_size"])
        _assert_eq("test_metadata", f"SGF step {step} komi", 7.5, md["rules"]["komi"])
        _assert_eq("test_metadata", f"SGF step {step} handicap", 0, md["rules"]["handicap"])
        _assert_eq("test_metadata", f"SGF step {step} capture_black", 0, md["capture"]["black"])
        _assert_eq("test_metadata", f"SGF step {step} capture_white", 0, md["capture"]["white"])
        _assert_eq("test_metadata", f"SGF step {step} next_move", expected_next, md["next_move"])
        _assert_eq("test_metadata", f"SGF step {step} steps", expected_steps, md["step"])
        _assert_eq(
            "test_metadata",
            f"SGF step {step} time_main",
            0.0,
            md["time_control"]["main_time_seconds"],
        )
        _assert_eq(
            "test_metadata",
            f"SGF step {step} byo_yomi_periods",
            0,
            md["time_control"]["byo_yomi"]["periods"],
        )
        _assert_eq(
            "test_metadata",
            f"SGF step {step} time_black",
            0.0,
            md["time"][0]["main_time_seconds"],
        )
        _assert_eq(
            "test_metadata",
            f"SGF step {step} time_white",
            0.0,
            md["time"][1]["main_time_seconds"],
        )

