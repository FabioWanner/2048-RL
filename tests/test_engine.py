from random import Random
from unittest.mock import MagicMock

import pytest

from src.engine.engine import Direction, Engine2048, UnsupportedDirection
from src.engine.typings import State


@pytest.fixture
def initial_state() -> State:
    return [[0, 8, 2, 2], [4, 2, 0, 2], [0, 0, 0, 0], [0, 0, 0, 2]]


def test_generate_state():
    rng = Random(0)
    result_state = Engine2048.generate_state(3, [0, 3, 7], rng.choice)
    assert result_state == [[3, 3, 0], [3, 7, 3], [3, 3, 3]]


def test_generate_state_does_never_return_all_zero_tiles():
    choice_mock = MagicMock()
    choice_mock.side_effect = [0, 0, 0, 0, 0, 0, 0, 1]
    result_state = Engine2048.generate_state(2, [], choice_mock)

    assert result_state == [[0, 0], [0, 1]]


@pytest.mark.parametrize(
    "direction, expected_state",
    [
        (
            Direction.LEFT,
            [[8, 4, 0, 0], [4, 4, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]],
        ),
        (
            Direction.RIGHT,
            [[0, 0, 8, 4], [0, 0, 4, 4], [0, 0, 0, 0], [0, 0, 0, 2]],
        ),
        (
            Direction.UP,
            [[4, 8, 2, 4], [0, 2, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
        ),
        (
            Direction.DOWN,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 8, 0, 2], [4, 2, 2, 4]],
        ),
    ],
)
def test_merge(initial_state, direction, expected_state):
    result_state = Engine2048.merge(initial_state, direction)

    assert result_state == expected_state


@pytest.mark.parametrize("direction", [None, 4, "string"])
def test_merge_in_unsupported_direction(initial_state, direction):
    with pytest.raises(UnsupportedDirection):
        Engine2048.merge(initial_state, direction)


@pytest.mark.parametrize(
    "state, expect_game_over",
    [
        ([[1, 1], [0, 0]], False),
        ([[1, 1], [2, 2]], False),
        ([[0, 1], [2, 3]], False),
        ([[1, 2], [3, 4]], True),
    ],
)
def test_game_over(state, expect_game_over):
    assert Engine2048.check_game_over(state) is expect_game_over
