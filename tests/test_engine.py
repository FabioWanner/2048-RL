import pytest

from src.engine.engine import Direction, Engine2048
from src.engine.typings import State


@pytest.fixture
def initial_state() -> State:
    return [[0, 8, 2, 2], [4, 2, 0, 2], [0, 0, 0, 0], [0, 0, 0, 2]]


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
