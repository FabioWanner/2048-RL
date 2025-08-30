from random import Random
from typing import Callable, Sequence, Any
from unittest.mock import MagicMock

import pytest

from src.engine.engine import Direction, Engine2048, UnsupportedDirection
from src.engine.typings import State


@pytest.fixture
def initial_state() -> State:
    return [[0, 8, 2, 2], [4, 2, 0, 2], [0, 0, 0, 0], [0, 0, 0, 2]]


@pytest.fixture
def choice_fn() -> Callable[[Sequence[Any]], Any]:
    rng = Random(0)
    return rng.choice


def test_generate_state(choice_fn):
    result_state = Engine2048.generate_state(3, [0, 3, 7], choice_fn)
    assert result_state == [[3, 3, 0], [3, 7, 3], [3, 3, 3]]


def test_generate_state_does_never_return_all_zero_tiles():
    choice_mock = MagicMock()
    choice_mock.side_effect = [0, 0, 0, 0, 0, 0, 0, 1]
    result_state = Engine2048.generate_state(2, [], choice_mock)

    assert result_state == [[0, 0], [0, 1]]


@pytest.mark.parametrize(
    "direction, expected_state, expected_score",
    [
        (
            Direction.LEFT,
            [[8, 4, 0, 0], [4, 4, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]],
            8,
        ),
        (
            Direction.RIGHT,
            [[0, 0, 8, 4], [0, 0, 4, 4], [0, 0, 0, 0], [0, 0, 0, 2]],
            8,
        ),
        (
            Direction.UP,
            [[4, 8, 2, 4], [0, 2, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
            4,
        ),
        (
            Direction.DOWN,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 8, 0, 2], [4, 2, 2, 4]],
            4,
        ),
    ],
)
def test_merge(initial_state, direction, expected_state, expected_score):
    result_state, result_score = Engine2048.merge(initial_state, direction)

    assert result_state == expected_state
    assert result_score == expected_score


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


def test_spawn_random(initial_state, choice_fn):
    result_state = Engine2048.spawn_random(initial_state, [2, 4], choice_fn)
    assert result_state == [[0, 8, 2, 2], [4, 2, 0, 2], [0, 0, 0, 0], [4, 0, 0, 2]]


def test_engine_init():
    engine = Engine2048(board_size=3, seed=0)
    assert engine.state == [[2, 2, 0], [2, 2, 2], [2, 2, 2]]
    assert engine.game_over == False
    assert engine.score == 0


def test_engine_evolve(initial_state):
    engine = Engine2048(board_size=4, seed=0)
    engine.state = initial_state

    assert engine.evolve(Direction.LEFT)

    assert engine.state == [
        [8, 4, 0, 0],
        [4, 4, 0, 0],
        [2, 0, 0, 0],
        [2, 0, 0, 0],
    ]
    assert engine.score == 8
    assert engine.game_over == False

    assert engine.evolve(Direction.DOWN)

    assert engine.state == [
        [0, 0, 0, 0],
        [8, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 8, 0, 2],
    ]
    assert engine.score == 20
    assert engine.game_over == False


def test_engine_evolve_when_game_already_over(initial_state):
    engine = Engine2048(board_size=2, seed=0)
    engine.state = [[1, 2], [3, 4]]
    assert not engine.evolve(Direction.LEFT)


def test_engine_evolve_when_game_over_after_move(initial_state):
    engine = Engine2048(board_size=2, seed=0)
    engine.state = [[1, 1], [3, 5]]
    assert not engine.evolve(Direction.LEFT)
    assert engine.state == [[2, 4], [3, 5]]


def test_engine_evolve_when_move_does_not_alter_state(initial_state):
    engine = Engine2048(board_size=2, seed=0)
    engine.state = [[1, 0], [0, 0]]
    assert not engine.evolve(Direction.LEFT)
