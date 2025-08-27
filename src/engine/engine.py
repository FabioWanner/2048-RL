import copy
from enum import Enum
from random import Random
from typing import List, Callable, Sequence, Any

from src.engine.typings import State, StateRow


class UnsupportedDirection(Exception):
    pass


class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Engine2048:
    init_choices = [0, 2]
    spawn_choices = [2, 4]

    def __init__(self, board_size: int, seed: int | None = None):
        self.rng = Random(seed)
        self.state = self.generate_state(board_size, self.init_choices, self.rng.choice)

    @staticmethod
    def generate_state(
        size: int, choices: List[int | None], choice_fn: Callable[[Sequence[int]], int]
    ) -> State:
        state = [choice_fn(choices) for _ in range(size * size)]

        if all(v == 0 for v in state):
            return Engine2048.generate_state(size, choices, choice_fn)
        return [state[i : i + size] for i in range(0, len(state), size)]

    @staticmethod
    def merge_row_left(row: StateRow) -> StateRow:
        row_length = len(row)
        row = [e for e in row if e != 0]

        merged_row: StateRow = []
        i = 0
        while i < len(row):
            current = row[i]
            if i + 1 < len(row) and row[i + 1] == current:
                merged_value = current * 2
                merged_row.append(merged_value)
                i += 2
            else:
                merged_row.append(current)
                i += 1

        if len(merged_row) < row_length:
            merged_row.extend([0] * (row_length - len(merged_row)))
        return merged_row

    @staticmethod
    def merge_left(state: State) -> State:
        return [Engine2048.merge_row_left(row) for row in state]

    @staticmethod
    def flip(state: State) -> State:
        return [list(reversed(row)) for row in state]

    @staticmethod
    def rotate_clockwise(state: State) -> State:
        return [[state[j][i] for j in range(len(state))] for i in range(len(state))]

    @staticmethod
    def merge(state: State, direction: Direction) -> State:
        if direction == Direction.LEFT:
            return Engine2048.merge_left(state)
        if direction == Direction.RIGHT:
            prepared_state = Engine2048.flip(state)
            next_state = Engine2048.merge_left(prepared_state)
            return Engine2048.flip(next_state)
        if direction == Direction.UP:
            prepared_state = Engine2048.rotate_clockwise(Engine2048.flip(state))
            next_state = Engine2048.merge_left(prepared_state)
            return Engine2048.flip(Engine2048.rotate_clockwise(next_state))
        if direction == Direction.DOWN:
            prepared_state = Engine2048.flip(Engine2048.rotate_clockwise(state))
            next_state = Engine2048.merge_left(prepared_state)
            return Engine2048.rotate_clockwise(Engine2048.flip(next_state))

        raise UnsupportedDirection(
            f"The given direction '{direction}' is not supported."
        )

    @staticmethod
    def check_game_over(state: State) -> bool:
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == 0:
                    return False
                if i != 0 and state[i - 1][j] == state[i][j]:
                    return False
                if j != 0 and state[i][j - 1] == state[i][j]:
                    return False
        return True

    @staticmethod
    def spawn_random(
        state: State,
        choices: List[int],
        choice_fn: Callable[[Sequence[Any]], Any],
    ) -> State:
        empty_field_indexes = [
            (i_col, i_row)
            for i_col in range(len(state))
            for i_row in range(len(state[i_col]))
            if state[i_col][i_row] == 0
        ]

        index = choice_fn(empty_field_indexes)

        new_state = copy.deepcopy(state)
        new_state[index[0]][index[1]] = choice_fn(choices)
        return new_state

    @property
    def game_over(self):
        return self.check_game_over(self.state)

    def evolve(self, direction: Direction) -> bool:
        if self.game_over:
            return False

        evolved_state = self.merge(self.state, direction)

        if self.state == evolved_state:
            return False

        self.state = evolved_state

        self.state = self.spawn_random(self.state, self.spawn_choices, self.rng.choice)

        if self.game_over:
            return False

        return True
