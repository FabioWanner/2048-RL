from enum import Enum

from src.engine.typings import State, StateRow


class UnsupportedDirection(Exception):
    pass


class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Engine2048:
    def __init__(self):
        pass

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
