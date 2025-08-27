from enum import Enum

from src.engine.typings import State


class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Engine2048:
    def __init__(self):
        pass

    @staticmethod
    def merge(state: State, direction: Direction) -> State:
        return state
