import dataclasses
import enum


class Direction(enum.Enum):
    UP = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()
    NULL = enum.auto()


@dataclasses.dataclass
class Position:
    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))


@dataclasses.dataclass
class Agent:
    pos: Position


@dataclasses.dataclass
class Goal:
    pos: Position


@dataclasses.dataclass
class ObservationAction:
    agent_id: int
    target_id: int | None

    def __hash__(self) -> int:
        return hash((self.agent_id, self.target_id))


@dataclasses.dataclass
class Action:
    direction: Direction


@dataclasses.dataclass
class State:
    agent: Agent
    goal: Goal


@dataclasses.dataclass
class PaiParameters:
    steps: int
    lambdas: list[tuple[tuple[State, State, Action], float]]
    etas: list[tuple[tuple[Action, Action], float]]
    prior_etas: list[tuple[Action, float]]


@dataclasses.dataclass
class Model:
    state: State
    maze: list[list[int]]


def in_bounds(maze: list[list[int]], pos: Position) -> bool:
    in_bounds = 0 <= pos.x < len(maze[0]) and 0 <= pos.y < len(maze)
    if not in_bounds:
        return False
    obstacle = maze[pos.x][pos.y] == 1
    return not obstacle


def distance(pos1: Position, pos2: Position) -> float:
    return ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5


def move(pos: Position, direction: Direction) -> Position:
    match direction:
        case Direction.UP:
            return Position(pos.x, pos.y + 1)
        case Direction.DOWN:
            return Position(pos.x, pos.y - 1)
        case Direction.LEFT:
            return Position(pos.x - 1, pos.y)
        case Direction.RIGHT:
            return Position(pos.x + 1, pos.y)
        case Direction.NULL:
            return pos
        case _:
            raise ValueError(f'Invalid direction: {direction}')


def sample_model() -> Model:
    # Create a 8x8 grid with a single agent, goal, and some obstacles
    maze = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    state = State(Agent(Position(4, 0)), Goal(Position(5, 7)))

    return Model(state, maze)
