import dataclasses
import enum
import itertools
from typing import Generator

from phase3.pai import model


class Direction(enum.Enum):
    NULL = -1
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclasses.dataclass
class Position:
    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))


@dataclasses.dataclass
class ObservationAction:
    agent_id: int
    target_id: int | None

    def __hash__(self) -> int:
        return hash((self.agent_id, self.target_id))


@dataclasses.dataclass
class Action:
    direction: Direction

    def __str__(self) -> str:
        return self.direction.name


@dataclasses.dataclass
class State:
    agent_pos: Position


@dataclasses.dataclass
class Environment:
    goal_pos: Position
    maze: list[list[int]]


@dataclasses.dataclass
class Model:
    state: State
    env: Environment
    params: model.PaiParameters

    @property
    def total_states(self) -> int:
        return len(self.env.maze) * len(self.env.maze[0]) * len(Direction)

    @property
    def goal_state(self) -> State:
        return State(self.env.goal_pos)

    def action_hash(self, action: Action) -> int:
        return action.direction.value + 1

    def state_action_hash(self, state: State, action: Action) -> int:
        width = len(self.env.maze)
        size = width * len(self.env.maze[0])
        return (
            state.agent_pos.x
            + state.agent_pos.y * width
            + self.action_hash(action) * size
        )

    def actions(self) -> Generator[Action, None, None]:
        for direction in Direction:
            yield Action(direction)

    def states(self) -> Generator[State, None, None]:
        for x, y in itertools.product(
            range(len(self.env.maze)), range(len(self.env.maze[0]))
        ):
            yield State(Position(x, y))

    def valid_state(self, state: State) -> bool:
        maze = self.env.maze
        pos = state.agent_pos
        in_bounds = 0 <= pos.x < len(maze[0]) and 0 <= pos.y < len(maze)
        if not in_bounds:
            return False
        obstacle = maze[pos.x][pos.y] == 1
        return not obstacle

    def valid_action(self, state: State, state_p: State, action: Action) -> bool:
        dist = grid_distance(state.agent_pos, state_p.agent_pos)
        too_far = dist > 1
        cant_move = action.direction is Direction.NULL and dist == 1
        return not too_far and not cant_move

    def lambda_sp_s_a(self, state_p: State, state: State, action: Action) -> float:
        if grid_distance(state_p.agent_pos, state.agent_pos) > 1:
            return 0

        intended_pos = move(state.agent_pos, action.direction)
        if intended_pos == state_p.agent_pos:
            if action.direction is Direction.NULL:
                return 1
            return self.params.lambda_sp_s_a
        elif action.direction is Direction.NULL:
            return 0

        if state_p == state:
            return self._p_false_move(state, action)
        return (1 - self.params.lambda_sp_s_a) / 3

    def _p_false_move(self, state: State, action: Action) -> float:
        if action.direction is Direction.NULL:
            return 1

        p = 0
        for d in Direction:
            if d is Direction.NULL:
                continue
            new_pos = move(state.agent_pos, d)
            val = (
                self.params.lambda_sp_s_a
                if d == action.direction
                else (1 - self.params.lambda_sp_s_a) / 3
            )
            if not self.valid_state(State(new_pos)):
                p += val
        return p


def model_parameters(maze: list[list[int]], extra_steps: int) -> model.PaiParameters:
    return model.PaiParameters(
        steps=int(len(maze) + len(maze[0]) - 2 + extra_steps),
        lambda_sp_s_a=0.7,
        eta_ap_a=0.2,
        eta_a=0.2,
    )


def grid_distance(pos1: Position, pos2: Position) -> int:
    return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)


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

    state = State(Position(4, 0))
    env = Environment(Position(4, 7), maze)
    params = model_parameters(maze, extra_steps=2)

    return Model(state, env, params)


def simple_model() -> Model:
    # Create a 4x4 grid with a single agent, goal, and some obstacles
    # maze = [
    #     [0, 0, 1, 0],
    #     [1, 0, 1, 0],
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    # ]

    # state = State(Agent(Position(0, 0)), Goal(Position(3, 3)))

    # Create a 3x3 grid with a single agent, goal, and some obstacles
    maze = [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]

    state = State(Position(0, 0))
    env = Environment(Position(2, 2), maze)
    params = model_parameters(maze, extra_steps=2)

    return Model(state, env, params)
