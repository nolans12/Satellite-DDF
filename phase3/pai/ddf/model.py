import dataclasses
import itertools


@dataclasses.dataclass
class Position:
    x: float
    y: float


@dataclasses.dataclass
class Agent:
    id: int
    pos: Position
    sensor_range: float
    neighbors: list[int]


@dataclasses.dataclass
class Targets:
    id: int
    pos: Position


@dataclasses.dataclass
class ObservationAction:
    agent_id: int
    target_id: int | None

    def __hash__(self) -> int:
        return hash((self.agent_id, self.target_id))


@dataclasses.dataclass
class Action:
    observations: list[ObservationAction]


@dataclasses.dataclass
class TargetState:
    well_observed: bool


@dataclasses.dataclass
class State:
    target_states: list[TargetState]


@dataclasses.dataclass
class PaiParameters:
    steps: int
    lambdas: list[tuple[tuple[State, State, Action], float]]
    etas: list[tuple[tuple[Action, Action], float]]
    prior_etas: list[tuple[Action, float]]


@dataclasses.dataclass
class Model:
    agents: list[Agent]
    targets: list[Targets]


def distance(pos1: Position, pos2: Position) -> float:
    return ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5


def sample_model() -> Model:
    # 4 agents at the corners of a 10x10 grid
    agents = [
        Agent(id=0, pos=Position(i, j), sensor_range=5, neighbors=[])
        for i, j in itertools.product((1, 9), repeat=2)
    ]
    for i, agent in enumerate(agents):
        agent.id = i
        agent.neighbors = [
            j
            for j, neighbor in enumerate(agents)
            # Nearest neighbors within 10 units
            if distance(agent.pos, neighbor.pos) <= 10 and i != j
        ]
    # 9 targets at the centers of a 3x3 grid
    targets = [
        Targets(id=0, pos=Position(i, j))
        for i, j in itertools.product((2, 5, 8), repeat=2)
    ]
    for i, target in enumerate(targets):
        target.id = i

    return Model(agents=agents, targets=targets)
