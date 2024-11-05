import dataclasses
from typing import Generator, Protocol


@dataclasses.dataclass
class PaiParameters:
    # Number of steps to plan over
    steps: int
    # Prior of arriving at a (nominal) state given a previous state and action
    lambda_sp_s_a: float
    # Prior of taking an action given a previous action
    eta_ap_a: float
    # Prior of taking a given action
    eta_a: float


class State(Protocol): ...


class Action(Protocol): ...


class Model(Protocol):
    state: State
    params: PaiParameters

    @property
    def total_states(self) -> int: ...

    @property
    def goal_state(self) -> State: ...

    def action_hash(self, action: Action) -> int: ...

    def state_action_hash(self, state: State, action: Action) -> int: ...

    def actions(self) -> Generator[Action, None, None]: ...

    def states(self) -> Generator[State, None, None]: ...

    def valid_state(self, state: State) -> bool: ...

    def valid_action(self, state: State, state_p: State, action: Action) -> bool: ...

    def lambda_sp_s_a(self, state_p: State, state: State, action: Action) -> float: ...
