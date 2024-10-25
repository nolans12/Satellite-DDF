import random
from typing import Callable

from phase3.pai.ddf import model


def propagate(state: model.State, action: model.Action) -> model.State:
    """Propagate the state forward one step given the action."""
    target_states = state.target_states[:]
    for obs in action.observations:
        if obs.target_id is None:
            continue
        # Large chance of observing the target successfully
        if random.random() < 0.85:
            target_states[obs.target_id] = model.TargetState(True)

    return model.State(target_states)


def simulate(
    m: model.Model,
    init_state: model.State,
    num_steps: int,
    heuristic: Callable[[model.Model, model.State], model.Action],
) -> tuple[list[model.State], list[model.Action | None]]:
    """Simulate the model for a number of steps."""
    state = init_state
    states = [state]
    actions: list[model.Action | None] = [None]
    for _ in range(num_steps):
        action = heuristic(m, state)
        state = propagate(state, action)
        states.append(state)
        actions.append(action)

    return states, actions
