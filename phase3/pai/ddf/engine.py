import random
from typing import Callable

from phase3.pai.ddf import model


def propagate(state: model.State, action: model.Action) -> model.State:
    """Propagate the state forward one step given the action."""
    new_target_states = []
    for target_id, target_state in enumerate(state.target_states):
        for obs in action.observations:
            if obs.target_id == target_id:
                observation = obs
                break
        else:
            observation = None

        # 0.9 chance of observing the target successfully
        new_target_state = target_state
        if observation is not None and random.random() < 0.9:
            new_target_state = model.TargetState(True)

        new_target_states.append(new_target_state)

    return model.State(new_target_states)


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
