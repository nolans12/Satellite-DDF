import random
from typing import Callable

from phase3.pai.maze import model


def propagate(m: model.Model, action: model.Action) -> model.State:
    """Propagate the state forward one step given the action."""
    agent_pos = m.state.agent.pos

    maybe_new_pos = model.move(agent_pos, action.direction)

    # Check if the new position is invalid
    if not model.in_bounds(m.maze, maybe_new_pos):
        return m.state

    # There's a 0.0 chance of the agent failing to move
    if random.random() < 0.0:
        return m.state

    return model.State(model.Agent(maybe_new_pos), m.state.goal)


def simulate(
    m: model.Model,
    num_steps: int,
    heuristic: Callable[[model.Model], model.Action],
) -> tuple[list[model.State], list[model.Action | None]]:
    """Simulate the model for a number of steps."""
    states = [m.state]
    actions: list[model.Action | None] = [None]
    for _ in range(num_steps):
        action = heuristic(m)
        m.state = propagate(m, action)
        states.append(m.state)
        actions.append(action)

    return states, actions
