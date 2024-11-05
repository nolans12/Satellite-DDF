import random
from typing import Callable

from phase3.pai.maze import model


def propagate(m: model.Model, action: model.Action) -> model.State:
    """Propagate the state forward one step given the action."""
    if action.direction is model.Direction.NULL:
        return m.state

    agent_pos = m.state.agent_pos

    direction = action.direction
    # There's a 0.3 chance of the agent moving to another state
    if (rand := random.random()) < 0.0:
        direction = model.Direction((direction.value + 1 + int(rand * 10)) % 4)

    maybe_new_pos = model.move(agent_pos, direction)

    # Check if the new position is invalid
    if not m.valid_state(model.State(maybe_new_pos)):
        return m.state

    return model.State(maybe_new_pos)


def simulate(
    m: model.Model,
    num_steps: int,
    planner: Callable[[model.Model], list[model.Action]],
) -> tuple[list[model.State], list[list[model.Action] | None]]:
    """Simulate the model for a number of steps."""
    states = [m.state]
    actions: list[list[model.Action] | None] = [None]
    for _ in range(num_steps):
        action_plan = planner(m)
        m.state = propagate(m, action_plan[0])
        m.params.steps -= 1
        states.append(m.state)
        actions.append(action_plan)

    return states, actions
