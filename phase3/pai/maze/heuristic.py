from phase3.pai.maze import model


def generate_action(m: model.Model) -> list[model.Action]:
    """Generate an action given the current state.

    The alorithm just runs A* search to find the shortest path to the target.
    """
    agent_pos = m.state.agent_pos
    target_pos = m.goal_state.agent_pos

    # A* search
    open_set = {agent_pos}
    came_from: dict[model.Position, tuple[model.Position, model.Direction]] = {}
    g_score = {agent_pos: 0}
    f_score = {agent_pos: model.distance(agent_pos, target_pos)}

    while open_set:
        current = min(open_set, key=lambda pos: f_score[pos])
        if current == target_pos:
            break

        open_set.remove(current)
        for direction in model.Direction:
            new_pos = model.move(current, direction)
            if not m.valid_state(model.State(new_pos)):
                continue

            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(new_pos, float('inf')):
                came_from[new_pos] = (current, direction)
                g_score[new_pos] = tentative_g_score
                f_score[new_pos] = tentative_g_score + model.distance(
                    new_pos, target_pos
                )
                open_set.add(new_pos)

    # Reconstruct the path
    direction = model.Direction.NULL
    path: list[model.Direction] = [direction]
    current = target_pos
    while current != agent_pos:
        path.append(direction)
        current, direction = came_from[current]
    path.append(direction)
    path.reverse()

    # Move the agent towards the target
    return [model.Action(p) for p in path]
