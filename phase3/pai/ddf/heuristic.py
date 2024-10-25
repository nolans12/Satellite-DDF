from phase3.pai.ddf import model


def generate_action(m: model.Model, state: model.State) -> model.Action:
    """Generate an action given the current state."""
    available_agents = set([agent.id for agent in m.agents])
    unobserved_targets = set(
        [
            target_id
            for target_id, target_state in enumerate(state.target_states)
            if not target_state.well_observed
        ]
    )
    unassigned_targets = unobserved_targets.copy()
    nearby_targets: dict[int, set[int]] = {agent.id: set() for agent in m.agents}

    # Find all targets within sensor range of each agent
    for agent in m.agents:
        for target in m.targets:
            if model.distance(agent.pos, target.pos) <= agent.sensor_range:
                nearby_targets[agent.id].add(target.id)

    observations = []

    # First, assign observations to targets only one agent can observe
    for agent in m.agents:
        nearby_targs = nearby_targets[agent.id].copy()
        for agent_id, targs in nearby_targets.items():
            if agent_id == agent.id:
                continue
            nearby_targs -= targs

        for target_id in nearby_targs.intersection(unassigned_targets):
            unassigned_targets.remove(target_id)
            available_agents.remove(agent.id)
            observations.append(model.ObservationAction(agent.id, target_id))
            # 1 observation per agent
            break

    # Next, assign observations to targets multiple agents can observe
    for agent_id in available_agents.copy():
        nearby_targs = nearby_targets[agent_id]
        for target_id in nearby_targs.intersection(unassigned_targets):
            unassigned_targets.remove(target_id)
            available_agents.remove(agent_id)
            observations.append(model.ObservationAction(agent_id, target_id))
            # 1 observation per agent
            break

    # Lastly, allow double observations for remaining agents
    for agent_id in available_agents:
        nearby_targs = nearby_targets[agent_id]
        for target_id in nearby_targs.intersection(unobserved_targets):
            observations.append(model.ObservationAction(agent_id, target_id))
            # 1 observation per agent
            break

    return model.Action(observations)
