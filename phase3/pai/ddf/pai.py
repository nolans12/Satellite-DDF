import itertools
from collections import defaultdict

from phase3.pai.ddf import model


def model_parameters(m: model.Model) -> model.PaiParameters:
    # The probability of an agent observing any target is 0.4;
    # for any target is 0.4 / (num of targets in range of agent) if agent is in range, 0 otherwise

    # Find the number of targets in range of each agent
    num_targets_in_range = {agent.id: 0 for agent in m.agents}
    for agent in m.agents:
        for target in m.targets:
            if model.distance(agent.pos, target.pos) <= agent.sensor_range:
                num_targets_in_range[agent.id] += 1

    # Calculate observation actions & probabilities for each agent
    agent_observation_actions: dict[int, dict[model.ObservationAction, float]] = (
        defaultdict(dict)
    )
    for agent in m.agents:
        for target in m.targets:
            if model.distance(agent.pos, target.pos) <= agent.sensor_range:
                agent_observation_actions[agent.id][
                    model.ObservationAction(agent.id, target.id)
                ] = (0.4 / num_targets_in_range[agent.id])
        agent_observation_actions[agent.id][
            model.ObservationAction(agent.id, None)
        ] = 0.6

    # Calculate priors of each action
    prior_etas: list[tuple[model.Action, float]] = []
    for observation_actions in itertools.product(*agent_observation_actions.values()):
        prior = 1
        for agent_id, observation_action in zip(
            agent_observation_actions.keys(), observation_actions
        ):
            prior *= agent_observation_actions[agent_id][observation_action]
        prior_etas.append((model.Action(list(observation_actions)), prior))

    # Calculate priors of each action pair
    # This is uniform across each agent's actions
    etas = []
    for new_action, _ in prior_etas:
        for old_action, _ in prior_etas:
            etas.append(((new_action, old_action), 1 / len(prior_etas)))

    # For each state pair, calculate the probability of transitioning between them.
    # An observation action has a 0.9 probability of successfully observing a target
    all_states = [
        model.State(
            [model.TargetState(well_observed) for well_observed in well_observed_states]
        )
        for well_observed_states in itertools.product(
            [True, False], repeat=len(m.targets)
        )
    ]
    lambdas = []
    for new_state in all_states:
        for old_state in all_states:
            for action, _ in prior_etas:
                # Find the number of flips between the two states
                num_flips = 0
                # Find the number of "stays" between the two states
                num_stays = 0
                impossible = False
                for target_id, (new_target_state, old_target_state) in enumerate(
                    zip(new_state.target_states, old_state.target_states)
                ):
                    if (
                        not new_target_state.well_observed
                        and old_target_state.well_observed
                    ):
                        impossible = True
                        break
                    if new_target_state.well_observed != old_target_state.well_observed:
                        # Check if an observation action made the target well-observed
                        for obs_action in action.observations:
                            if obs_action.target_id == target_id:
                                num_flips += 1
                            else:
                                impossible = True
                                break
                    elif not new_target_state.well_observed:
                        # Check if an observation action failed to make the target well-observed
                        for obs_action in action.observations:
                            if obs_action.target_id == target_id:
                                num_stays += 1
                if impossible:
                    continue
                probability = 0.9**num_flips * 0.1**num_stays
                lambdas.append(((new_state, old_state, action), probability))

    return model.PaiParameters(
        steps=3, lambdas=lambdas, etas=etas, prior_etas=prior_etas
    )
