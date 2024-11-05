import itertools

import numpy as np
from numpy import typing as npt

from phase3.pai import model


def plan(m: model.Model) -> list[model.Action]:
    total_states = m.total_states

    init_state = m.state

    beta_hash = np.zeros((m.params.steps + 1, total_states))
    beta_hash[m.params.steps, :] = 1

    alpha_inter_hash = np.zeros((m.params.steps + 1, total_states, total_states))
    alpha_hash = np.zeros((m.params.steps + 1, total_states))
    alpha_action_hash = np.zeros(
        (m.params.steps, len(list(m.actions())), len(list(m.actions())))
    )

    print(f'{init_state=}')
    print(f'{m.params=}')

    # Backwards pass
    for t in range(m.params.steps - 1, 0, -1):
        for state, action in itertools.product(m.states(), m.actions()):
            if not m.valid_state(state):
                # lambda(s', s, a) = 0 for all s'
                continue

            for state_p, action_p in itertools.product(m.states(), m.actions()):
                if not m.valid_state(state_p) or not m.valid_action(
                    state, state_p, action
                ):
                    # lambda(s', s, a) = 0
                    continue

                # Always the same (uniform distribution)
                eta_ap_a = m.params.eta_ap_a

                if t == 3:
                    print(state_p, state, action)
                    print(m.lambda_sp_s_a(state_p, state, action))

                value = (
                    m.lambda_sp_s_a(state_p, state, action)
                    * eta_ap_a
                    * beta_hash[t + 1, m.state_action_hash(state_p, action_p)]
                )
                if t == m.params.steps - 1:
                    value *= m.lambda_sp_s_a(m.goal_state, state_p, action_p)

                beta_hash[t, m.state_action_hash(state, action)] += value

    for t in range(m.params.steps - 1, 0, -1):
        for state, action in itertools.product(m.states(), m.actions()):
            for state_p, action_p in itertools.product(m.states(), m.actions()):
                value = (
                    m.lambda_sp_s_a(state_p, state, action)
                    * m.params.eta_ap_a
                    * beta_hash[t + 1, m.state_action_hash(state_p, action_p)]
                    / (beta_hash[t, m.state_action_hash(state, action)] or np.inf)
                )
                if t == m.params.steps - 1:
                    value *= m.lambda_sp_s_a(m.goal_state, state_p, action_p)
                alpha_inter_hash[
                    t,
                    m.state_action_hash(state_p, action_p),
                    m.state_action_hash(state, action),
                ] = value

    print(f'{beta_hash=}')

    # Only consider the initial state for step=0
    for action in m.actions():
        for action_p in m.actions():
            value = (
                m.params.eta_a
                * beta_hash[
                    1,
                    m.state_action_hash(init_state, action_p),
                ]
            )
            if m.params.steps == 1:
                value *= m.lambda_sp_s_a(m.goal_state, init_state, action_p)
            beta_hash[0, m.state_action_hash(init_state, action)] += value

    # print(f'beta_hash=\n{beta_hash}')

    # Forward pass step=0
    for action in m.actions():
        value = (
            m.params.eta_a
            * beta_hash[1, m.state_action_hash(init_state, action)]
            / beta_hash[0, m.state_action_hash(init_state, action)]
        )
        if m.params.steps == 1:
            value *= m.lambda_sp_s_a(m.goal_state, init_state, action)
        alpha_hash[0, m.state_action_hash(init_state, action)] = value
        alpha_action_hash[0, m.action_hash(action), :] = value

    # print(f'{alpha_hash=}')
    # print(f'{alpha_inter_hash=}')

    # Forward pass
    for t in range(1, m.params.steps):
        for state_p, action_p in itertools.product(m.states(), m.actions()):
            alpha_hash[t, m.state_action_hash(state_p, action_p)] = np.dot(
                alpha_inter_hash[t, m.state_action_hash(state_p, action_p), :],
                alpha_hash[t - 1, :],
            )

        for action_p in m.actions():
            for action in m.actions():
                denominator = 0
                numerator = 0
                for state in m.states():
                    denominator += alpha_hash[t - 1, m.state_action_hash(state, action)]

                    for state_p in m.states():
                        numerator += (
                            alpha_inter_hash[
                                t,
                                m.state_action_hash(state_p, action_p),
                                m.state_action_hash(state, action),
                            ]
                            * alpha_hash[t - 1, m.state_action_hash(state, action)]
                        )

                alpha_action_hash[t, m.action_hash(action_p), m.action_hash(action)] = (
                    (numerator / denominator) if denominator != 0 else 0
                )

    print(f'alpha_action_hash=\n{alpha_action_hash}')

    # TODO: Use the Viterbi algorithm to find the best action
    best_action_sequence = None
    best_action_sequence_prob = 0
    i = 0
    for action_sequence in itertools.product(m.actions(), repeat=m.params.steps):
        i += 1
        if i % 10_000_000 == 0:
            print(f'{i=}')
        prob = 1
        prob *= alpha_action_hash[
            0,
            m.action_hash(action_sequence[0]),
            m.action_hash(  # This index doesn't matter; should be the same
                action_sequence[0]
            ),
        ]
        for t in range(1, m.params.steps):
            prob *= alpha_action_hash[
                t,
                m.action_hash(action_sequence[t]),
                m.action_hash(action_sequence[t - 1]),
            ]

        if prob > best_action_sequence_prob:
            best_action_sequence = action_sequence
            best_action_sequence_prob = prob

    assert best_action_sequence is not None

    print(best_action_sequence)

    return list(best_action_sequence)

    # # Greedily find the best action at each step
    # greedy_action_sequence = []
    # initial_index = np.argmax(alpha_action_hash[0, :, :])
    # greedy_action_sequence.append(
    #     _index_to_action(initial_index // len(model.Direction))
    # )

    # for t in range(1, m.params.steps):
    #     best_index = np.argmax(
    #         alpha_action_hash[
    #             t, :, _direction_hash(greedy_action_sequence[-1].direction)
    #         ]
    #     )
    #     greedy_action_sequence.append(
    #         _index_to_action(best_index // len(model.Direction))
    #     )

    # print(greedy_action_sequence)

    # return greedy_action_sequence[0]


def viterbi(
    states: npt.NDArray,
    init: npt.NDArray,
    trans: npt.NDArray,
    emit: npt.NDArray,
    obs: npt.NDArray,
) -> npt.NDArray:
    """Compute the Viterbi algorithm for a Hidden Markov Model."""
    prob = np.zeros((len(trans), len(states)))
    prev = np.zeros((len(trans), len(states)))
    for s in range(states):
        prob[0, s] = init[s] * emit[s, obs[0]]

    for t in range(1, len(obs)):
        for s in range(states):
            for r in range(states):
                new_prob = prob[t - 1, r] * trans[r, s] * emit[s, obs[t]]
                if new_prob > prob[t, s]:
                    prob[t, s] = new_prob
                    prev[t, s] = r

    # Find the best path
    best_path = np.zeros(len(obs))
    best_path[-1] = np.argmax(prob[-1])
    for t in range(len(obs) - 2, -1, -1):
        best_path[t] = prev[t + 1, best_path[t + 1]]

    return best_path
