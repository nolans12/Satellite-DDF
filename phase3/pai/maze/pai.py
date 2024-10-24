import itertools

import numpy as np
from numpy import typing as npt

from phase3.pai.maze import model


def model_parameters(m: model.Model) -> model.PaiParameters:
    return model.PaiParameters(
        steps=int(len(m.maze) + len(m.maze[0]) - 2 + 2),
        lambda_sp_s_a=0.7,
        eta_ap_a=0.2,
        eta_a=0.2,
    )


def _direction_hash(direction: model.Direction) -> int:
    return direction.value + 1


def _state_action_hash(
    pos: model.Position, direction: model.Direction, maze_shape: tuple[int, int]
) -> int:
    width = maze_shape[0]
    size = width * maze_shape[1]
    return pos.x + pos.y * width + _direction_hash(direction) * size


def p_false_move(
    m: model.Model,
    params: model.PaiParameters,
    pos: model.Position,
    direction: model.Direction,
) -> float:
    """Find the probability of the agent failing to move at all."""
    if direction is model.Direction.NULL:
        return 1

    p = 0
    for d in model.Direction:
        if d is model.Direction.NULL:
            continue
        new_pos = model.move(pos, d)
        val = params.lambda_sp_s_a if d == direction else (1 - params.lambda_sp_s_a) / 3
        if not model.in_bounds(m.maze, new_pos):
            p += val
    return p


def lambda_sp_s_a(
    m: model.Model,
    params: model.PaiParameters,
    pos_p: model.Position,
    pos: model.Position,
    direction: model.Direction,
) -> float:
    """Find the probability of arriving at a state given a previous state and action."""
    if model.grid_distance(pos_p, pos) > 1:
        return 0

    intended_pos = model.move(pos, direction)
    if intended_pos == pos_p:
        if direction is model.Direction.NULL:
            return 1
        return params.lambda_sp_s_a
    elif direction is model.Direction.NULL:
        return 0
    if pos_p == pos:
        return p_false_move(m, params, pos, direction)
    return (1 - params.lambda_sp_s_a) / 3


def plan(m: model.Model, params: model.PaiParameters) -> list[model.Action]:
    total_states = len(m.maze) * len(m.maze[0]) * len(model.Direction)
    shape = (len(m.maze), len(m.maze[0]))

    init_pos = m.state.agent.pos

    beta_hash = np.zeros((params.steps + 1, total_states))
    beta_hash[params.steps, :] = 1

    alpha_inter_hash = np.zeros((params.steps + 1, total_states, total_states))
    alpha_hash = np.zeros((params.steps + 1, total_states))
    alpha_action_hash = np.zeros(
        (params.steps, len(model.Direction), len(model.Direction))
    )

    print(f'{init_pos=}')

    # Backwards pass
    for t in range(params.steps - 1, 0, -1):
        for x, y, direction in itertools.product(
            range(len(m.maze)), range(len(m.maze[0])), model.Direction
        ):
            pos = model.Position(x, y)

            if not model.in_bounds(m.maze, pos):
                # lambda(s', s, a) = 0 for all s'
                continue

            for x_p, y_p, direction_p in itertools.product(
                range(len(m.maze)), range(len(m.maze[0])), model.Direction
            ):
                pos_p = model.Position(x_p, y_p)

                if (
                    not model.in_bounds(m.maze, pos_p)
                    or model.grid_distance(pos, pos_p) > 1
                    or (
                        direction is model.Direction.NULL
                        and model.grid_distance(pos, pos_p) == 1
                    )
                ):
                    # lambda(s', s, a) = 0
                    continue

                # Always the same (uniform distribution)
                eta_ap_a = params.eta_ap_a

                value = (
                    lambda_sp_s_a(m, params, pos_p, pos, direction)
                    * eta_ap_a
                    * beta_hash[t + 1, _state_action_hash(pos_p, direction_p, shape)]
                )
                if t == params.steps - 1:
                    value *= lambda_sp_s_a(
                        m, params, m.state.goal.pos, pos_p, direction_p
                    )

                beta_hash[t, _state_action_hash(pos, direction, shape)] += value

    for t in range(params.steps - 1, 0, -1):
        for x, y, direction in itertools.product(
            range(len(m.maze)), range(len(m.maze[0])), model.Direction
        ):
            pos = model.Position(x, y)
            for x_p, y_p, direction_p in itertools.product(
                range(len(m.maze)), range(len(m.maze[0])), model.Direction
            ):
                pos_p = model.Position(x_p, y_p)
                value = (
                    lambda_sp_s_a(m, params, pos_p, pos, direction)
                    * params.eta_ap_a
                    * beta_hash[t + 1, _state_action_hash(pos_p, direction_p, shape)]
                    / (
                        beta_hash[t, _state_action_hash(pos, direction, shape)]
                        or np.inf
                    )
                )
                if t == params.steps - 1:
                    value *= lambda_sp_s_a(
                        m, params, m.state.goal.pos, pos_p, direction_p
                    )
                alpha_inter_hash[
                    t,
                    _state_action_hash(pos_p, direction_p, shape),
                    _state_action_hash(pos, direction, shape),
                ] = value

    # Only consider the initial state for step=0
    for direction in model.Direction:
        for direction_p in model.Direction:
            value = (
                params.eta_a
                * beta_hash[
                    1,
                    _state_action_hash(init_pos, direction_p, shape),
                ]
            )
            if params.steps == 1:
                value *= lambda_sp_s_a(
                    m, params, m.state.goal.pos, init_pos, direction_p
                )
            beta_hash[0, _state_action_hash(init_pos, direction, shape)] += value

    # print(f'beta_hash=\n{beta_hash}')

    # Forward pass step=0
    for direction in model.Direction:
        value = (
            params.eta_a
            * beta_hash[1, _state_action_hash(init_pos, direction, shape)]
            / beta_hash[0, _state_action_hash(init_pos, direction, shape)]
        )
        if params.steps == 1:
            value *= lambda_sp_s_a(m, params, m.state.goal.pos, init_pos, direction)
        alpha_hash[0, _state_action_hash(init_pos, direction, shape)] = value
        alpha_action_hash[0, _direction_hash(direction), :] = value

    # Forward pass
    for t in range(1, params.steps):
        for x_p, y_p, direction_p in itertools.product(
            range(len(m.maze)), range(len(m.maze[0])), model.Direction
        ):
            pos_p = model.Position(x_p, y_p)

            alpha_hash[t, _state_action_hash(pos_p, direction_p, shape)] = np.dot(
                alpha_inter_hash[t, _state_action_hash(pos_p, direction_p, shape), :],
                alpha_hash[t - 1, :],
            )

        for direction_p in model.Direction:
            for direction in model.Direction:
                denominator = 0
                numerator = 0
                for x, y in itertools.product(
                    range(len(m.maze)), range(len(m.maze[0]))
                ):
                    pos = model.Position(x, y)

                    denominator += alpha_hash[
                        t, _state_action_hash(pos, direction, shape)
                    ]

                    for x_p, y_p in itertools.product(
                        range(len(m.maze)), range(len(m.maze[0]))
                    ):
                        pos_p = model.Position(x_p, y_p)

                        numerator += (
                            alpha_inter_hash[
                                t,
                                _state_action_hash(pos_p, direction_p, shape),
                                _state_action_hash(pos, direction, shape),
                            ]
                            * alpha_hash[
                                t - 1, _state_action_hash(pos, direction, shape)
                            ]
                        )

                # XXX 3x3 [[0, 0, 0], [1, 1, 0], [0, 0, 0]] super sus (4.9)
                # XXX initial plan for 4x4 is sus
                alpha_action_hash[
                    t, _direction_hash(direction_p), _direction_hash(direction)
                ] = ((numerator / denominator) if denominator != 0 else 0)

    # print(f'alpha_action_hash=\n{alpha_action_hash}')

    # TODO: Use the Viterbi algorithm to find the best action
    best_direction_sequence = None
    best_direction_sequence_prob = 0
    i = 0
    for direction_sequence in itertools.product(model.Direction, repeat=params.steps):
        i += 1
        if i % 10_000_000 == 0:
            print(f'{i=}')
        prob = 1
        prob *= alpha_action_hash[
            0,
            _direction_hash(direction_sequence[0]),
            _direction_hash(model.Direction.NULL),
        ]
        for t in range(1, params.steps):
            prob *= alpha_action_hash[
                t,
                _direction_hash(direction_sequence[t]),
                _direction_hash(direction_sequence[t - 1]),
            ]

        if prob > best_direction_sequence_prob:
            best_direction_sequence = direction_sequence
            best_direction_sequence_prob = prob

    assert best_direction_sequence is not None

    print(best_direction_sequence)

    return [model.Action(direction) for direction in best_direction_sequence]

    # # Greedily find the best action at each step
    # greedy_action_sequence = []
    # initial_index = np.argmax(alpha_action_hash[0, :, :])
    # greedy_action_sequence.append(
    #     _index_to_action(initial_index // len(model.Direction))
    # )

    # for t in range(1, params.steps):
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
