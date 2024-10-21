import dataclasses

from matplotlib import animation
from matplotlib import artist
from matplotlib import axes
from matplotlib import figure
from matplotlib import lines
from matplotlib import patches
from matplotlib import pyplot as plt

from phase3.pai.ddf import engine
from phase3.pai.ddf import heuristic
from phase3.pai.ddf import model


@dataclasses.dataclass
class ModelPlot:
    ax: axes.Axes
    agents: list[lines.Line2D]
    network_edges: list[lines.Line2D]
    targets: list[lines.Line2D]
    sensor_ranges: list[artist.Artist]


def plot_model(m: model.Model) -> tuple[figure.Figure, ModelPlot]:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    plot = ModelPlot(ax, [], [], [], [])

    # Plot agents as blue dots and targets as red dots
    for agent in m.agents:
        plot.agents.extend(ax.plot(agent.pos.x, agent.pos.y, 'bo'))
    for target in m.targets:
        plot.targets.extend(ax.plot(target.pos.x, target.pos.y, 'ro'))

    # Plot lines between agents and their neighbors
    for agent in m.agents:
        for neighbor_id in agent.neighbors:
            neighbor = m.agents[neighbor_id]
            plot.network_edges.extend(
                ax.plot(
                    [agent.pos.x, neighbor.pos.x], [agent.pos.y, neighbor.pos.y], 'g-'
                )
            )

    # Plot the sensor range as a circle around each agent
    for agent in m.agents:
        circle = patches.Circle(
            (agent.pos.x, agent.pos.y), agent.sensor_range, color='black', fill=False
        )
        plot.sensor_ranges.append(ax.add_artist(circle))

    return fig, plot


def animate_states(
    fig: figure.Figure,
    plot: ModelPlot,
    states: list[model.State],
    actions: list[model.Action | None],
) -> animation.FuncAnimation:
    # Function to update the positions of the satellites and links
    def update(
        num: int, targets: list[lines.Line2D], observations: list[lines.Line2D]
    ) -> list[lines.Line2D]:
        print(num)

        # Clear previous observations
        for obs in observations:
            obs.remove()

        state = states[num]
        action = actions[num]
        for i, target in enumerate(targets):
            target.set_color('g' if state.target_states[i].well_observed else 'r')
        if action is None:
            return targets

        observations.clear()
        for obs in action.observations:
            if obs.target_id is not None:
                agent_pos = plot.agents[obs.agent_id].get_data()
                target_pos = plot.targets[obs.target_id].get_data()
                observations.extend(
                    plot.ax.plot(
                        (agent_pos[0], target_pos[0]),  # type: ignore
                        (agent_pos[1], target_pos[1]),  # type: ignore
                        'b--',
                    )
                )

        return targets + observations

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(states),
        fargs=(plot.targets, []),
        interval=1000,
        blit=False,
    )

    return ani


if __name__ == '__main__':
    m = model.sample_model()
    states, actions = engine.simulate(
        m,
        model.State([model.TargetState(False) for _ in m.targets]),
        3,
        heuristic.generate_action,
    )

    fig, plot = plot_model(m)

    anim = animate_states(
        fig,
        plot,
        states,
        actions,
    )

    plt.show()
