import dataclasses

from matplotlib import animation
from matplotlib import axes
from matplotlib import figure
from matplotlib import image
from matplotlib import patches
from matplotlib import pyplot as plt

from common import path_utils
from phase3.pai import pai
from phase3.pai.maze import engine
from phase3.pai.maze import heuristic
from phase3.pai.maze import model


@dataclasses.dataclass
class ModelPlot:
    ax: axes.Axes
    maze: image.AxesImage


def plot_model(m: model.Model) -> tuple[figure.Figure, ModelPlot]:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    m.env.maze[m.goal_state.agent_pos.x][m.goal_state.agent_pos.y] = 2
    m.env.maze[m.state.agent_pos.x][m.state.agent_pos.y] = 3
    maze_im = ax.imshow(m.env.maze, cmap='viridis', interpolation='nearest')
    # Revert changes on the shallow copy
    m.env.maze[m.goal_state.agent_pos.x][m.goal_state.agent_pos.y] = 0
    m.env.maze[m.state.agent_pos.x][m.state.agent_pos.y] = 0

    plot = ModelPlot(ax, maze_im)

    return fig, plot


def animate_states(
    fig: figure.Figure,
    plot: ModelPlot,
    states: list[model.State],
    action_plans: list[list[model.Action] | None],
    m: model.Model,
) -> animation.FuncAnimation:
    maze = m.env.maze
    goal_pos = m.goal_state.agent_pos

    # Function to update the positions of the satellites and links
    def update(
        num: int, movements: list[patches.FancyArrow]
    ) -> list[patches.FancyArrow]:
        for movement in movements:
            movement.remove()
        movements.clear()

        state = states[num]
        action = actions[num]

        maze[goal_pos.x][goal_pos.y] = 2
        maze[state.agent_pos.x][state.agent_pos.y] = 3
        plot.maze.set_data(maze)
        # Revert changes on the shallow copy
        maze[goal_pos.x][goal_pos.y] = 0
        maze[state.agent_pos.x][state.agent_pos.y] = 0

        # Show the action plan taken
        if num < len(action_plans) - 1:
            action_plan = action_plans[num + 1]
            assert action_plan is not None
            pos = state.agent_pos
            for action in action_plan:
                new_pos = model.move(pos, action.direction)
                movements.append(
                    plot.ax.arrow(
                        pos.y,
                        pos.x,
                        new_pos.y - pos.y,
                        new_pos.x - pos.x,
                        head_width=0.1,
                        head_length=0.1,
                        fc='b',
                        ec='b',
                    )
                )
                pos = new_pos

        # Show the action taken as an arrow from the previous position to the new position
        if action is not None and num > 0:

            old_pos = states[num - 1].agent_pos
            color = 'r'
            movements.append(
                plot.ax.arrow(
                    old_pos.y,
                    old_pos.x,
                    state.agent_pos.y - old_pos.y,
                    state.agent_pos.x - old_pos.x,
                    head_width=0.1,
                    head_length=0.1,
                    fc=color,
                    ec=color,
                )
            )

        return movements

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(states),
        fargs=([],),
        interval=500,
        blit=False,
    )

    return ani


if __name__ == '__main__':
    # m = model.sample_model()
    m = model.simple_model()
    params = m.params
    fig, plot = plot_model(m)

    states, actions = engine.simulate(
        m,
        params.steps,
        # heuristic.generate_action,
        pai.plan,  # type: ignore
    )

    anim = animate_states(
        fig,
        plot,
        states,
        actions,
        m,
    )

    plt.show()

    if False:
        # Save a gif of the animation
        anim.save(
            str(path_utils.REPO_ROOT / 'phase3/pai/maze/maze.gif'),
            writer='imagemagick',
        )
