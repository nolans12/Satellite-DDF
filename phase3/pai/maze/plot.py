import dataclasses

from matplotlib import animation
from matplotlib import artist
from matplotlib import axes
from matplotlib import figure
from matplotlib import image
from matplotlib import lines
from matplotlib import patches
from matplotlib import pyplot as plt

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

    m.maze[m.state.goal.pos.x][m.state.goal.pos.y] = 2
    m.maze[m.state.agent.pos.x][m.state.agent.pos.y] = 3
    maze_im = ax.imshow(m.maze, cmap='viridis', interpolation='nearest')
    # Revert changes on the shallow copy
    m.maze[m.state.goal.pos.x][m.state.goal.pos.y] = 0
    m.maze[m.state.agent.pos.x][m.state.agent.pos.y] = 0

    plot = ModelPlot(ax, maze_im)

    return fig, plot


def animate_states(
    fig: figure.Figure,
    plot: ModelPlot,
    states: list[model.State],
    actions: list[model.Action | None],
    maze: list[list[int]],
) -> animation.FuncAnimation:
    # Function to update the positions of the satellites and links
    def update(
        num: int, movements: list[patches.FancyArrow]
    ) -> list[patches.FancyArrow]:
        for movement in movements:
            movement.remove()
        movements.clear()

        state = states[num]
        action = actions[num]

        maze[state.goal.pos.x][state.goal.pos.y] = 2
        maze[state.agent.pos.x][state.agent.pos.y] = 3
        plot.maze.set_data(maze)
        # Revert changes on the shallow copy
        maze[state.goal.pos.x][state.goal.pos.y] = 0
        maze[state.agent.pos.x][state.agent.pos.y] = 0

        # Show the action taken as an arrow from the previous position to the new position
        if action is not None and num > 0:
            old_pos = states[num - 1].agent.pos
            color = 'r' if old_pos == state.agent.pos else 'b'
            movements.append(
                plot.ax.arrow(
                    old_pos.y,
                    old_pos.x,
                    state.agent.pos.y - old_pos.y,
                    state.agent.pos.x - old_pos.x,
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
    m = model.sample_model()
    fig, plot = plot_model(m)

    states, actions = engine.simulate(
        m,
        10,
        heuristic.generate_action,
    )

    anim = animate_states(
        fig,
        plot,
        states,
        actions,
        m.maze,
    )

    plt.show()
