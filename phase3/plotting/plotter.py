import io
import pathlib
from typing import Sequence, cast

import imageio
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import axes3d
from numpy import typing as npt
from poliastro import bodies

from common import path_utils
from phase3 import comms
from phase3 import ground_station
from phase3 import satellite
from phase3 import sim_config
from phase3 import target


class Plotter:
    def __init__(self, config: sim_config.PlotConfig):
        self._config = config

        # Environment Plotting parameters
        self._fig = plt.figure(
            figsize=(10, 8)
        )  # I keep having errors here with matplot lib version, sometimes errors, some not, -nolan
        self._ax = cast(axes3d.Axes3D, self._fig.add_subplot(111, projection='3d'))

        # If you want to do clustered case:
        self._ax.set_xlim([2000, 8000])
        self._ax.set_ylim([-6000, 6000])
        self._ax.set_zlim([2000, 8000])
        self._ax.view_init(elev=30, azim=0)

        # auto scale the axis to be equal
        self._ax.set_box_aspect([0.5, 1, 0.5])

        # Label the axes and set title
        self._ax.set_xlabel('X (km)')
        self._ax.set_ylabel('Y (km)')
        self._ax.set_zlabel('Z (km)')
        self._ax.set_title('Satellite Orbit Visualization')

        # Earth parameters for plotting
        u_earth = np.linspace(0, 2 * np.pi, 100)
        v_earth = np.linspace(0, np.pi, 100)
        self._x_earth = bodies.Earth.R.to(u.km).value * np.outer(
            np.cos(u_earth), np.sin(v_earth)
        )
        self._y_earth = bodies.Earth.R.to(u.km).value * np.outer(
            np.sin(u_earth), np.sin(v_earth)
        )
        self._z_earth = bodies.Earth.R.to(u.km).value * np.outer(
            np.ones(np.size(u_earth)), np.cos(v_earth)
        )

        self._imgs: list[npt.NDArray] = []

    def plot(
        self,
        time: float,
        sats: Sequence[satellite.Satellite],
        targets: Sequence[target.Target],
        ground_stations: Sequence[ground_station.GroundStation],
        comms: comms.Comms,
    ) -> None:
        """
        Plot the current state of the environment.
        """
        self._reset_plot()
        self._plot_earth()
        self._plot_satellites(sats)
        self._plot_targets(targets)
        self._plot_communication(comms)
        self._plot_ground_stations(ground_stations)
        self._plot_legend_time(time)
        self._export_image()

        pause_step = 0.001
        if self._config.show_live:
            plt.pause(pause_step)
            plt.draw()

    def _reset_plot(self) -> None:
        """
        Reset the plot by removing all lines, collections, and texts.
        """
        for line in self._ax.lines:
            line.remove()
        for collection in self._ax.collections:
            collection.remove()
        for text in self._ax.texts:
            text.remove()

    def _plot_satellites(self, sats: Sequence[satellite.Satellite]) -> None:
        """
        Plot the current state of each satellite, including their positions and sensor projections.
        """
        for sat in sats:
            # Plot the current xyz location of the satellite
            x, y, z = sat.orbit.r.value
            # Cut the label of a satellite off before the first underscore
            satName = sat.name.split('.')[0]
            self._ax.scatter(x, y, z, s=40, color=sat.color, label=satName)

            # Plot the visible projection of the satellite sensor
            points = sat.get_projection_box()
            if points is None:
                continue
            self._ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],  # type: ignore
                color=sat.color,
                marker='x',
            )

            box = np.array([points[1], points[4], points[2], points[3], points[1]])
            self._ax.add_collection3d(
                art3d.Poly3DCollection(
                    [box],
                    facecolors=sat.color,
                    linewidths=1,
                    edgecolors=sat.color,
                    alpha=0.1,
                )
            )

    def _plot_targets(self, targets: Sequence[target.Target]) -> None:
        """
        Plot the current state of each target, including their positions and velocities.
        """
        for targ in targets:
            # Plot the current xyz location of the target
            x, y, z = targ.pos
            vx, vy, vz = targ.vel
            mag = np.linalg.norm([vx, vy, vz])
            if mag > 0:
                vx, vy, vz = vx / mag, vy / mag, vz / mag

            # do a standard scatter plot for the target
            self._ax.scatter(
                x, y, z, s=40, marker='x', color=targ.color, label=targ.name
            )

    def _plot_earth(self) -> None:
        """
        Plot the Earth's surface.
        """
        self._ax.plot_surface(
            self._x_earth, self._y_earth, self._z_earth, color='k', alpha=0.1
        )
        # ### ALSO USE IF YOU WANT EARTH TO NOT BE SEE THROUGH
        # self._ax.plot_surface(self._x_earth*0.9, self._y_earth*0.9, self._z_earth*0.9, color = 'white', alpha=1)

    def _plot_communication(self, comms: comms.Comms) -> None:
        """
        Plot the communication structure between satellites.
        """
        if not self._config.show_comms:
            return

        for edge in comms.G.edges:
            sat1 = edge[0]
            sat2 = edge[1]
            x1, y1, z1 = sat1.orbit.r.value
            x2, y2, z2 = sat2.orbit.r.value
            if comms.G.edges[edge]['active']:
                self._ax.plot(
                    [x1, x2],
                    [y1, y2],
                    [z1, z2],
                    color=(0.3, 1.0, 0.3),
                    linestyle='dashed',
                    linewidth=2,
                )
            else:
                self._ax.plot(
                    [x1, x2],
                    [y1, y2],
                    [z1, z2],
                    color='k',
                    linestyle='dashed',
                    linewidth=1,
                )

    def _plot_ground_stations(
        self, ground_stations: Sequence[ground_station.GroundStation]
    ) -> None:
        """
        Plot the ground stations.
        """
        for gs in ground_stations:
            x, y, z = gs.pos
            self._ax.scatter(x, y, z, s=40, marker='s', color=gs.color, label=gs.name)

    def _plot_legend_time(self, time: float) -> None:
        """
        Plot the legend and the current simulation time.
        """
        # handles, labels = self._ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # self._ax.legend(by_label.values(), by_label.keys())
        self._ax.text2D(0.05, 0.95, f'Time: {time:.2f}', transform=self._ax.transAxes)

    def _export_image(self) -> None:
        ios = io.BytesIO()
        self._fig.savefig(ios, format='raw')
        ios.seek(0)
        w, h = self._fig.canvas.get_width_height()
        img = np.reshape(
            np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4)
        )[:, :, 0:4]
        self._imgs.append(img)

    def render_gifs(
        self,
        out_dir: pathlib.Path = path_utils.PHASE_3 / 'gifs',
        fps: int = 5,
    ):
        """
        Renders and saves GIFs based on the specified file type.

        Parameters:
        - out_dir: The directory path where the GIF files will be saved. Defaults to the directory of the script.
        - fps: Frames per second for the GIF. Defaults to 10.
        """
        # Make sure the output directory exists
        out_dir.mkdir(exist_ok=True)

        frame_duration = 1000 / fps  # in ms
        save_name = self._config.output_prefix

        if sim_config.GifType.SATELLITE_SIMULATION in self._config.gifs:
            file = out_dir / f'{save_name}_satellite_sim.gif'
            with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                for img in self._imgs:
                    writer.append_data(img)  # type: ignore
