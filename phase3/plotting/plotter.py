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
from phase3 import raidRegion
from phase3 import satellite
from phase3 import sim_config
from phase3 import target


class Plotter:
    def __init__(
        self,
        config: sim_config.PlotConfig,
        raid_regions: Sequence[raidRegion.RaidRegion],
    ):
        self._config = config
        self._raid_regions = raid_regions
        self._imgs = {
            raid._name: [] for raid in raid_regions
        }  # Initialize image lists per region

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

        # Create a separate figure and axes for each region
        self._figs = {}
        self._axes = {}
        for raid in raid_regions:
            fig = plt.figure(figsize=(10, 8))
            ax = cast(axes3d.Axes3D, fig.add_subplot(111, projection='3d'))

            # Get the elevation and azimuth angles such that the center of the raid region is in the center of the plot
            elev = raid._center[0]
            azim = raid._center[1]
            ax.view_init(elev=elev, azim=azim)

            # Label the axes and set title
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_zlabel('Z (km)')
            ax.set_title(f'500 Target Scenario, {raid._name} (Planning Horizon: 1 min)')

            self._figs[raid._name] = fig
            self._axes[raid._name] = ax

            # Set the limits as alot so dont have clipping issues!
            ax.set_xlim(-10000, 10000)
            ax.set_ylim(-10000, 10000)
            ax.set_zlim(-10000, 10000)

    def plot(
        self,
        time: float,
        sats: Sequence[satellite.Satellite],
        targets: Sequence[target.Target],
        raid_regions: Sequence[raidRegion.RaidRegion],
        comms: comms.Comms,
    ) -> None:
        """
        Plot the current state of the environment for each region.
        """
        for raid in raid_regions:
            ax = self._axes[raid._name]
            fig = self._figs[raid._name]

            self._reset_plot(ax)
            self._plot_earth(ax)
            self._plot_raid_regions(raid_regions, ax)
            if self._config.show_only_fusion:
                self._plot_satellite_only_fusion(sats, ax, raid)
            else:
                self._plot_satellites(sats, ax)
            self._plot_targets(targets, ax, raid)
            self._plot_communication(comms, sats, ax)
            self._plot_legend_time(time, ax)
            self._export_image(raid._name, fig, ax)

            pause_step = 0.001
            plt.pause(pause_step)
            plt.draw()

            # Add the following lines to create a legend
            ax.plot([], [], 'go', label='Fusion Layer')  # Green satellites
            ax.plot([], [], 'bo', label='Sensing Layer')  # Blue satellites
            ax.plot(
                [], [], 'k--', label='Measurement Transmissions'
            )  # Dashed red lines

            for r in raid_regions:
                ax.plot([], [], 'o', color=r._color, label=r._name, markersize=10)

            ax.legend(loc='upper right')  # Display the legend in top right corner

    def _reset_plot(self, ax: axes3d.Axes3D) -> None:
        """
        Reset the plot by removing all lines, collections, and texts for a specific axis.
        """
        for line in ax.lines:
            line.remove()
        for collection in ax.collections:
            collection.remove()
        for text in ax.texts:
            text.remove()

    def _export_image(
        self, region_name: str, fig: plt.Figure, ax: axes3d.Axes3D
    ) -> None:
        ios = io.BytesIO()
        fig.savefig(ios, format='raw')
        ios.seek(0)
        w, h = fig.canvas.get_width_height()
        img = np.reshape(
            np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4)
        )[:, :, 0:4]
        self._imgs[region_name].append(img)

    def render_gifs(
        self,
        out_dir: pathlib.Path = path_utils.PHASE_3 / 'gifs',
        fps: int = 5,
    ):
        """
        Renders and saves GIFs for each region based on the specified file type.

        Parameters:
        - out_dir: The directory path where the GIF files will be saved. Defaults to the directory of the script.
        - fps: Frames per second for the GIF. Defaults to 5.
        """
        # Make sure the output directory exists
        out_dir.mkdir(exist_ok=True)

        frame_duration = 1000 / fps  # in ms
        save_name = self._config.output_prefix

        for region_name, images in self._imgs.items():
            file = out_dir / f'{save_name}_{region_name}_satellite_sim.gif'
            with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                for img in images:
                    writer.append_data(img)  # type: ignore

    # Ensure that other plotting methods also accept the ax parameter
    def _plot_satellites(
        self, sats: Sequence[satellite.Satellite], ax: axes3d.Axes3D
    ) -> None:
        """
        Plot all satellites, with different visibility based on their importance.
        """
        for sat in sats:

            # Determine importance of satellite
            is_important = False
            if isinstance(sat, satellite.FusionSatellite):
                is_important = bool(sat.custody and any(sat.custody.values()))
            elif isinstance(sat, satellite.SensingSatellite):
                is_important = bool(sat.bounty)

            # Plot the current xyz location of the satellite
            x, y, z = sat.orbit.r.value

            # Use different alpha values based on importance
            alpha = 1.0 if is_important else 0.1

            ax.scatter(x, y, z, s=40, color=sat.color, alpha=alpha)

            # Only plot sensor projection for important satellites
            if is_important:
                points = sat.get_projection_box()
                if points is None:
                    continue

                box = np.array([points[1], points[4], points[2], points[3], points[1]])
                ax.add_collection3d(
                    art3d.Poly3DCollection(
                        [box],
                        facecolors=sat.color,
                        linewidths=1,
                        edgecolors=sat.color,
                        alpha=0.05,
                        zorder=10000,
                    )
                )

    def _plot_satellite_only_fusion(
        self,
        sats: Sequence[satellite.Satellite],
        ax: axes3d.Axes3D,
        current_raid: raidRegion.RaidRegion,
    ) -> None:
        for sat in sats:

            is_important = False
            if isinstance(sat, satellite.FusionSatellite):
                is_important = bool(sat.custody and any(sat.custody.values()))

            if not is_important:
                # Plot a dimmed version of the satellite
                x, y, z = sat.orbit.r.value
                ax.scatter(x, y, z, s=40, color=sat.color, alpha=0.1)
                continue

            # Figure out which raid region this sat has custody of
            colors = []
            for raid in self._raid_regions:
                for target_name, custody in sat.custody.items():
                    if custody and raid._name in target_name:
                        colors.append(raid._color)

            x, y, z = sat.orbit.r.value
            if len(colors) == 0:
                print("ERROR: No color found for fusion satellite")
                exit(1)
            if current_raid._color in colors:  # If have a multi
                ax.scatter(
                    x, y, z, s=40, color=current_raid._color, alpha=1, zorder=1000000
                )
            else:
                ax.scatter(x, y, z, s=40, color=colors[0], alpha=0.5, zorder=1000000)

    def _plot_targets(
        self,
        targets: Sequence[target.Target],
        ax: axes3d.Axes3D,
        current_raid: raidRegion.RaidRegion,
    ) -> None:
        """
        Plot the current state of each target, including their positions and velocities.
        """
        for targ in targets:
            x, y, z = targ.pos
            vx, vy, vz = targ.vel
            mag = np.linalg.norm([vx, vy, vz])
            if mag > 0:
                vx, vy, vz = vx / mag, vy / mag, vz / mag

            if current_raid._color == targ.color:
                alpha = 1
            else:
                alpha = 0.5
            ax.scatter(x, y, z, s=5, marker='x', color=targ.color, alpha=alpha)

    def _plot_raid_regions(
        self,
        raid_regions: Sequence[raidRegion.RaidRegion],
        ax: axes3d.Axes3D,
    ) -> None:
        """
        Plot the raid regions as boxes showing their extent.
        """
        for raid in raid_regions:
            # Get the corners in lat/lon/alt coordinates
            corners = []
            center = np.array(raid._center)
            extent = np.array(raid._extent)

            # Generate all 4 corners by adding/subtracting lat/lon extent from center
            r = center[2] + 6378  # Fixed altitude + Earth radius
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    corner = center + np.array([dx, dy, 0]) * extent
                    # Convert from lat/lon/alt to xyz
                    lat = np.deg2rad(corner[0])
                    lon = np.deg2rad(corner[1])
                    x = r * np.cos(lat) * np.cos(lon)
                    y = r * np.cos(lat) * np.sin(lon)
                    z = r * np.sin(lat)
                    # Add both top and bottom corners at same altitude
                    corners.append([x, y, z])
                    corners.append([x, y, z])

            corners = np.array(corners)

            # Create the box faces
            faces = [
                [corners[0], corners[1], corners[3], corners[2]],  # Bottom
                [corners[4], corners[5], corners[7], corners[6]],  # Top
                [corners[0], corners[1], corners[5], corners[4]],  # Front
                [corners[2], corners[3], corners[7], corners[6]],  # Back
                [corners[0], corners[2], corners[6], corners[4]],  # Left
                [corners[1], corners[3], corners[7], corners[5]],  # Right
            ]

            # Plot the box
            ax.add_collection3d(
                art3d.Poly3DCollection(
                    faces,
                    facecolors=raid._color,
                    linewidths=1,
                    edgecolors=raid._color,
                    alpha=0.05,
                )
            )

    def _plot_earth(self, ax: axes3d.Axes3D) -> None:
        """
        Plot the Earth's surface.
        """
        ax.plot_surface(
            self._x_earth * 0.90,
            self._y_earth * 0.90,
            self._z_earth * 0.90,
            color='white',
            alpha=1,
            zorder=1,
        )

    def _plot_communication(
        self, comms: comms.Comms, sats: Sequence[satellite.Satellite], ax: axes3d.Axes3D
    ) -> None:
        """
        Plot the communication structure between satellites.
        """
        if not self._config.show_comms:
            return

        for edge in comms.G.edges:
            if comms.G.edges[edge]['active'] == "":
                continue
            sat1_name = edge[0]
            sat2_name = edge[1]
            sat1 = next((sat for sat in sats if sat.name == sat1_name), None)
            sat2 = next((sat for sat in sats if sat.name == sat2_name), None)
            if sat1 is None or sat2 is None:
                continue
            x1, y1, z1 = sat1.orbit.r.value
            x2, y2, z2 = sat2.orbit.r.value
            if (
                comms.G.edges[edge]['active'] == "Track Handoff"
            ):  # TODO: make it a list, not just a singlar string
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    [z1, z2],
                    color=(0.5, 0.0, 0.5),  # Purple color
                    linestyle='dashed',
                    linewidth=2,
                    zorder=2,
                )
                plt.pause(0.5)  # Pause for 0.5 seconds
            elif comms.G.edges[edge]['active'] == "Measurement":
                # Calculate distance between satellites
                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                # Create points spaced by 100 units
                num_points = int(dist // 100) + 1
                if num_points > 1:
                    x_points = x1 + (dx / dist) * np.arange(0, dist, 100)
                    y_points = y1 + (dy / dist) * np.arange(0, dist, 100)
                    z_points = z1 + (dz / dist) * np.arange(0, dist, 100)

                    ax.scatter(
                        x_points,
                        y_points,
                        z_points,
                        marker='.',
                        color="black",  # Black color
                        s=1,
                        zorder=1000,
                        alpha=0.5,
                    )

    def _plot_legend_time(self, time: float, ax: axes3d.Axes3D) -> None:
        """
        Plot the legend and the current simulation time.
        """
        ax.text2D(0.05, 0.95, f'Time: {time:.2f}', transform=ax.transAxes)
