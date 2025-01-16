import numpy as np
from numpy import typing as npt

from phase3 import target


class RaidRegion:
    def __init__(
        self,
        name: str,
        center: npt.NDArray,
        extent: npt.NDArray,
        initial_targs: int,
        spawn_rate: float,
        color: str,
        priority: int,
    ):
        """
        A raid region is a region of space where targets are initialized and can also appear.

        Args:
            center: The center of the raid region [lat, lon, alt].
            extent: The extent of the raid region +-[lat, lon, alt].
            initial_targs: The number of targets to initialize in the raid region.
            spawn_rate: The rate at which targets spawn in the raid region [targets/s].
        """
        self._name = name
        self._center = center
        self._extent = extent
        self._initial_targs = initial_targs
        self._spawn_rate = spawn_rate
        self._color = color
        self.targets = self._init_targs()
        self._priority = priority

    def _init_targs(self) -> list[target.Target]:
        # For the number of initial targets, we need to generate a random distribution
        # within the raid region.
        initial_targs_pos = np.random.uniform(
            low=self._center - self._extent,
            high=self._center + self._extent,
            size=(self._initial_targs, 3),
        )

        # Cap the low of altitude to be 0
        initial_targs_pos[:, 2] = np.maximum(initial_targs_pos[:, 2], 0)

        # Also randomly select heading and speed for the targets
        initial_targs_heading = np.random.uniform(
            low=0, high=360, size=self._initial_targs
        )
        initial_targs_speed = np.random.uniform(
            low=10, high=50, size=self._initial_targs
        )

        # Now, we need to generate the targets
        targets = [
            target.Target(
                target_id=f"{self._name}_Targ{i + 1}",
                coords=initial_targs_pos[i],
                heading=initial_targs_heading[i],
                speed=initial_targs_speed[i],
                color=self._color,
            )
            for i in range(self._initial_targs)
        ]

        return targets
