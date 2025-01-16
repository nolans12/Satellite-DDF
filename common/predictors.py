import numpy as np
import numpy.typing as npt


def state_transition(estPrior: npt.NDArray, dt: float) -> npt.ArrayLike:
    """
    State Transition Function: Converts ECI Cartesian state to spherical coordinates,
    propagates state over time using Runge-Kutta 4th order integration, and converts
    back to Cartesian. Inputs current state and time step, returns next state.

    Parameters:
    - estPrior: Current state in Cartesian coordinates [x, vx, y, vy, z, vz].
    - dt: Time step for integration.

    Returns:
    - Next state in Cartesian coordinates [x, vx, y, vy, z, vz].
    """

    if dt == 0:
        return estPrior
    x, vx, y, vy, z, vz = estPrior

    # Convert to Spherical Coordinates
    range = np.sqrt(x**2 + y**2 + z**2)
    elevation = np.arcsin(z / range)
    azimuth = np.arctan2(y, x)

    # Calculate the Range Rate
    rangeRate = (x * vx + y * vy + z * vz) / range

    # Calculate Elevation Rate
    elevationRate = -(z * (vx * x + vy * y) - (x**2 + y**2) * vz) / (
        (x**2 + y**2 + z**2) * np.sqrt(x**2 + y**2)
    )

    # Calculate Azimuth Rate
    azimuthRate = (x * vy - y * vx) / (x**2 + y**2)

    # Previous Spherical State
    prev_spherical_state = np.array(
        [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]
    )

    # Define function to compute derivatives for Runge-Kutta method
    def derivatives(spherical_state: npt.ArrayLike) -> npt.ArrayLike:
        """
        Computes derivatives of spherical state variables.

        Parameters:
        - spherical_state (array-like): Spherical state variables [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate].

        Returns:
        - array-like: Derivatives [rangeRate, rangeAccel, elevationRate, elevationAccel, azimuthRate, azimuthAccel].
        """
        r, rdot, elevation, elevationRate, azimuth, azimuthRate = spherical_state

        rangeRate = rdot
        rangeAccel = 0  # No acceleration assumed in this simplified model
        elevationRate = elevationRate
        elevationAccel = 0  # No acceleration assumed in this simplified model
        azimuthRate = azimuthRate
        azimuthAccel = 0  # No acceleration assumed in this simplified model

        return np.array(
            [
                rangeRate,
                rangeAccel,
                elevationRate,
                elevationAccel,
                azimuthRate,
                azimuthAccel,
            ]
        )

    # Runge-Kutta 4th order integration
    k1 = derivatives(prev_spherical_state)
    k2 = derivatives(prev_spherical_state + 0.5 * dt * k1)
    k3 = derivatives(prev_spherical_state + 0.5 * dt * k2)
    k4 = derivatives(prev_spherical_state + dt * k3)

    next_spherical_state = prev_spherical_state + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )

    # Extract components from spherical state
    range = next_spherical_state[0]
    rangeRate = next_spherical_state[1]
    elevation = next_spherical_state[2]
    elevationRate = next_spherical_state[3]
    azimuth = next_spherical_state[4]
    azimuthRate = next_spherical_state[5]

    # Convert back to Cartesian coordinates
    x = range * np.cos(elevation) * np.cos(azimuth)
    y = range * np.cos(elevation) * np.sin(azimuth)
    z = range * np.sin(elevation)

    # Approximate velocities conversion (simplified version)
    vx = (
        rangeRate * np.cos(elevation) * np.cos(azimuth)
        - range * elevationRate * np.sin(elevation) * np.cos(azimuth)
        - range * azimuthRate * np.cos(elevation) * np.sin(azimuth)
    )

    vy = (
        rangeRate * np.cos(elevation) * np.sin(azimuth)
        - range * elevationRate * np.sin(elevation) * np.sin(azimuth)
        + range * azimuthRate * np.cos(elevation) * np.cos(azimuth)
    )

    vz = rangeRate * np.sin(elevation) + range * elevationRate * np.cos(elevation)

    # Return the next state in Cartesian coordinates
    return np.array([x, vx, y, vy, z, vz])
