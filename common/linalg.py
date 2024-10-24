import numpy as np
from astropy import units as u
from numpy import typing as npt
from poliastro import bodies


def sphere_line_intersection(
    sphere_center: npt.NDArray,
    line_point: npt.NDArray,
    line_direction: npt.NDArray,
    sphere_radius: float = bodies.Earth.R.to(u.km).value,
) -> npt.NDArray | None:
    """Check if a line intersects with a sphere.

    Args:
        sphere_center: Coordinates of the sphere center.
        line_point: Point on the line.
        line_direction: Direction of the line.
        sphere_radius: Radius of the sphere.

    Returns:
        array or None: Intersection point(s) or None if no intersection.
    """
    # Unpack sphere parameters
    x0, y0, z0 = sphere_center
    r = sphere_radius

    # Unpack line parameters
    x1, y1, z1 = line_point
    dx, dy, dz = line_direction

    # Compute coefficients for the quadratic equation
    a = dx**2 + dy**2 + dz**2
    b = 2 * (dx * (x1 - x0) + dy * (y1 - y0) + dz * (z1 - z0))
    c = (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2 - r**2

    # Compute discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return None
    elif discriminant == 0:
        # One intersection
        t = -b / (2 * a)
        intersection_point = np.array([x1 + t * dx, y1 + t * dy, z1 + t * dz])
        return intersection_point
    else:
        # Two intersections
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        intersection_point1 = np.array([x1 + t1 * dx, y1 + t1 * dy, z1 + t1 * dz])
        intersection_point2 = np.array([x1 + t2 * dx, y1 + t2 * dy, z1 + t2 * dz])

        # Calculate distances
        dist1 = np.linalg.norm(intersection_point1 - line_point)
        dist2 = np.linalg.norm(intersection_point2 - line_point)

        if dist1 < dist2:
            return intersection_point1
        else:
            return intersection_point2


def intersects_earth(pos_1: npt.NDArray, pos_2: npt.NDArray) -> bool:
    """Check if the Earth is blocking the two positions using line-sphere intersection.

    Args:
        pos_1: Position of the first point.
        pos_2: Position of the second point.

    Returns:
        bool: True if the Earth is blocking the line of sight, False otherwise.
    """
    # Make a line between the two points
    line = pos_2 - pos_1

    # Check if there is an intersection with the Earth
    if sphere_line_intersection(np.array((0, 0, 0)), pos_1, line) is not None:
        return True

    # If there is no intersection, return False
    return False
