from astropy import units as u

from phase3 import orbit


def walker_delta(
    total_sats: int,
    num_planes: int,
    inc_deg: float,
    altitude_km: float,
    phasing_deg: float,
) -> list[orbit.Orbit]:
    """Generate a Walker Delta constellation.

    Args:
        total_sats: Total number of satellites in the constellation.
        num_planes: Number of planes.
        inc_deg: Inclination of the planes in degrees.
        altitude_km: Altitude of the satellites in kilometers.
        phasing: Phasing between planes in degrees.

    Returns:
        List of Orbit objects representing the satellites in the constellation.
    """
    assert total_sats % num_planes == 0

    # Calculate the separation angle between satellites in a plane
    sep = 360 / (total_sats // num_planes)
    # Calculate the separation angle between planes
    plane_sep = 360 / num_planes

    orbits = []

    for plane in range(num_planes):
        for sat in range(total_sats // num_planes):
            nu = sep * sat + plane * phasing_deg
            raan = plane_sep * plane
            orbits.append(
                orbit.Orbit(
                    altitude=u.Quantity(altitude_km, u.km),
                    ecc=u.Quantity(0, u.dimensionless_unscaled),
                    inc=u.Quantity(inc_deg, u.deg),
                    raan=u.Quantity(raan, u.deg),
                    argp=u.Quantity(0, u.deg),
                    nu=u.Quantity(nu, u.deg),
                )
            )

    return orbits
