import dataclasses

from astropy import units as u


@dataclasses.dataclass
class Orbit:
    altitude: u.Quantity[u.km]
    ecc: u.Quantity[u.dimensionless_unscaled]
    inc: u.Quantity[u.deg]
    raan: u.Quantity[u.deg]
    argp: u.Quantity[u.deg]
    nu: u.Quantity[u.deg]
