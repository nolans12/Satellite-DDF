import dataclasses

from astropy import units as u
from poliastro import bodies
from poliastro import twobody

from phase3 import sim_config


@dataclasses.dataclass
class Orbit:
    altitude: u.Quantity[u.km]
    ecc: u.Quantity[u.dimensionless_unscaled]
    inc: u.Quantity[u.deg]
    raan: u.Quantity[u.deg]
    argp: u.Quantity[u.deg]
    nu: u.Quantity[u.deg]

    def to_poliastro(self) -> twobody.Orbit:
        return twobody.Orbit.from_classical(
            bodies.Earth,
            bodies.Earth.R.to(u.km) + self.altitude,
            self.ecc,
            self.inc,
            self.raan,
            self.argp,
            self.nu,
        )

    @classmethod
    def from_sim_config(cls, config: sim_config.Orbit) -> 'Orbit':
        return cls(
            altitude=config.altitude * u.km,
            ecc=config.ecc * u.dimensionless_unscaled,
            inc=config.inc * u.deg,
            raan=config.raan * u.deg,
            argp=config.argp * u.deg,
            nu=config.nu * u.deg,
        )
