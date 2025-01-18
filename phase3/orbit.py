import dataclasses

from astropy import units as u
from numpy import typing as npt
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

    @staticmethod
    def from_vectors(r: npt.NDArray, v: npt.NDArray) -> twobody.Orbit:
        return twobody.Orbit.from_vectors(bodies.Earth, r, v)

    @classmethod
    def from_sim_config(cls, config: sim_config.Orbit) -> 'Orbit':
        return cls(
            altitude=u.Quantity(config.altitude, u.km),
            ecc=u.Quantity(config.ecc, u.dimensionless_unscaled),
            inc=u.Quantity(config.inc, u.deg),
            raan=u.Quantity(config.raan, u.deg),
            argp=u.Quantity(config.argp, u.deg),
            nu=u.Quantity(config.nu, u.deg),
        )
