import dataclasses
import enum
import pathlib
from typing import cast

import marshmallow_dataclass
import yaml

from common import path_utils
from phase3 import util


class GifType(enum.Enum):
    SATELLITE_SIMULATION = enum.auto()
    UNCERTAINTY_ELLIPSE = enum.auto()
    DYNAMIC_COMMS = enum.auto()


class PlotType(enum.Enum):
    ESTIMATION = enum.auto()
    COMMUNICATION = enum.auto()
    ET_NETWORK = enum.auto()
    UNCERTAINTY_ELLIPSES = enum.auto()
    GROUND_STATION_RESULTS = enum.auto()


class Estimators(enum.Enum):
    CENTRAL = enum.auto()
    LOCAL = enum.auto()
    COVARIANCE_INTERSECTION = enum.auto()
    EVENT_TRIGGERED = enum.auto()
    FEDERATED = enum.auto()


@dataclasses.dataclass
class PlotConfig:
    # Prefix for output files
    output_prefix: str

    # Show the commmunications sent between nodes
    show_comms: bool

    # Only shows the fusion sats, all sensing are dimmed
    show_only_fusion: bool


@dataclasses.dataclass
class CommsConfig:
    max_bandwidth: int
    max_neighbors: int
    max_range: int
    min_range: int


@dataclasses.dataclass
class Target:
    target_id: int
    coords: tuple[float, float, float]
    heading: int
    speed: int
    uncertainty: tuple[float, float, float, float, float]
    color: str


@dataclasses.dataclass
class Sensor:
    fov: int
    bearings_error: tuple[float, float]


@dataclasses.dataclass
class Orbit:
    altitude: float
    ecc: float
    inc: float
    raan: float
    argp: float
    nu: float


@dataclasses.dataclass
class Satellite:
    sensor: str | None
    orbit: Orbit
    color: str


@dataclasses.dataclass
class GroundStation:
    lat: float
    lon: float
    fov: int
    comms_range: int
    color: str


@dataclasses.dataclass
class RaidRegion:
    center: tuple[float, float, float]
    extent: tuple[float, float, float]
    initial_targs: int
    spawn_rate: float
    color: str
    priority: int


@dataclasses.dataclass
class SimConfig:
    # Simulation duration in minutes
    sim_duration_m: int

    sim_time_step_m: float

    # Plot configuration
    plot: PlotConfig

    # Comms network
    comms: CommsConfig

    # Estimators
    estimator: Estimators

    # Raid Regions
    raids: dict[str, RaidRegion]

    # Sensors
    sensors: dict[str, Sensor]

    # Satellites
    sensing_satellites: dict[str, Satellite]
    fusion_satellites: dict[str, Satellite]

    # Commanders' Intents
    commanders_intent: util.CommandersIndent

    # Planning horizon
    plan_horizon_m: float

    # Flag for if to do EKFs or just use perfect data
    do_ekfs: bool

    # Exchange rate
    exchange_rate: float

    # Ground Stations
    ground_stations: dict[str, GroundStation]

    def merge_overrides(
        self,
        sim_duration_m: int | None = None,
        output_prefix: str | None = None,
    ) -> 'SimConfig':
        return SimConfig(
            sim_duration_m=sim_duration_m or self.sim_duration_m,
            sim_time_step_m=self.sim_time_step_m,
            plot=PlotConfig(
                output_prefix=output_prefix or self.plot.output_prefix,
                show_comms=self.plot.show_comms,
                show_only_fusion=self.plot.show_only_fusion,
            ),
            comms=self.comms,
            estimator=self.estimator,
            raids=self.raids,
            sensors=self.sensors,
            sensing_satellites=self.sensing_satellites,
            fusion_satellites=self.fusion_satellites,
            commanders_intent=self.commanders_intent,
            ground_stations=self.ground_stations,
            plan_horizon_m=self.plan_horizon_m,
            do_ekfs=self.do_ekfs,
            exchange_rate=self.exchange_rate,
        )


SimConfigSchema = marshmallow_dataclass.class_schema(SimConfig)


def load_sim_config(file: pathlib.Path) -> SimConfig:
    schema = SimConfigSchema()
    return cast(SimConfig, schema.load(yaml.safe_load(file.read_text())))


def load_default_sim_config() -> SimConfig:
    return load_sim_config(path_utils.SCENARIOS / 'default.yaml')


def save_sim_config(config: SimConfig, file: pathlib.Path) -> None:
    schema = SimConfigSchema()
    contents = yaml.dump(schema.dump(config))
    # Purge the !!python/tuple tag
    contents = contents.replace(' !!python/tuple', '')
    file.write_text(contents)
