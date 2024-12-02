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

    # Show live plots
    show_live: bool

    show_comms: bool

    # Plots to generate
    plots: list[PlotType]

    # GIFs to generate
    gifs: list[GifType]


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

    # Targets
    targets: dict[str, Target]

    # Sensors
    sensors: dict[str, Sensor]

    # Satellites
    sensing_satellites: dict[str, Satellite]
    fusion_satellites: dict[str, Satellite]

    # Commanders' Intents
    commanders_intent: util.CommandersIndent

    # Plan horizon
    plan_horizon_m: int

    # Ground Stations
    ground_stations: dict[str, GroundStation]

    def merge_overrides(
        self,
        sim_duration_m: int | None = None,
        output_prefix: str | None = None,
        show_live: bool | None = None,
        plots: list[PlotType] | None = None,
        gifs: list[GifType] | None = None,
    ) -> 'SimConfig':
        return SimConfig(
            sim_duration_m=sim_duration_m or self.sim_duration_m,
            sim_time_step_m=self.sim_time_step_m,
            plot=PlotConfig(
                output_prefix=output_prefix or self.plot.output_prefix,
                show_live=show_live or self.plot.show_live,
                show_comms=self.plot.show_comms,
                plots=plots or self.plot.plots,
                gifs=gifs or self.plot.gifs,
            ),
            comms=self.comms,
            estimator=self.estimator,
            targets=self.targets,
            sensors=self.sensors,
            sensing_satellites=self.sensing_satellites,
            fusion_satellites=self.fusion_satellites,
            commanders_intent=self.commanders_intent,
            ground_stations=self.ground_stations,
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
