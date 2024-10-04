import dataclasses
import enum

import marshmallow_dataclass

from phase3 import util


class GifType(enum.Enum):
    SATELLITE_SIMULATION = 'satellite_simulation'
    UNCERTAINTY_ELLIPSE = 'uncertainty_ellipse'
    DYNAMIC_COMMS = 'dynamic_comms'


@dataclasses.dataclass
class PlotConfig:
    # Prefix for output files
    output_prefix: str

    # Show live plots
    show_env: bool

    # Enable plot types
    plot_estimation: bool
    plot_communication: bool
    plot_et_network: bool
    plot_uncertainty_ellipses: bool
    plot_groundStation_results: bool

    # GIFs to generate
    gifs: list[GifType]


@dataclasses.dataclass
class CommsConfig:
    max_bandwidth: int
    max_neighbors: int
    max_range: int
    min_range: int
    display_struct: bool


@dataclasses.dataclass
class EstimatorConfig:
    central: bool
    local: bool
    ci: bool
    et: bool


@dataclasses.dataclass
class Target:
    tq_req: int
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
class Satellite:
    sensor: str
    altitude: float
    ecc: float
    inc: float
    raan: float
    argp: float
    nu: float
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
    estimators: EstimatorConfig

    # Targets
    targets: dict[str, Target]

    # Sensors
    sensors: dict[str, Sensor]

    # Satellites
    satellites: dict[str, Satellite]

    # Commanders' Intents
    commanders_intent: util.CommandersIndent

    # Ground Stations
    ground_stations: dict[str, GroundStation]

    def merge_overrides(
        self,
        sim_duration_m: int | None = None,
        output_prefix: str | None = None,
        show_env: bool | None = None,
        plot_estimation: bool | None = None,
        plot_communication: bool | None = None,
        plot_et_network: bool | None = None,
        plot_uncertainty_ellipses: bool | None = None,
        plot_groundStation_results: bool | None = None,
        gifs: list[GifType] | None = None,
    ) -> 'SimConfig':
        return SimConfig(
            sim_duration_m=sim_duration_m or self.sim_duration_m,
            sim_time_step_m=self.sim_time_step_m,
            plot=PlotConfig(
                output_prefix=output_prefix or self.plot.output_prefix,
                show_env=show_env or self.plot.show_env,
                plot_estimation=plot_estimation or self.plot.plot_estimation,
                plot_communication=plot_communication or self.plot.plot_communication,
                plot_et_network=plot_et_network or self.plot.plot_et_network,
                plot_uncertainty_ellipses=plot_uncertainty_ellipses
                or self.plot.plot_uncertainty_ellipses,
                plot_groundStation_results=plot_groundStation_results
                or self.plot.plot_groundStation_results,
                gifs=gifs or self.plot.gifs,
            ),
            comms=self.comms,
            estimators=self.estimators,
            targets=self.targets,
            sensors=self.sensors,
            satellites=self.satellites,
            commanders_intent=self.commanders_intent,
            ground_stations=self.ground_stations,
        )


SimConfigSchema = marshmallow_dataclass.class_schema(SimConfig)
