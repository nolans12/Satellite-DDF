import dataclasses
import enum

import marshmallow_dataclass


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

    # GIFs to generate
    gifs: list[GifType]


@dataclasses.dataclass
class SimConfig:
    # Simulation duration in minutes
    sim_duration_m: int

    # Plot configuration
    plot: PlotConfig

    def merge_overrides(
        self,
        sim_duration_m: int | None = None,
        output_prefix: str | None = None,
        show_env: bool | None = None,
        plot_estimation: bool | None = None,
        plot_communication: bool | None = None,
        plot_et_network: bool | None = None,
        plot_uncertainty_ellipses: bool | None = None,
        gifs: list[GifType] | None = None,
    ) -> 'SimConfig':
        return SimConfig(
            sim_duration_m=sim_duration_m or self.sim_duration_m,
            plot=PlotConfig(
                output_prefix=output_prefix or self.plot.output_prefix,
                show_env=show_env or self.plot.show_env,
                plot_estimation=plot_estimation or self.plot.plot_estimation,
                plot_communication=plot_communication or self.plot.plot_communication,
                plot_et_network=plot_et_network or self.plot.plot_et_network,
                plot_uncertainty_ellipses=plot_uncertainty_ellipses
                or self.plot.plot_uncertainty_ellipses,
                gifs=gifs or self.plot.gifs,
            ),
        )


SimConfigSchema = marshmallow_dataclass.class_schema(SimConfig)
