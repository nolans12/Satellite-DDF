import numpy as np

from canonical import comms
from canonical import orbit
from canonical import satellite
from canonical import sensor
from canonical import sim_config
from canonical import target


class Environment:
    def __init__(self, *args, network: comms.Comms, **kwargs):
        self.comms = network

    @classmethod
    def from_config(cls, cfg: sim_config.SimConfig) -> 'Environment':
        targs = [
            target.Target(
                name=name,
                target_id=t.target_id,
                coords=np.array(t.coords),
                heading=t.heading,
                speed=t.speed,
                uncertainty=np.array(t.uncertainty),
                color=t.color,
            )
            for name, t in cfg.targets.items()
        ]

        sensing_sats = {
            name: satellite.SensingSatellite(
                name=name,
                sensor=sensor.Sensor(
                    name=s.sensor,
                    fov=cfg.sensors[s.sensor].fov,
                    bearingsError=np.array(cfg.sensors[s.sensor].bearings_error),
                ),
                orbit=orbit.Orbit.from_sim_config(s.orbit),
                color=s.color,
            )
            for name, s in cfg.sensing_satellites.items()
        }

        fusion_sats = {
            name: satellite.FusionSatellite(
                name=name,
                orbit=orbit.Orbit.from_sim_config(s.orbit),
                local_estimator=None,
                color=s.color,
            )
            for name, s in cfg.sensing_satellites.items()
        }

        # Define the communication network:
        comms_network = comms.Comms(
            list(sensing_sats.values()),
            list(fusion_sats.values()),
            [],
            config=cfg.comms,
        )

        # Create and return an environment instance:
        return cls(
            list(sensing_sats.values()),
            list(fusion_sats.values()),
            targs,
            cfg.estimators,
            network=comms_network,
        )
