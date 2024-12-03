"""Interfaces & utilities for collected data from the simulation."""

import dataclasses
import enum
from collections import defaultdict
from typing import Sequence

import numpy as np
from numpy import typing as npt

# Target -> Satellite -> # of numbers
TargetAggregator = dict[int, dict[str, int]]


class GsDataType(enum.Enum):
    COVARIANCE_INTERSECTION = enum.auto()
    MEASUREMENT = enum.auto()


@dataclasses.dataclass
class State:
    time: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float


@dataclasses.dataclass
class Measurement:
    target_id: int
    time: float
    alpha: float
    beta: float
    sat_name: str  # name of the satellite that took the measurement
    sat_state: npt.NDArray  # 6x1 of the [x,y,z,vx,vy,vz]
    R_mat: npt.NDArray  # sensor noise matrix


@dataclasses.dataclass
class Estimate:
    target_id: int
    time: float
    estimate: npt.NDArray
    covariance: npt.NDArray
    innovation: npt.NDArray
    innovation_covariance: npt.NDArray
    track_uncertainty: (
        float  # The trace uncertainty of the covariance matrix in the track
    )


@dataclasses.dataclass
class Transmission:
    sender: str  # sat/gs name (point to point)
    receiver: str  # sat/gs name (point to point)
    source: str  # sat/gs name (path, start)
    destination: str  # sat/gs name (path, end)
    size: float  # number of bytes (estimate)
    time: float  # time of transmission


@dataclasses.dataclass
class MeasurementTransmission(Transmission):
    measurements: list[Measurement]


@dataclasses.dataclass
class EstimateTransmission(Transmission):
    estimates: list[Estimate]


@dataclasses.dataclass
class BountyTransmission(Transmission):
    target_id: int


@dataclasses.dataclass
class GsTransmission:
    target_id: int
    # Sending satellite
    sender: str
    # Receiving ground station
    receiver: str
    time: float


@dataclasses.dataclass
class GsEstimateTransmission(GsTransmission):
    estimate: npt.NDArray
    covariance: npt.NDArray


@dataclasses.dataclass
class GsMeasurementTransmission(GsTransmission):
    measurement: npt.NDArray | None


def aggregate_transmissions(
    transmissions: Sequence[Transmission], sat_names: list[str]
) -> tuple[dict[str, int], TargetAggregator, TargetAggregator]:
    """Aggregate the transmissions by target ID and satellite name.

    Args:
        transmissions: List of transmissions to aggregate.
        sat_names: List of satellite names.

    Returns:
        Aggregated transmissions of `prev_data`, `target_sent_data`, and `target_rec_data`.
    """

    # Save previous data, to stack the bars
    # prev_data = np.zeros(len(satNames))
    # make prev_data a dictionary
    prev_data = {sat: 0 for sat in sat_names}

    # Per-target transmissions
    target_sent_data: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    target_rec_data: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for transmission in transmissions:
        target_sent_data[transmission.target_id][
            transmission.sender
        ] += transmission.size
        target_rec_data[transmission.target_id][
            transmission.receiver
        ] += transmission.size

    # Ensure parity between sent and received data keys
    for target_id in target_sent_data.keys() | target_rec_data.keys():
        for sat in target_sent_data[target_id]:
            target_rec_data[target_id][sat] += 0
        for sat in target_rec_data[target_id]:
            target_sent_data[target_id][sat] += 0

        # Order the data the same way, according to "sats" variable
        target_sent_data[target_id] = dict(
            sorted(
                target_sent_data[target_id].items(),
                key=lambda item: sat_names.index(item[0]),
            )
        )
        target_rec_data[target_id] = dict(
            sorted(
                target_rec_data[target_id].items(),
                key=lambda item: sat_names.index(item[0]),
            )
        )

        # Add the rec_data values to the prev_data
        for sat in target_rec_data[target_id].keys():
            prev_data[sat] += target_rec_data[target_id][sat]

    return prev_data, target_sent_data, target_rec_data
