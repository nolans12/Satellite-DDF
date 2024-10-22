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
class Transmission:
    target_id: int
    sender: str
    receiver: str
    time: float
    size: int


@dataclasses.dataclass
class MeasurementTransmission(Transmission):
    alpha: float
    beta: float

    @property
    def has_alpha_beta(self) -> bool:
        return not np.isnan(self.alpha) and not np.isnan(self.beta)


@dataclasses.dataclass
class EstimateTransmission(Transmission):
    estimate: npt.NDArray
    covariance: npt.NDArray

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
