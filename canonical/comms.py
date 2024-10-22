import itertools
import logging
from typing import Generic, Protocol, TypeVar, overload

import networkx as nx
import numpy as np
from numpy import typing as npt

from canonical import collection
from canonical import sim_config
from common import dataclassframe
from common import linalg


class Agent(Protocol):
    """Satellite, ground station, or w/e."""

    name: str
    # [x, y, z]
    pos: npt.NDArray


S = TypeVar('S', bound=Agent)
F = TypeVar('F', bound=Agent)
G = TypeVar('G', bound=Agent)


class Comms(Generic[S, F, G]):
    """Communication network between satellites.

    The comms network is abstracted as a centrally-managed graph of satellites
    for simulation purposes. The graph is used to determine which satellites
    can communicate with each other, simulate the transmission of data between
    satellites, enforce bandwidth constraints, and track the amount of data
    sent and received between satellites.
    """

    def __init__(
        self,
        sensing_sats: list[S],
        fusion_sats: list[F],
        ground_stations: list[G],
        config: sim_config.CommsConfig,
    ):
        """Initialize the communications network.

        Using networkx, a python library, to create a graph of the satellites and their connections.

        Args:
            nodes: List of satellites.
            config: Configuration for the communication network.
        """
        self._config = config

        self.estimates = dataclassframe.DataClassFrame(
            clz=collection.EstimateTransmission
        )
        # Separate dataframe for estimated used by ET DDF
        self.used_estimates = dataclassframe.DataClassFrame(
            clz=collection.EstimateTransmission
        )

        self.measurements = dataclassframe.DataClassFrame(
            clz=collection.MeasurementTransmission
        )
        # Separate dataframe for measurements used by ET DDF
        self.used_measurements = dataclassframe.DataClassFrame(
            clz=collection.MeasurementTransmission
        )

        # Create a graph instance with the names as nodes
        self._nodes = {
            node.name: node for node in sensing_sats + fusion_sats + ground_stations
        }
        self.G = nx.DiGraph()
        # Add nodes with a dict for queued data (list of arrays)
        for node in self._nodes:
            self.G.add_node(node)
        self.update_edges()

    @overload
    def send_estimate(
        self,
        est_meas: npt.NDArray,
        cov_meas: npt.NDArray,
        target_id: int,
        time: float,
        *,
        sender: str,
        receiver: str,
    ) -> None: ...

    @overload
    def send_estimate(
        self,
        est_meas: npt.NDArray,
        cov_meas: npt.NDArray,
        target_id: int,
        time: float,
        *,
        path: list[str],
    ) -> None: ...

    def send_estimate(
        self,
        est_meas: npt.NDArray,
        cov_meas: npt.NDArray,
        target_id: int,
        time: float,
        *,
        sender: str | None = None,
        receiver: str | None = None,
        path: list[str] | None = None,
    ) -> None:
        """Send an estimate from one satellite to another

        Simulate "sending a measurement" by adding the estimate to the receiver's queued
        data on the commnication node. This way, at the end of the time step, the reciever
        satellite can just loop through the queued data and update its estimate using DDF
        algorithms on it.

        Args:
            sender: Satellite sending the estimate.
            receiver: Satellite receiving the estimate.
            est_meas: Estimate to send.
            cov_meas: Covariance estimate to send.
            target_id: ID of the target the estimate is from.
            time: Time the estimate was taken.
        """
        if path is not None and self._valid_path(path):
            for i in range(1, len(path)):
                self.send_estimate(
                    est_meas,
                    cov_meas,
                    target_id,
                    time,
                    sender=path[i - 1],
                    receiver=path[i],
                )
            return

        assert sender is not None and receiver is not None

        # Check if the receiver is in the sender's neighbors
        if not self.G.has_edge(sender, receiver):
            return

        # Before we decide if we want to send the estimate, make sure it wont
        # violate the bandwidth constraints
        if (
            self.G.edges[sender, receiver]['used_bandwidth']
            + est_meas.size * 2
            + cov_meas.size / 2
            > self.G.edges[sender, receiver]['max_bandwidth']
        ):
            logging.warning(
                f'Bandwidth exceeded between {sender} and {receiver} with current '
                f'bandwith of {self.G.edges[sender, receiver]["used_bandwidth"]} and '
                f'max bandwidth of {self.G.edges[sender, receiver]["max_bandwidth"]}'
            )
            return
        else:
            # Update the used bandwidth
            self.G.edges[sender, receiver]['used_bandwidth'] += (
                est_meas.size * 2 + cov_meas.size / 2
            )

        self.estimates.append(
            collection.EstimateTransmission(
                target_id=target_id,
                sender=sender,
                receiver=receiver,
                time=time,
                size=est_meas.size * 2 + cov_meas.size // 2,
                estimate=est_meas,
                covariance=cov_meas,
            )
        )

    def receive_estimates(
        self, receiver: str, time: float
    ) -> list[collection.EstimateTransmission]:
        """Receive all estimates for a node.

        Args:
            receiver: Node to receive estimates for.

        Returns:
            List of estimates for the node.
        """
        estimates = self.estimates.loc[
            (self.estimates['receiver'] == receiver) & (self.estimates['time'] == time)
        ]

        return self.estimates.to_dataclasses(estimates)

    @overload
    def send_measurements(
        self,
        alpha: float,
        beta: float,
        target_id: int,
        time: float,
        *,
        sender: str,
        receiver: str,
    ) -> None: ...

    @overload
    def send_measurements(
        self,
        alpha: float,
        beta: float,
        target_id: int,
        time: float,
        *,
        path: list[str],
    ) -> None: ...

    def send_measurements(
        self,
        alpha: float,
        beta: float,
        target_id: int,
        time: float,
        *,
        sender: str | None = None,
        receiver: str | None = None,
        path: list[str] | None = None,
    ) -> None:
        """Send a vector of measurements from one satellite to another.

        Share the measurement vector from the sender to the receiver by
        adding it to the receiver's measurement data on the communication node.

        Args:
            sender: Satellite sending the measurements.
            receiver: Satellite receiving the measurements.
            alpha: Alpha measurement to send.
            beta: Beta measurement to send.
            target_id: ID of the target the measurements are from.
            time: Time the measurements were taken.
        """
        if path is not None and self._valid_path(path):
            for i in range(1, len(path)):
                self.send_measurements(
                    alpha, beta, target_id, time, sender=path[i - 1], receiver=path[i]
                )
            return

        assert sender is not None and receiver is not None

        # Check if the receiver is in the sender's neighbors
        if not self.G.has_edge(sender, receiver):
            return

        measurement_size = 2 + 2  # 2 for the meas vector, 2 for the sensor noise
        if np.isnan(alpha):
            measurement_size -= 1

        if np.isnan(beta):
            measurement_size -= 1

        self.measurements.append(
            collection.MeasurementTransmission(
                target_id=target_id,
                sender=sender,
                receiver=receiver,
                time=time,
                size=measurement_size,
                alpha=alpha,
                beta=beta,
            )
        )

    def receive_measurements(
        self, receiver: str, time: float
    ) -> list[collection.MeasurementTransmission]:
        """Receive all measurements for a node.

        Args:
            receiver: Node to receive measurements for.

        Returns:
            List of measurements for the node.
        """
        measurements = self.measurements.loc[
            (self.measurements['receiver'] == receiver)
            & (self.measurements['time'] == time)
        ]

        return self.measurements.to_dataclasses(measurements)

    def get_neighbors(self, node: str) -> list[str]:
        """Get the neighbors of a node.

        Args:
            node: Satellite to get the neighbors of.

        Returns:
            List of names of neighbors of the satellite.
        """
        return list(self.G.neighbors(node))

    def get_path(self, node1: str, node2: str) -> list[str] | None:
        """Get the shortest path between two nodes.

        Args:
            node1: Starting node.
            node2: Ending node.

        Returns:
            Shortest path between the two nodes.
        """
        try:
            path: list[str] = nx.shortest_path(self.G, node1, node2)  # type: ignore
            return [self._nodes[node].name for node in path]
        except nx.NetworkXNoPath:
            logging.warning(
                f'No path between {node1} and {node2} in the communication network.'
            )
            return None

    def _valid_path(self, path: list[str]) -> bool:
        """Check if a path is valid."""
        for i in range(1, len(path)):
            if not self.G.has_edge(path[i - 1], path[i]):
                return False
        return True

    def update_edges(self) -> None:
        """Re-compute the edges in the graph

        This assumes that the position of the underlying nodes has changed.

        TODO: Different logic for ground stations and satellites.
        """
        # Clear all edges in the graph
        self.G.clear_edges()

        # Loop through each agent pair and remake the edges
        for agent1, agent2 in itertools.combinations(self._nodes.values(), 2):
            # Check if the distance is within range
            dist = np.linalg.norm(agent1.pos - agent2.pos)
            if self._config.min_range < dist < self._config.max_range:
                # Check if the Earth is blocking the two agents
                if not linalg.intersects_earth(agent1.pos, agent2.pos):
                    # Add the edge width bandwidth metadata
                    self.G.add_edge(
                        agent1.name,
                        agent2.name,
                        max_bandwidth=self._config.max_bandwidth,
                        used_bandwidth=0,
                    )
                    # also add the edge in the opposite direction
                    self.G.add_edge(
                        agent2.name,
                        agent1.name,
                        max_bandwidth=self._config.max_bandwidth,
                        used_bandwidth=0,
                    )

        # Restrict to just the maximum number of neighbors
        for agent in self._nodes.values():
            # If the number of neighbors is greater than the max, remove the extra neighbors
            if (
                len(neighbors := list(self.G.neighbors(agent.name)))
                <= self._config.max_neighbors
            ):
                continue

            # Get the list of neighbors
            neighbors = list(self._nodes[neighbor] for neighbor in neighbors)

            # Get the distances to each neighbor
            dists = [np.linalg.norm(neighbor.pos - agent.pos) for neighbor in neighbors]

            # Sort the neighbors by distance
            sorted_neighbors = [
                x for _, x in sorted(zip(dists, neighbors), key=lambda pair: pair[0])
            ]

            # Remove the extra neighbors
            for i in range(self._config.max_neighbors, len(sorted_neighbors)):
                self.G.remove_edge(agent.name, sorted_neighbors[i].name)
