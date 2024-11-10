import itertools
import logging
from typing import Generic, Protocol, Sequence, TypeVar, overload

import networkx as nx
import numpy as np
from numpy import typing as npt

from common import dataclassframe
from common import linalg
from phase3 import collection
from phase3 import sim_config


class Agent(Protocol):
    """Satellite, ground station, or w/e."""

    name: str

    # [x, y, z]
    @property
    def pos(self) -> npt.NDArray: ...


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
        sensing_sats: Sequence[S],
        fusion_sats: Sequence[F],
        ground_stations: Sequence[G],
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

        self.measurements = dataclassframe.DataClassFrame(
            clz=collection.MeasurementTransmission
        )

        self._sensing_sats = {sat.name for sat in sensing_sats}
        self._fusion_sats = {sat.name for sat in fusion_sats}
        self._ground_stations = {gs.name for gs in ground_stations}

        # Create a graph instance with the names as nodes
        self._nodes = {
            node.name: node for node in sensing_sats + fusion_sats + ground_stations
        }
        self.G = nx.DiGraph()
        # Add nodes with a dict for queued data (list of arrays)
        for node in self._nodes:
            self.G.add_node(node)
        self.update_edges()

    def send_measurements_path(
        self,
        measurements: list[collection.Measurement],
        sender: str,
        receiver: str,
        time: float,
        size: float,
    ) -> None:
        """
        Send a measurement through a chain of satellites in the network.
        """

        # Check what the path is

        # here send measurement through a chain of satellites in the network
        # and then post processing the measurements recieved
        # will check if the target id were talking about is in the custody of the fusion satellite
        # if so, process in EKF the measurements
        # otherwise skip

        # get the path from sender to receiver
        path = self.get_path(sender, receiver, size)

        if path is None:
            return

        # send the measurements through the path
        for i in range(1, len(path)):
            self.send_measurements_pair(measurements, path[i - 1], path[i], time, size)

    def send_measurements_pair(
        self,
        measurements: list[collection.Measurement],
        sender: str,
        receiver: str,
        time: float,
        size: float,
    ) -> None:
        """
        Send a measurement through a pair of satellites in the network.
        """
        # Should only enter this if valid path that doesnt violate bandwidth constraints... so dont check
        print(f'Sending {size} bytes from {sender} to {receiver} at time {time}')

        # Create a transmition
        self.measurements.append(
            collection.MeasurementTransmission(
                sender, receiver, size, time, measurements
            )
        )

        # Update the edge bandwidth
        self.G[sender][receiver]['used_bandwidth'] += size

    def receive_measurements(
        self, receiver: str, time: float
    ) -> list[collection.Measurement]:
        """Receive all measurements for a node.

        Args:
            receiver: Node to receive measurements for.
            time: Time to get measurements for.

        Returns:
            List of measurements for the node.
        """
        transmissions = self.measurements.loc[
            (self.measurements['receiver'] == receiver)
            & (self.measurements['time'] >= time)
        ]

        # Convert transmissions to list of Measurements
        measurements = []
        for _, transmission in transmissions.iterrows():
            # Convert each dict back to a Measurement dataclass
            measurements.extend(
                [
                    collection.Measurement(**m) if isinstance(m, dict) else m
                    for m in transmission.measurements
                ]
            )

        return measurements

    def get_neighbors(self, node: str) -> list[str]:
        """Get the neighbors of a node.

        Args:
            node: Satellite to get the neighbors of.

        Returns:
            List of names of neighbors of the satellite.
        """
        return list(self.G.neighbors(node))

    def get_distance(self, node1: str, node2: str) -> float:
        """Get the distance between two nodes.

        Args:
            node1: Starting node.
            node2: Ending node.
        """
        return np.linalg.norm(self._nodes[node1].pos - self._nodes[node2].pos)

    def get_path(self, node1: str, node2: str, size: float) -> list[str] | None:
        """Get the shortest path between two nodes.

        Args:
            node1: Starting node.
            node2: Ending node.

        Returns:
            Shortest path between the two nodes.
        """
        try:
            # Create a copy of the graph with edges filtered by available bandwidth
            valid_edges = [
                (u, v)
                for u, v in self.G.edges()
                if self.G[u][v]['max_bandwidth'] - self.G[u][v]['used_bandwidth']
                >= size
            ]

            # Create subgraph with only valid edges that can support the bandwidth
            G_valid = self.G.edge_subgraph(valid_edges)

            # Find shortest path in the valid subgraph
            path: list[str] = nx.shortest_path(G_valid, node1, node2)  # type: ignore
            return [self._nodes[node].name for node in path]

        except nx.NetworkXNoPath:
            logging.warning(
                f'No path with sufficient bandwidth between {node1} and {node2} in the communication network.'
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
