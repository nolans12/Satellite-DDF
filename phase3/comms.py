import itertools
import logging
import threading
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

        self.bounties = dataclassframe.DataClassFrame(clz=collection.BountyTransmission)

        self._sensing_sats = {sat.name for sat in sensing_sats}
        self._fusion_sats = {sat.name for sat in fusion_sats}
        self._ground_stations = {gs.name for gs in ground_stations}

        # Create a graph instance with the names as nodes
        self._nodes = {node.name: node for node in sensing_sats + fusion_sats}
        self.G = nx.DiGraph()
        # Add nodes with a dict for queued data (list of arrays)
        for node in self._nodes:
            self.G.add_node(node)
        self.update_edges()

        # Concurrency handling
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

    def send_measurements_path(
        self,
        measurements: list[collection.Measurement],
        source: str,
        destination: str,
        time: float,
        size: float,
    ) -> None:
        """
        Send a measurement through a chain of satellites in the network.
        """

        # get the path from sender to receiver
        path = self.get_path(source, destination, size)

        if path is None:
            return

        # send the measurements through the path
        for i in range(1, len(path)):
            self.send_measurements_pair(
                measurements, path[i - 1], path[i], source, destination, time, size
            )

    def send_measurements_pair(
        self,
        measurements: list[collection.Measurement],
        sender: str,
        receiver: str,
        source: str,
        destination: str,
        time: float,
        size: float,
    ) -> None:
        """
        Send a measurement through a pair of satellites in the network.
        """
        # Should only enter this if valid path that doesnt violate bandwidth constraints... so dont check
        # print(f'Sending {size} bytes from {sender} to {receiver} at time {time}')

        # Set the edge to be active
        self.G[sender][receiver]['active'] = "Measurement"
        # print(f"Activated edge from {sender} to {receiver}")

        # Create a transmition
        self.measurements.append(
            collection.MeasurementTransmission(
                sender,
                receiver,
                source,
                destination,
                size,
                time,
                measurements,
            )
        )

        # Update the edge bandwidth
        self.G[sender][receiver]['used_bandwidth'] += size

    def receive_measurements(
        self, receiver: str, time: float
    ) -> list[collection.Measurement]:
        """Receive all measurements for a node.
        Node has to be on the "destination" end of the transmission!
        Not just an edge between some path.

        Args:
            receiver: Node to receive measurements for.
            time: Time to get measurements for.

        Returns:
            List of measurements for the node.
        """
        transmissions = self.measurements.loc[
            (self.measurements['receiver'] == receiver)
            & (self.measurements['time'] >= time)
            & (self.measurements['destination'] == receiver)
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

    def send_bounty_path(
        self,
        sender: str,
        receiver: str,
        source: str,
        destination: str,
        target_id: int,
        size: float,
        time: float,
    ) -> None:
        """Send a bounty through a chain of satellites in the network.

        Args:
            sender: Initial sending node
            receiver: Final receiving node
            source: Original source of the bounty
            destination: Final destination of the bounty
            target_id: ID of the target the bounty is for
            size: Size of the bounty transmission in bytes
            time: Time the transmission occurs
        """
        # get the path from sender to receiver
        path = self.get_path(source, destination, size)

        if path is None:
            return

        # send the bounty through the path
        for i in range(1, len(path)):
            self.send_bounty_pair(
                path[i - 1], path[i], source, destination, target_id, size, time
            )

    def send_bounty_pair(
        self,
        sender: str,
        receiver: str,
        source: str,
        destination: str,
        target_id: int,
        size: float,
        time: float,
    ) -> None:
        """Send a bounty between a pair of satellites in the network.

        Args:
            sender: Sending node
            receiver: Receiving node
            source: Original source of the bounty
            destination: Final destination of the bounty
            target_id: ID of the target the bounty is for
            size: Size of the bounty transmission in bytes
            time: Time the transmission occurs
        """

        # # Set the edge to be active
        # self.G[sender][receiver]['active'] = "Bounty"

        self.bounties.append(
            collection.BountyTransmission(
                sender=sender,
                receiver=receiver,
                source=source,
                destination=destination,
                target_id=target_id,
                size=size,
                time=time,
            )
        )

        # Update the edge bandwidth
        self.G[sender][receiver]['used_bandwidth'] += size

    def get_neighbors(self, node: str) -> list[str]:
        """Get the neighbors of a node.

        Args:
            node: Satellite to get the neighbors of.

        Returns:
            List of names of neighbors of the satellite.
        """
        return list(self.G.neighbors(node))

    def get_nodes(self, sat_type: str) -> list[str]:
        """Get all nodes of a given type.

        Loop through all nodes and return the ones of the given type
        """
        if sat_type == "fusion":
            return list(self._fusion_sats)
        elif sat_type == "sensing":
            return list(self._sensing_sats)
        else:
            # Log an error
            logging.error(f"Invalid satellite type: {sat_type}")
            exit(1)

    def get_nearest(
        self, position: npt.NDArray, sat_type: str, number: int
    ) -> list[str]:
        """Get the nearest X amount of satellites to a given position of a given type."""

        # Get all nodes of a given type
        options = self.get_nodes(sat_type)

        # Get the nearest X amount of nodes
        nearest = sorted(
            options,
            key=lambda x: np.linalg.norm(position - self._nodes[x].pos),
        )[:number]

        return nearest

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
            path: list[str] = nx.shortest_path(G_valid, source=node1, target=node2, weight='distance')  # type: ignore
            # print(f"Path from {node1} to {node2}: {path}")
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

        Assume the fusion layer has to abide by max_neighbors when connecting fusion to fusion.
        But, sensing layer can connect to any other node in its range.
        """

        # Clear all edges and their active states in the graph
        self.G.clear_edges()

        # Add all edges in the fusion layer first
        for agent1, agent2 in itertools.combinations(self._nodes.values(), 2):
            # If either agent is not in the fusion layer, skip
            if not (
                agent1.name.startswith("FusionSat")
                and agent2.name.startswith("FusionSat")
            ):
                continue

            # If edge already exists, skip
            if self.G.has_edge(agent1.name, agent2.name):
                continue

            dist = np.linalg.norm(agent1.pos - agent2.pos)
            if self._config.min_range < dist < self._config.max_range:
                if not linalg.intersects_earth(agent1.pos, agent2.pos):
                    # Add edges in both directions, between the fusion nodes
                    self.G.add_edge(
                        agent1.name,
                        agent2.name,
                        max_bandwidth=self._config.max_bandwidth,
                        used_bandwidth=0,
                        distance=dist,
                    )
                    self.G.add_edge(
                        agent2.name,
                        agent1.name,
                        max_bandwidth=self._config.max_bandwidth,
                        used_bandwidth=0,
                        distance=dist,
                    )
                    # Set the edge to be inactive
                    self.G[agent1.name][agent2.name]['active'] = ""
                    self.G[agent2.name][agent1.name]['active'] = ""

        # Now loop through and remove the edges to abide by max_neighbors
        for agent in self._nodes.values():
            if agent.name.startswith("FusionSat"):
                # If the number of neighbors is greater than the max, remove the extra neighbors
                if (
                    len(neighbors := list(self.G.neighbors(agent.name)))
                    <= self._config.max_neighbors
                ):
                    continue

                # Get the list of neighbors
                neighbors = list(self._nodes[neighbor] for neighbor in neighbors)

                # If the number of neighbors is greater than the max, remove the farthest neighbors
                dists = [
                    np.linalg.norm(neighbor.pos - agent.pos) for neighbor in neighbors
                ]

                # Sort the neighbors by distance
                sorted_neighbors = [
                    x
                    for _, x in sorted(zip(dists, neighbors), key=lambda pair: pair[0])
                ]

                # Remove the extra neighbors
                for i in range(self._config.max_neighbors, len(sorted_neighbors)):
                    self.G.remove_edge(agent.name, sorted_neighbors[i].name)

        # Now, finally, add all sensing to fusion edges!
        for agent1, agent2 in itertools.combinations(self._nodes.values(), 2):

            # Only do fusion - sensing combo!
            if not (
                agent1.name.startswith("SensingSat")
                and agent2.name.startswith("FusionSat")
            ):
                continue

            # If edge already exists, skip
            if self.G.has_edge(agent1.name, agent2.name):
                continue

            # Get all fusion nodes and their distances from this sensing node
            fusion_nodes = [
                node
                for node in self._nodes.values()
                if node.name.startswith("FusionSat")
            ]
            dists = [np.linalg.norm(agent1.pos - node.pos) for node in fusion_nodes]

            # Sort fusion nodes by distance
            sorted_fusion = [
                x for _, x in sorted(zip(dists, fusion_nodes), key=lambda pair: pair[0])
            ]

            # Only connect to the nearest max_neighbors fusion nodes that are in range
            for fusion_node in sorted_fusion[: self._config.max_neighbors]:
                dist = np.linalg.norm(agent1.pos - fusion_node.pos)
                if self._config.min_range < dist < self._config.max_range:
                    if not linalg.intersects_earth(agent1.pos, fusion_node.pos):
                        self.G.add_edge(
                            agent1.name,
                            fusion_node.name,
                            max_bandwidth=self._config.max_bandwidth,
                            used_bandwidth=0,
                            distance=dist,
                        )
                        self.G.add_edge(
                            fusion_node.name,
                            agent1.name,
                            max_bandwidth=self._config.max_bandwidth,
                            used_bandwidth=0,
                            distance=dist,
                        )
                        # Set the edge to be inactive
                        self.G[agent1.name][fusion_node.name]['active'] = ""
                        self.G[fusion_node.name][agent1.name]['active'] = ""

    def print_average_comms_distance(self, time: float) -> None:
        """
        Given a time, find all measurements that were sent at the time and
        print the average distance a measurement had to travel from sender to reciever
        """

        # Get all measurements at a given time:
        meas = self.measurements.loc[self.measurements['time'] == time]

        # Get all unique instances of source destination about a given target_Id
        unique_meas = meas.drop_duplicates(subset=['source', 'destination'])

        total_distance = 0
        # Get each path:
        for _, row in unique_meas.iterrows():
            source = row['source']
            destination = row['destination']
            path = self.get_path(source, destination, row['size'])

            # Get the distance of the path
            distance = sum(
                self.G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1)
            )
            total_distance += distance

        print(f"Average distance: {total_distance / len(unique_meas)}")
        exit()
