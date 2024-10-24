import itertools

import networkx as nx
import numpy as np
from astropy import units as u
from numpy import typing as npt

from common import dataclassframe
from common import linalg
from phase3 import collection
from phase3 import satellite


class Comms:
    """
    Communication network class.
    """

    def __init__(
        self,
        sats: list[satellite.Satellite],
        maxNeighbors: int,
        maxRange: u.Quantity[u.km],
        minRange: u.Quantity[u.km],
        maxBandwidth: int = 100000000,
    ):
        """Initialize the communications network.
                Using networkx, a python library, to create a graph of the satellites and their connections.

        Args:
            sats: List of satellites.
            max_neighbors: Maximum number of neighbors.
            max_range: Maximum range for communication.
            min_range: Minimum range for communication.
            data_rate: Data rate. Defaults to 0.
        """
        self.maxBandwidth = maxBandwidth

        # Create a empty dicitonary to store the amount of data sent/recieved between satellites
        self.total_comm_data = dataclassframe.DataClassFrame(
            clz=collection.Transmission
        )
        self.used_comm_data = dataclassframe.DataClassFrame(clz=collection.Transmission)

        self.total_comm_et_data = dataclassframe.DataClassFrame(
            clz=collection.MeasurementTransmission
        )
        self.used_comm_et_data = dataclassframe.DataClassFrame(
            clz=collection.MeasurementTransmission
        )

        self.max_neighbors = maxNeighbors
        self.max_range = maxRange
        self.min_range = minRange

        # Create a graph instance with the satellites as nodes
        self._sats = sats
        self.G = nx.DiGraph()
        # Add nodes with a dict for queued data (list of arrays)
        for sat in sats:
            self.G.add_node(
                sat,
                estimate_data={},
                received_measurements={},
                sent_measurements={},
            )
        self.update_edges()

    def send_estimate_path(
        self,
        path: list[satellite.Satellite],
        est_meas_source: npt.NDArray,
        cov_meas_source: npt.NDArray,
        target_id: int,
        time: float,
    ) -> None:
        """Send an estimate along a path of satellites.
        Includes ferrying the data!!!
        """

        # Take the path, and from path[0] (source node) share the est_meas and cov_meas to all other nodes in the path, send the data
        for i in range(1, len(path)):
            self.send_estimate(
                path[i - 1], path[i], est_meas_source, cov_meas_source, target_id, time
            )

    def send_estimate(
        self,
        sender: satellite.Satellite,
        receiver: satellite.Satellite,
        est_meas: npt.NDArray,
        cov_meas: npt.NDArray,
        target_id: int,
        time: float,
    ) -> None:
        """Send an estimate from one satellite to another
                First checks if two satellits are neighbors,
                then, if they are neighbors, we want to share the most recent estimate
                from sender to the receiver. We will simulate "sending a measurement"
                by adding the estimate to the receiver's queued data on the commnication node.
                This way, at the end of the time step, the reciever satellite can just loop through
                the queued data and update its estimate using DDF algorithms on it.

                The queued data is a dictionary of dictionaries of lists. The first key is the time,
                the second key is the target ID, and the list contains the estimates, covariances, and who sent them.

        Args:
            sender: Satellite sending the estimate.
            receiver): Satellite receiving the estimate.
            est_meas: Estimate to send.
            cov_meas: Covariance estimate to send.
            target_id: ID of the target the estimate is from.
            time: Time the estimate was taken.
        """
        # Check if the receiver is in the sender's neighbors
        if not self.G.has_edge(sender, receiver):
            return

        # Before we decide if we want to send the estimate, need to make sure it wont violate the bandwidth constraints
        # Check if the bandwidth is available
        if (
            self.G.edges[sender, receiver]['usedBandwidth']
            + est_meas.size * 2
            + cov_meas.size / 2
            > self.G.edges[sender, receiver]['maxBandwidth']
        ):
            # print(f"Bandwidth exceeded between {sender.name} and {receiver.name} with current bandwith of {self.G.edges[sender, receiver]['usedBandwidth']} and max bandwidth of {self.G.edges[sender, receiver]['maxBandwidth']}")
            return
        else:
            # Update the used bandwidth
            self.G.edges[sender, receiver]['usedBandwidth'] += (
                est_meas.size * 2 + cov_meas.size / 2
            )

        # Initialize the target_id key in the receiver's queued data if not present
        if time not in self.G.nodes[receiver]['estimate_data']:
            self.G.nodes[receiver]['estimate_data'][time] = {}

        # Initialize the time key in the target_id's queued data if not present
        if target_id not in self.G.nodes[receiver]['estimate_data'][time]:
            self.G.nodes[receiver]['estimate_data'][time][target_id] = {
                'est': [],
                'cov': [],
                'sender': [],
            }

        # Add the estimate to the receiver's queued data at the specified target_id and time
        self.G.nodes[receiver]['estimate_data'][time][target_id]['est'].append(est_meas)
        self.G.nodes[receiver]['estimate_data'][time][target_id]['cov'].append(cov_meas)
        self.G.nodes[receiver]['estimate_data'][time][target_id]['sender'].append(
            sender.name
        )

        self.total_comm_data.append(
            collection.Transmission(
                target_id=target_id,
                sender=sender.name,
                receiver=receiver.name,
                time=time,
                size=est_meas.size * 2 + cov_meas.size // 2,
            )
        )

    def send_measurements(
        self,
        sender: satellite.Satellite,
        receiver: satellite.Satellite,
        alpha: float,
        beta: float,
        target_id: int,
        time: float,
    ) -> None:
        """Send a vector of measurements from one satellite to another.
                First checks if two satellites are neighbors,
                then, if they are neighbors, we share the measurement vector
                from the sender to the receiver by adding it to the receiver's
                measurement data on the communication node.

                The measurement data is a dictionary of dictionaries of lists.
                The first key is the time, the second key is the target ID,
                and the list contains the measurement vectors and who sent them.

        Args:
            sender: Satellite sending the measurements.
            receiver: Satellite receiving the measurements.
            alpha: Alpha measurement to send.
            beta: Beta measurement to send.
            target_id: ID of the target the measurements are from.
            time: Time the measurements were taken.
        """
        # Check if the receiver is in the sender's neighbors
        if not self.G.has_edge(sender, receiver):
            return

        # Initialize the time key in the receiver's measurement data if not present
        if time not in self.G.nodes[receiver]['received_measurements']:
            self.G.nodes[receiver]['received_measurements'][time] = {}

        # Initialize the target_id key in the measurement data if not present
        if target_id not in self.G.nodes[receiver]['received_measurements'][time]:
            self.G.nodes[receiver]['received_measurements'][time][target_id] = {
                'meas': [],
                'sender': [],
            }

        # Add the measurement vector to the receiver's measurement data at the specified target_id and time
        self.G.nodes[receiver]['received_measurements'][time][target_id]['meas'].append(
            (alpha, beta)
        )
        self.G.nodes[receiver]['received_measurements'][time][target_id][
            'sender'
        ].append(sender)

        # Add the measurement vector to the senders sent measurements at the specified target_id and time
        if time not in self.G.nodes[sender]['sent_measurements']:
            self.G.nodes[sender]['sent_measurements'][time] = {}

        if target_id not in self.G.nodes[sender]['sent_measurements'][time]:
            self.G.nodes[sender]['sent_measurements'][time][target_id] = {
                'meas': [],
                'receiver': [],
            }

        self.G.nodes[sender]['sent_measurements'][time][target_id]['meas'].append(
            (alpha, beta)
        )
        self.G.nodes[sender]['sent_measurements'][time][target_id]['receiver'].append(
            receiver
        )

        measVector_size = 2 + 2  # 2 for the meas vector, 2 for the sensor noise
        if np.isnan(alpha):
            measVector_size -= 1

        if np.isnan(beta):
            measVector_size -= 1

        self.total_comm_et_data.append(
            collection.MeasurementTransmission(
                target_id=target_id,
                sender=sender.name,
                receiver=receiver.name,
                time=time,
                size=measVector_size,
                alpha=alpha,
                beta=beta,
            )
        )

        # self.total_comm_data[target_id][receiver.name][sender.name][time] = meas_vector.size # TODO: need a new dicitonary to store this and sent data

    def update_edges(self) -> None:
        """Reset the edges in the graph and redefine them based on range and if the Earth is blocking them.
        Iterates through combinations of satellites to check known pairs.
        An edge represnts a theorical communication link between two satellites.
        """
        # Clear all edges in the graph
        self.G.clear_edges()

        # Loop through each satellite pair and remake the edges
        for sat1, sat2 in itertools.combinations(self._sats, 2):
            # Check if the distance is within range
            dist = np.linalg.norm(sat1.orbit.r - sat2.orbit.r)
            if self.min_range < dist < self.max_range:
                # Check if the Earth is blocking the two satellites
                if not linalg.intersects_earth(sat1.orbit.r, sat2.orbit.r):
                    # Add the edge
                    self.G.add_edge(
                        sat1,
                        sat2,
                        maxBandwidth=self.maxBandwidth,
                        usedBandwidth=0,
                    )
                    # also add the edge in the opposite direction
                    self.G.add_edge(
                        sat2,
                        sat1,
                        maxBandwidth=self.maxBandwidth,
                        usedBandwidth=0,
                    )

        # Restrict to just the maximum number of neighbors
        for sat in self._sats:
            # If the number of neighbors is greater than the max, remove the extra neighbors
            if len(list(self.G.neighbors(sat))) > self.max_neighbors:
                # Get the list of neighbors
                neighbors = list(self.G.neighbors(sat))

                # Get the distances to each neighbor
                dists = [
                    np.linalg.norm(neighbor.orbit.r - sat.orbit.r)
                    for neighbor in neighbors
                ]

                # Sort the neighbors by distance
                sorted_neighbors = [
                    x
                    for _, x in sorted(zip(dists, neighbors), key=lambda pair: pair[0])
                ]

                # Remove the extra neighbors
                for i in range(self.max_neighbors, len(sorted_neighbors)):
                    self.G.remove_edge(sat, sorted_neighbors[i])
