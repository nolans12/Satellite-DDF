import numpy as np

from phase3 import util
from phase3 import estimator

class GroundStation:
    """
    Ground station class
    """

    def __init__(
        self,
        estimator: 'estimator.Estimator',
        lat: float,
        lon: float,
        fov: float,
        commRange: float,
        name: str,
        color: str,
    ):

        """Initialize a GroundStation object.

        Args:
            estimator (object): Estimator to use for the ground station.
            lat (float): Latitude of the ground station.
            long (float): Longitude of the ground station.
            fov (float): Spherical field of view of the groundStation
            commRange (float): Communication range of the ground station (km).
            name (str): Name of the ground station.
            color (str): Color of the ground station for visualization.
        """

        # Estimator object to use
        self.estimator = estimator

        # Location of the ground station in ECEF
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        self.loc = np.array([6371 * np.cos(lat) * np.cos(lon), 6371 * np.cos(lat) * np.sin(lon), 6371 * np.sin(lat)])

        # Track communication sent/recieved
            # Dictionary containing [type][time][target][sat] = measurement, for mailbox system
        self.queued_data = util.NestedDict()
            # Make a dictionary containing [targetID][time][satName] = measurement, for post processing
        self.commData = util.NestedDict()

        # Communication range of a satellite to the ground station
        self.fov = fov
        self.commRange = commRange

        # Other parameters
        self.name = name
        self.color = color
        self.time = 0

    def queue_data(
        self, data: dict[int, dict[str, dict[int, int]]]
    ) -> None:
        """
        Adds the data to the queued data struct to be used later in processing, the mailbox system.

        Args:
            data, in order of [type][time][targetID][sat] = measurement
        """

        # Add the data to the queued data struct
        for type in data.keys():
            for time in data[type].keys():
                for targ in data[type][time].keys():
                    for sat in data[type][time][targ].keys():
                        self.queued_data[type][time][targ][sat] = data[type][time][targ][sat]

    def process_queued_data(
        self, time: int
    ) -> None:
        """
        Processes the data queued to be sent to the ground station.

        This function uses the estimator object to process the data.
        Queued data is of the form: [type][time][targetID][sat] = measurement

        Args:
            time: The time at which the data is being processed.
        """

        # First, figure out what data is available
        meas_data = self.queued_data['meas']
        est_data = self.queued_data['ci']

        if meas_data:
            # DO STANDARD EKF HERE WITH QUEUE MEASUREMENTS
            # Get the time for the data
            for time in meas_data.keys():
                # Get the target we are talking about
                for targ in meas_data[time].keys():
                    # Create blank list for all measurements and sats at that time on that target
                    measurements = []
                    sats = []
                    for sat in meas_data[time][targ].keys():
                        # Get the measurement
                        meas = meas_data[time][targ][sat]

                        # Store the measurement and satellite
                        measurements.append(meas)
                        sats.append(sat)

                        # Store the queued data into the commData, for post processing
                        self.commData[targ.targetID][time][sat.name] = meas

                    # Now, with the lists of measurements and sats, send to the estimator
                    if len(self.estimator.estHist[targ.targetID]) < 1:
                        self.estimator.gs_EKF_initialize(targ, time)
                        return

                    # Else, update the estimator
                    self.estimator.gs_EKF_pred(targ.targetID, time)
                    self.estimator.gs_EKF_update(sats, measurements, targ.targetID, time)

        if est_data:
            # DO COVARIANCE INTERSECTION HERE!!!
            # Get the time for the data
            for time in est_data.keys():
                # Get the target we are talking about
                for targ in est_data[time].keys():
                    for sat in est_data[time][targ].keys():
                        # Get the est and cov
                        est = est_data[time][targ][sat]['est']
                        cov = est_data[time][targ][sat]['cov']

                        # Now do CI with the data
                        self.estimator.CI_gs(targ.targetID, est, cov, time)

                        # Store the queued data into the commData, for post processing
                        self.commData[targ.targetID][time][sat.name] = {'est': est, 'cov': cov}

        # Clear the queued data
        self.queued_data = util.NestedDict()

    def can_communicate(
        self, x_sat: float, y_sat: float, z_sat: float
    ) -> bool:
        """
        Returns True if the satellite can communicate with the ground station at the given time.

        Args:
            x_sat, y_sat, z_sat (float): ECEF position of the satellite

        Returns:
            bool: True if the satellite can communicate with the ground station at the given time.
        """

        # Create two lines, one from the center of earth to GS and one from GS to satellite

        # Get the ground station position
        x_gs, y_gs, z_gs = self.loc

        # Earth to GS vec
        e_to_gs_vec = [x_gs - 0, y_gs - 0, z_gs - 0]

        # GS to satellite vec
        gs_to_sat_vec = [x_sat - x_gs, y_sat - y_gs, z_sat - z_gs]

        # Get the angle between the two vectors
        angle = np.arccos(np.dot(e_to_gs_vec, gs_to_sat_vec) / (np.linalg.norm(e_to_gs_vec) * np.linalg.norm(gs_to_sat_vec)))

        # Now check, can the satellite talk with the ground station
        if angle < np.deg2rad(self.fov):
            if np.linalg.norm(gs_to_sat_vec) < self.commRange:
                return True

        return False
