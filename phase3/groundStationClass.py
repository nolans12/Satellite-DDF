from import_libraries import *
## Creates the ground station class. Will contain a ground station with an estimator onboard.

class groundStation:
    def __init__(self, lat, long, fov, commRange, estimator, name, color):
        """Initialize a Ground Station object.

        Args:
            lat (float): Latitude of the ground station.
            long (float): Longitude of the ground station.
            estimator (object): Estimator to use for the ground station.
            name (str): Name of the ground station.
            color (str): Color of the ground station for visualization.
        """

        # Location of the ground station in ECEF
        lat = np.deg2rad(lat)
        long = np.deg2rad(long)
        self.loc = np.array([6371 * np.cos(lat) * np.cos(long), 6371 * np.cos(lat) * np.sin(long), 6371 * np.sin(lat)])

        # Estimator object to use
        self.estimator = None

        # Communication range of a satellite to the ground station
        self.fov = fov
        self.commRange = commRange

        # Other parameters
        self.name = name
        self.color = color
        self.time = 0
       

    # TODO: REDEFINE THIS TO BE BASED ON THE CONE OF GROUND STATION COMMS
    def canCommunicate(self, sat):
        """Check if the ground station can communicate with a satellite.

        Args:
            sat (object): Satellite object to check communication with.

        Returns:
            bool: True if the ground station can communicate with the satellite, False otherwise.
        """

        # Create two lines, one from the center of earth to GS and one from GS to satellite

        # Get the satellite position
        x_sat, y_sat, z_sat = sat.orbit.r.value

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
                # print("Satellite " + sat.name + " can communicate with ground station " + self.name)
                return True
            
        return False
