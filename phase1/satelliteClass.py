from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, fovNarrow, fovWide, sensorDetectError, sensorError, name, color):
    
    # Create the orbit
        # Check if already in units, if not convert
        if type(a) == int:
            a = a * u.km
        self.a = a
        
        if type(ecc) == int:
            ecc = ecc * u.dimensionless_unscaled
        self.ecc = ecc 
       
        if type(inc) == int:
            inc = inc * u.deg
        self.inc = inc 
        
        if type(raan) == int:
            raan = raan * u.deg
        self.raan = raan 
        
        if type(argp) == int:
            argp = argp * u.deg
        self.argp = argp 
        
        if type(nu) == int:
            nu = nu * u.deg
        self.nu = nu 
        
        # Create the poliastro orbit
        self.orbit = Orbit.from_classical(Earth, self.a, self.ecc, self.inc, self.raan, self.argp, self.nu)
        self.orbitHist = []; # contains xyz of orbit history
        self.fullHist = []; # contains xyz and time of full history
        self.time = 0

        self.estimateHist = [] # contains xyz and t of estimated history

    # Initalize the projection xyz
        self.projBox = np.array([0, 0, 0])

    # Define sensor settings
        self.fovNarrow = fovNarrow
        self.fovWide = fovWide
        self.sensorDetectError = sensorDetectError
        self.sensorError = sensorError

    # Other parameters
        self.name = name
        self.color = color

# Calculate the visible projection of the satellite
    # Returns the 4 points of xyz intersection with the earth that approximately define the visible projection
    def visible_projection(self):

    # Need the 4 points of intersection with the earth
        # Get the current xyz position of the satellite
        x, y, z = self.orbit.r.value

        # Get the altitude above earth of the satellite
        alt = np.linalg.norm([x, y, z]) - 6378.0

        # Now calculate the magnitude of fov onto earth
        wideMag = np.tan(np.radians(self.fovWide)/2) * alt
        narrowMag = np.tan(np.radians(self.fovNarrow)/2) * alt

        # Then vertices of the fov box onto the earth is xyz projection +- magnitudes
        # Get the pointing vector of the satellite
        point_vec = np.array([x, y, z])/np.linalg.norm([x, y, z])
        
        # Now get the projection onto earth of center of fov box
        center_proj = np.array([x - point_vec[0] * alt, y - point_vec[1] * alt, z - point_vec[2] * alt])

        # Now get the 4 xyz points that define the fov box
        # Define vectors representing the edges of the FOV box
        wide_vec = np.cross(point_vec, [0, 0, 1])/np.linalg.norm(np.cross(point_vec, [0, 0, 1]))
        narrow_vec = np.cross(point_vec, wide_vec)/np.linalg.norm(np.cross(point_vec, wide_vec))

        # Calculate the four corners of the FOV box
        corner1 = center_proj + wide_vec * wideMag + narrow_vec * narrowMag
        corner2 = center_proj + wide_vec * wideMag - narrow_vec * narrowMag
        corner3 = center_proj - wide_vec * wideMag - narrow_vec * narrowMag
        corner4 = center_proj - wide_vec * wideMag + narrow_vec * narrowMag

        box = np.array([corner1, corner2, corner3, corner4])

        return box