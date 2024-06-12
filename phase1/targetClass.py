from import_libraries import *
## Creates the target class
class boat_target:
    def __init__(self, name, targetID, r, color):
        # Intial target ID
        self.targetID = targetID
        self.name = name
        self.color = color
        
        # Set time to 0
        self.time = 0
        
        # ECI Position
        self.pos = np.array([0, 0, 0])
        self.vel = np.array([0, 0, 0])
        self.hist = defaultdict(dict) # contains time and xyz and velocity history in ECI [x xdot y ydot z zdot]
        
        #  Target State r = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]'
        self.r = np.array(r)
        
    def propagate(self, time_step, time):
        # TODO: Make targets follow CWNA model but then constrain z to surface make this in terms for x xdot y ydot z zdot
        
        # Input: Target Current State,
        # TimeStep
        dt = time_step.value
        t = time.value
        
        # White Noise Intensity Vector -> should be order of maximim magnitude acceleration over dt    
        rangeAccelNoise = 0.000001
        elevationAccelNoise = 0.001
        azimuthAccelNoise = 0.001
        q = np.array([0, rangeAccelNoise, 0, elevationAccelNoise, 0, azimuthAccelNoise]).T
        
        # Define Continuous White Noise State Equation
        
        # State Dynamics Matrix
        A = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
        
        # Control Input Matrix --> Acceleration
        B = np.array([0, 1, 0, 1, 0, 1]).T
        
        # Covariance Matrix
        Q = np.array([[0, 0, 0, 0, 0, 0],
                      [0, dt, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, dt, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, dt]])*q**2        
        
        # Sample MultiVariate White Noise as input to State Equation
        mean = np.array([0, 0, 0, 0, 0, 0])
        
        # 50% chance of acceleration
        if(np.random.uniform(0,1) < 0.5):
            acceleration = np.random.multivariate_normal(mean, Q, 1).T
        else:
            acceleration = np.array([0, 0, 0, 0, 0, 0]).T
            
        # Propagate the state
        rDot = np.dot(A, self.r) + np.dot(B, acceleration)
        self.r = self.r + rDot*dt
        
        # Extract for reading simplicity
        range = self.r[0]
        rangeRate = self.r[1]
        elevation = self.r[2]
        elevationRate = self.r[3]
        azimuth = self.r[4]
        azimuthRate = self.r[5]
        
        # Convert back to x,y,z ECI
        x = range*np.sin(azimuth)*np.cos(elevation)
        y = range*np.sin(azimuth)*np.sin(elevation)
        z = range*np.cos(elevation)
        
        self.pos = np.array([x, y, z])
        
        # Convert back to xdot, ydot, zdot ECI
        vx = (rangeRate * np.sin(azimuth) * np.cos(elevation) 
               + range * np.cos(azimuth) * np.cos(elevation) * azimuthRate 
               - range * np.sin(azimuth) * np.sin(elevation) * elevationRate)
        
        vy = (rangeRate * np.sin(azimuth) * np.sin(elevation) 
               + range * np.cos(azimuth) * np.sin(elevation) * azimuthRate 
               + range * np.sin(azimuth) * np.cos(elevation) * elevationRate)
        
        vz = (rangeRate * np.cos(azimuth) 
               - range * np.sin(azimuth) * azimuthRate)
        
        self.vel = np.array([vx, vy, vz])
        
        
class HGV_target:
    def __init__(self, name, targetID, X, color):
        # Initialize a Hypersonic Glide Vehicle
        self.name = name
        self.targetID = targetID
        self.color = color
        
        # Set time to 0
        self.time = 0
        
        # ECI Position
        self.pos = np.array([0, 0, 0])
        self.vel = np.array([0, 0, 0])
        self.hist = defaultdict(dict) # contains time and xyz and velocity history in ECI [x xdot y ydot z zdot]
        
        # Define initial state: X = [speed, flight path angle, heading, radial distance, long, lat]
        # Notes: Flight Path Angle = Angle btwn velocity and horizon (climb or descend)
        #        Heading = Measured from north in CW direction
        #
        self.X = X
        self.m = 1 # mass kg
        self.sref = 1 # wetted area m^2
        self.bank = 0 # bank angle is initial set to zero
        self.Cd = 0.1 # coefficient of drag found from https://www.researchgate.net/figure/Drag-coefficients-for-a-different-type-of-nose-cones_tbl3_352657394#:~:text=The%20drag%20coefficient%20was%20found,ogive%20nose%20cone.%20...
        
        
        # TODO: Implement Control Input
        # Control inputs (alpha and nu)
        self.alpha = 0.05  # Angle of attack [rad]
        nu = 0.1  # Bank angle [rad]
        
    def CL(self, alpha, Ma):
    # Placeholder function, replace with actual model
        return 0.5

    def CD(self, alpha, Ma):
        # Placeholder function, replace with actual model
        return 0.02
        
    
    def propagate(self, dt, time):
        # Define the Differential Equations of Motion that describe vehicle dynamics
        #   Assume Equilibirum Glide Mode --> the total acceleration in the longitudinal plane is almost zero 
        #                                     and the corresponding trajectory is smooth
        #
        #       Implies that flight path angle is small and rate of change is approximately zero
        
        # Graviational Constant
        g = 9.81 * 60**2/10**3 # m/s^2 * 1km * (60s)^2 / (1min)^2 * (1000m) = km/min^2
        
        # TimeStep
        dt = dt.value
        t = time.value
        
        
        # Extract States for Reading Simplicity
        V = self.X[0]
        theta = np.deg2rad(self.X[1])
        sigma = np.deg2rad(self.X[2])
        radius = self.X[3]
        lon = self.X[4]
        lat = self.X[5]

        bankAngle = np.deg2rad(self.bank)
        
        # Get the height of the vehicle
        height = radius - 6378 
        
        # Create a standard atmospehre at the current height
        currAtmosphere = Atmosphere(height)
        
        # Get the density at that height
        rho = currAtmosphere.density # UNITS?
                
        # Get the dynamic pressure
        q = 0.5*rho*V**2
        
        # Get the coefficeint of lift
        #Cl = self.m * (g - V**2/radius)/ (q * self.sref * np.cos(bankAngle))
        lift = self.CL(self.alpha, V / 343) * q * self.sref / self.m
        drag = self.CD(self.alpha, V / 343) * q * self.sref / self.m

        # Compute magnitude of lift and drag
        # lift = 0.5*Cl*rho*V**2*self.sref / self.m
        # drag = 0.5*self.Cd*rho*V**2*self.sref / self.m
        
        # Compute DEOM
        V_dot = np.array(-drag - g*np.sin(theta))
        theta_dot = np.array((lift*np.cos(bankAngle) + (V**2/radius - g) * np.cos(theta))/V)
        sigma_dot = np.array(lift*np.sin(bankAngle)/(V*np.cos(theta)) + V*np.tan(lat)*np.cos(theta)*np.sin(sigma)/radius)
        r_dot = np.array([V*np.sin(theta)])
        lon_dot = np.array([-V*np.cos(theta)*np.sin(sigma)/(radius*np.cos(lat))])
        lat_dot = np.array([V*np.cos(theta)*np.sin(sigma)/(radius)])
        
        # For Small Time Step Assume Linear Change in Variables
        X_dot = np.array([V_dot, theta_dot, sigma_dot, r_dot, lon_dot, lat_dot])
        X_dot = X_dot.reshape((6, 1))
        
        # New State = Prev State + xdot*dt
        X_new = (self.X + np.dot(X_dot, dt).T).flatten()
        self.X = X_new
                
        radius = X_new[3]
        lon = X_new[4]
        lat = X_new[5]
        
        
        # Convert back to x,y,z ECI
        x = radius*np.sin(lon)*np.cos(lat)
        y = radius*np.sin(lon)*np.sin(lat)
        z = radius*np.cos(lat)
        
        self.pos = np.array([x, y, z])
        
        # Convert back to xdot, ydot, zdot ECI
        vx = (r_dot * np.sin(lon) * np.cos(lat) 
               + radius * np.cos(lon) * np.cos(lat) * lon_dot 
               - radius * np.sin(lon) * np.sin(lat) * lat_dot)
        
        vy = (r_dot * np.sin(lon) * np.sin(lat) 
               + radius * np.cos(lon) * np.sin(lat) * lon_dot 
               + radius * np.sin(lon) * np.cos(lat) * lat_dot)
        
        vz = (r_dot * np.cos(lon) 
               - radius * np.sin(lon) * lon_dot)
        
        self.vel = np.array([vx, vy, vz]).flatten()
        print("=" * 50)
        print(self.X)
        print("*" * 50)
        print(self.pos)
        print(self.vel)