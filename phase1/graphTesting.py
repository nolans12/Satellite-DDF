from import_libraries import *

from satelliteClass import satellite
from targetClass import target
from environmentClass import environment
from estimatorClass import centralEstimator, localEstimator
from sensorClass import sensor

# This file will be using to test the networkx python library

if __name__ == "__main__":

    # Define a sensor model:
    sens1 = sensor(fov = 115, sensorError = np.array([1, 1]), detectError= 0.05, resolution = 720, name = 'Sensor 1')
    sens2 = sensor(fov = 115, sensorError = np.array([1, 1]), detectError= 0.05, resolution = 720, name = 'Sensor 2')
    sens3 = sensor(fov = 115, sensorError = np.array([1, 1]), detectError= 0.05, resolution = 720, name = 'Sensor 3')

    # Define targets for the satellites to track:
    targetIDs1 = np.array([1])

    # Define local estimators:
    local1 = localEstimator(targetIDs = targetIDs1)

    # Add a node, of a satellite
    sat1 = satellite(name = 'Sat1', sensor = sens1, targetIDs=targetIDs1, estimator = local1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 70, nu = 0, color='b')
    sat2 = satellite(name = 'Sat2', sensor = sens2, targetIDs=targetIDs1, estimator = local1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 50, nu = 0, color='r')
    sat3 = satellite(name = 'Sat3', sensor = sens3, targetIDs=targetIDs1, estimator = local1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 30, nu = 0, color='g')

    sats = [sat1, sat2, sat3]

#     # Define some targets:
#     targ1 = target(name = 'Targ1', targetID=1, r = np.array([6378, 0, 0, 0, 0, 0]),color = 'k')
   
#     targs = [targ1]
    
# # Create an estimator instance with the satellites and targets:
#     central = centralEstimator(sats, targs)

# # Create an environment instance:
#     env = environment(sats, targs)

# # Simulate the satellites through a vector of time:
#     time_vec = np.linspace(0, 10, 21) * u.minute
#     env.simulate(time_vec, display = True)

# # Save the gif:
#     env.render_gif(fileName = 'satellite_simulation.gif', fps = 5)

    ## TESTING GRAPH

# Create a graph
    G = nx.Graph()

# Add a node, of a satellite
    # G.add_node(sat1) # to do an individual node
    G.add_nodes_from(sats) # to do a list

    # Add an edge b/w sat1 and 2 and b/w sat2 and 3
    G.add_edge(sat1, sat2)
    G.add_edge(sat2, sat3)
    

# Display graph with labels
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, labels={sat: sat.name for sat in sats}, font_weight='bold', node_color=[sat.color for sat in sats])
    plt.show()