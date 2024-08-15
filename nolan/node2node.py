import pulp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Declare a directed satellite graph structure
g = nx.DiGraph()

# Add nodes: Sat1a, Sat1b, Sat2a, Sat2b to the graph
g.add_nodes_from(["Sat1a", "Sat1b", "Sat2a", "Sat2b"])

# Add node attributes for trackUncertainty
track_uncertainty = {
    "Sat1a": {1: 50, 2: 50, 3: 50, 4: 50, 5: 50},
    "Sat1b": {1: 105, 2: 130, 3: 130, 4: 135, 5: 135},
    "Sat2a": {1: 150, 2: 155, 3: 161, 4: 162, 5: 165},
    "Sat2b": {1: 200, 2: 275, 3: 280, 4: 280, 5: 290}
}

nx.set_node_attributes(g, track_uncertainty, 'trackUncertainty')

edgeBandwidth = 60

# Add directed edges with bandwidth constraints of 90
edges_with_bandwidth = [
    ("Sat1a", "Sat1b", edgeBandwidth), ("Sat1a", "Sat2a", edgeBandwidth), ("Sat1a", "Sat2b", edgeBandwidth),
    ("Sat1b", "Sat1a", edgeBandwidth), ("Sat1b", "Sat2a", edgeBandwidth), ("Sat1b", "Sat2b", edgeBandwidth),
    ("Sat2a", "Sat1a", edgeBandwidth), ("Sat2a", "Sat1b", edgeBandwidth), ("Sat2a", "Sat2b", edgeBandwidth),
    ("Sat2b", "Sat1a", edgeBandwidth), ("Sat2b", "Sat1b", edgeBandwidth), ("Sat2b", "Sat2a", edgeBandwidth)
]
g.add_weighted_edges_from(edges_with_bandwidth, weight='bandwidth')

# Define the layout for nodes
pos = nx.spring_layout(g)  # or nx.circular_layout(g) for a circular layout

# Draw the graph
nx.draw(g, pos, with_labels=True, node_color='lightblue', font_weight='bold', arrows=True)

# Draw edge labels for bandwidth constraints
edge_labels = nx.get_edge_attributes(g, 'bandwidth')
nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

# To get a dictionary of node attributes:
# nx.get_node_attributes(g, 'trackUncertainty').items()

##### Now that the graph is setup, do the optimization

# Goal: Set up a mixed-integer linear programming problem where:
# You maximize the total goodness for selected pairs
# You only send data to a satellite pair that has a track uncertainty less than a certain amount
# You consider bandwidth constraints on the edges

def goodness(s, t, targetID):
    """Calculate the goodness between nodes s and t based on track uncertainty."""

    """THIS SIMPLY PRIORITIZES DOING THE BEST TRACK UPDATES POSSIBLE, NOT PRIORITIZING TARGETS"""

    # Check if the track uncertainty for the source and target nodes is available
    if s not in track_uncertainty or t not in track_uncertainty:
        return 0

    sourceTrackUncertainty = track_uncertainty[s][targetID]
    recieverTrackUncertainty = track_uncertainty[t][targetID]

    # Check, if the sats track uncertainty on that targetID needs help or not
    if recieverTrackUncertainty < (50 + 50*targetID):
        return 0
   
    # Else, calculate the goodness, + if the source is better, 0 if the sat is better
    if recieverTrackUncertainty - sourceTrackUncertainty < 0:
        return 0 # Source has higher uncertaninty than sat, no benefit
    
    # Else, add the goodness to the total goodness, taking the difference in track uncertainty
    return recieverTrackUncertainty - sourceTrackUncertainty

# Define a dictionary to store the goodness values
goodness_dict = {}

# Calculate goodness for all pairs of nodes and all target IDs
for s in g.nodes():
    for t in g.nodes():
        if s != t:
            for targetID in track_uncertainty[s].keys():
                goodness_value = goodness(s, t, targetID)
                goodness_dict[(s, t, targetID)] = goodness_value

# Define the LP problem
prob = pulp.LpProblem("Maximize_Goodness", pulp.LpMaximize)

# Define the fixed bandwidth consumption per data transfer
fixed_bandwidth_consumption = 30

# Create binary decision variables for each (source, target, targetID) combination
selection_vars = pulp.LpVariable.dicts(
    "selection", goodness_dict.keys(), 0, 1, pulp.LpBinary
)

# Objective function: Maximize the total goodness for selected pairs
prob += pulp.lpSum(
    selection_vars[(s, t, targetID)] * goodness_dict[(s, t, targetID)]
    for (s, t, targetID) in goodness_dict
)

# Ensure the total bandwidth consumption does not exceed the bandwidth constraints
for (s, t, _) in goodness_dict:
    prob += pulp.lpSum(
        selection_vars[(s, t, targetID)] * fixed_bandwidth_consumption
        for targetID in track_uncertainty[s]
    ) <= g[s][t]["bandwidth"]

# Solve the problem
prob.solve()

# Extract and interpret results
selected_pairs = [
    (s, t, targetID)
    for (s, t, targetID) in goodness_dict
    if selection_vars[(s, t, targetID)].value() == 1
]

# Print the selected pairs
print("Selected pairs:")
for (s, t, targetID) in selected_pairs:
    print(f"{s} -> {t} for targetID {targetID}")

# Print hte total goodness value
total_goodness = sum(
    goodness_dict[(s, t, targetID)]
    for (s, t, targetID) in selected_pairs
)

print(f"Total goodness: {total_goodness}")

# Now sum up the total bandwidth usage
total_bandwidth_usage = sum(
    fixed_bandwidth_consumption
    for (s, t, _) in selected_pairs
)

print(f"Total bandwidth usage: {total_bandwidth_usage}")
# Also figure out how much bandwidth is left
total_bandwidth = sum(
    g[u][v]['bandwidth']
    for (u, v) in g.edges()
)
print(f"Total bandwidth available: {total_bandwidth}")