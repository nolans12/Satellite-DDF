##### Goal: Use pulp linear programming to optimize the best paths of CI communication between satellites

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
    "Sat1a": {1: 90, 2: 115, 3: 120, 4: 120, 5: 120},
    "Sat1b": {1: 105, 2: 130, 3: 130, 4: 135, 5: 135},
    "Sat2a": {1: 150, 2: 155, 3: 161, 4: 162, 5: 165},
    "Sat2b": {1: 50, 2: 275, 3: 280, 4: 280, 5: 290}
}

nx.set_node_attributes(g, track_uncertainty, 'trackUncertainty')

edgeBandwidth = 90

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

# Show the graph:
# plt.show()

#### NOW OPTIMIZE USING LINEAR PROGRAMMING

# Redefine goodness function to be based on a source and reciever node pair, not a path:
def goodness(source, reciever, track_uncertainty, targetID):
    """ A paths goodness is defined as the sum of the deltas in track uncertainty on a targetID, as far as that node hasnt already recieved data from that satellite"""

    if source not in track_uncertainty or reciever not in track_uncertainty:
        return 0

    # Get the track uncertainty of the source node
    sourceTrackUncertainty = track_uncertainty[source][targetID]

    # Get the track uncertainty of the target node
    recieverTrackUncertainty = track_uncertainty[reciever][targetID]

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

# Calculate goodness for all individual links between nodes
for source in g.nodes():
    for reciever in g.nodes():
        if source == reciever:
            continue
        for targetID in track_uncertainty[source].keys():
            # Calculate the goodness of the path
            goodness_dict[(source, reciever, targetID)] = goodness(source, reciever, track_uncertainty, targetID)


# Now goal is to find the set of paths that maximize the total goodness, while also respecting the bandwidth constraints

# Generate all possible paths up to a reasonable length (e.g., max 3 hops)
def generate_all_paths(graph, max_hops):
    paths = []
    for source in graph.nodes():
        for target in graph.nodes():
            if source != target:
                for path in nx.all_simple_paths(graph, source=source, target=target, cutoff=max_hops):
                    paths.append(tuple(path))
    return paths

all_paths = generate_all_paths(g, 3)

# Create binary decision variables for each path combination
path_selection_vars = pulp.LpVariable.dicts(
    "path_selection", [(path, targetID) for path in all_paths for targetID in track_uncertainty[path[0]].keys()], 0, 1, pulp.LpBinary
)

# Create binary decision variables to track if a receiver has already received information about a targetID from a specific source
information_vars = pulp.LpVariable.dicts(
    "information", [(source, receiver, targetID) for source in g.nodes() for receiver in g.nodes() for targetID in track_uncertainty[source].keys()], 0, 1, pulp.LpBinary
)

# Define the optimization problem
prob = pulp.LpProblem("Path_Optimization", pulp.LpMaximize)

# Define the fixed bandwidth consumption per data transfer
fixed_bandwidth_consumption = 30

# Objective: Maximize the total goodness across all paths, considering the goodness of all links
prob += pulp.lpSum([
    sum(
        goodness(path[i], path[i+1], track_uncertainty, targetID) * path_selection_vars[(path, targetID)]
        for i in range(len(path) - 1)
    )
    for path in all_paths for targetID in track_uncertainty[path[0]].keys()
])

# Constraints to ensure that a receiver only receives information from a source once per targetID
for source in g.nodes():
    for targetID in track_uncertainty[source].keys():
        for receiver in g.nodes():
            if receiver != source:
                prob += pulp.lpSum(
                    path_selection_vars[(path, targetID)]
                    for path in all_paths
                    if path[0] == source and receiver in path
                ) <= 1, f"Single_path_for_target_{source}_{receiver}_{targetID}"

# Ensure the total bandwidth consumption does not exceed the bandwidth constraints
for edge in g.edges():
    u, v = edge  # unpack the edge
    prob += (
        pulp.lpSum(
            path_selection_vars[(path, targetID)] * fixed_bandwidth_consumption
            for (path, targetID) in path_selection_vars
            if any((path[i], path[i+1]) == edge for i in range(len(path) - 1))
        ) <= g[u][v]['bandwidth'],
        f"Bandwidth_constraint_{edge}"
    )

# Solve the problem
prob.solve()

# Output the results, paths selected
selected_paths = [
    (path, targetID)
    for (path, targetID) in path_selection_vars
    if path_selection_vars[(path, targetID)].value() == 1
]

# Print the selected paths
print("Selected paths:")
for (path, targetID) in selected_paths:
    # also print the total goodness of the selected paths
    total_goodness = sum(
        goodness(path[i], path[i+1], track_uncertainty, targetID)
        for i in range(len(path) - 1)
    )
    print(f"{path} for targetID {targetID}, total goodness: {total_goodness}")


test = 1