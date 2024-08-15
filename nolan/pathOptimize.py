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
        return 0 # EX: If i have uncertainty of 200 and share it with a sat with 100, theres no benefit to sharing that
    
    # Else, return the goodness of the link, difference between the two track uncertainties
    return recieverTrackUncertainty - sourceTrackUncertainty

# Define a dictionary to store the goodness values
goodness_dict = {}

# Calculate goodness for all individual links between nodes
for source in g.nodes():
    for reciever in g.nodes():
        if source == reciever:
            continue
        for targetID in track_uncertainty[source].keys():
            # Calculate the goodness of the link
            goodness_dict[(source, reciever, targetID)] = goodness(source, reciever, track_uncertainty, targetID)


# Now goal is to find the set of paths that maximize the total goodness, while also respecting the bandwidth constraints and not double counting, farying information is allowed

# Generate all possible non cyclic paths up to a reasonable length (e.g., max 3 hops)
def generate_all_paths(graph, max_hops):
    paths = []
    for source in graph.nodes():
        for target in graph.nodes():
            if source != target:
                for path in nx.all_simple_paths(graph, source=source, target=target, cutoff=max_hops):
                    paths.append(tuple(path))
    return paths

all_paths = generate_all_paths(g, 4)

# Define the fixed bandwidth consumption per data transfer
fixed_bandwidth_consumption = 30

# Define the optimization problem
prob = pulp.LpProblem("Path_Optimization", pulp.LpMaximize)

# Create binary decision variables for each path combination
# 1 if the path is selected, 0 otherwise
path_selection_vars = pulp.LpVariable.dicts(
    "path_selection", [(path, targetID) for path in all_paths for targetID in track_uncertainty[path[0]].keys()], 0, 1, pulp.LpBinary
)

# # Create binary decision variables to track if a receiver has already received information about a targetID from a specific source
# # 1 if the receiver sat has already received information about targetID from that satellite, 0 otherwise
# information_vars = pulp.LpVariable.dicts(
#     "information", [(source, receiver, targetID) for source in g.nodes() for receiver in g.nodes() for targetID in track_uncertainty[source].keys()], 0, 1, pulp.LpBinary
# )

#### OBJECTIVE FUNCTION

## Maximize the total goodness across all paths, considering the goodness of all links
prob += pulp.lpSum([
    sum(
        # Important to note: taking goodness from first sat to all other nodes, because assume information only coming from one source
        goodness(path[0], path[i+1], track_uncertainty, targetID) * path_selection_vars[(path, targetID)] 
        for i in range(len(path) - 1)
    )
    for path in all_paths for targetID in track_uncertainty[path[0]].keys()
])

#### CONSTRAINTS

## Ensure the total bandwidth consumption across a link does not exceed the bandwidth constraints
for edge in g.edges(): # Loop through all edges possible
    u, v = edge  # Unpack the edge

    # Create a list to accumulate the terms for the total bandwidth usage on this edge
    bandwidth_usage_terms = []
    
    # Iterate over all possible paths
    for (path, targetID) in path_selection_vars:

        # Check if the current path includes the edge in question
        if any((path[i], path[i+1]) == edge for i in range(len(path) - 1)):

            # Now path_selection_vars is a binary expression/condition will either be 0 or 1
            # Thus, the following term essentially accounts for the bandwidth usage on this edge, if its used
            bandwidth_usage = path_selection_vars[(path, targetID)] * fixed_bandwidth_consumption

            # Add the term to the list
            bandwidth_usage_terms.append(bandwidth_usage)
    
    # Sum all the expressions in the list to get the total bandwidth usage on this edge
    total_bandwidth_usage = pulp.lpSum(bandwidth_usage_terms)

    # Add the constraint to the linear programming problem
    # The constraint indicates that the total bandwidth usage on this edge should not exceed the bandwidth constraint
    # This constraint will be added for all edges in the graph after the loop
    prob += total_bandwidth_usage <= g[u][v]['bandwidth'], f"Bandwidth_constraint_{edge}"
    

## Ensure the reward for sharing information about a targetID from source node to another node is not double counted
for source in g.nodes(): # Loop over all source nodes possible

    for receiver in g.nodes(): # Loop over all receiver nodes possible

        # If the receiver is not the source node, we can add a constraint
        if receiver != source:
                
            # Loop over all targetIDs source and reciever could be talking about
            for targetID in track_uncertainty[source].keys():

                # Initalize a linear expression that will be used as a constraint
                # This expression is exclusivly for source -> reciever about targetID, gets reinitalized every time
                path_count = pulp.LpAffineExpression()

                # Now we want to add the constraint that no more than 1 
                # source -> reciever about targetID is selected

                # So we will count the number of paths that could be selected that are source -> reciever about targetID
                for path in all_paths:

                    # Check if the path starts at the source and contains the receiver
                    if path[0] == source and receiver in path:

                        # Add the path selection variable to the path sum if its selected and talking about the targetID
                        path_count += path_selection_vars[(path, targetID)]
                
                # Add a constraint to ensure the path sum is at most 1
                # Thus, there will be a constraint for every source -> reciever about targetID combo, 
                # ensuring the total number of paths selected that contain that isn't greater than 1
                prob += path_count <= 1, f"Single_path_for_target_{source}_{receiver}_{targetID}"

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
        goodness(path[0], path[i+1], track_uncertainty, targetID)
        for i in range(len(path) - 1)
    )
    print(f"{path} for targetID {targetID}, total goodness: {total_goodness}")

# Print the total goodness of all paths:
total_goodness = sum(
    sum(
        goodness(path[0], path[i+1], track_uncertainty, targetID)
        for i in range(len(path) - 1)
    )
    for (path, targetID) in selected_paths
)
print(f"Total goodness: {total_goodness}")

# Now take the select paths, and count the total bandwidth usage across all links in the graph, and print them
total_bandwidth_usage = sum(
    fixed_bandwidth_consumption
    for (path, targetID) in selected_paths
    for i in range(len(path) - 1)
)

print(f"Total bandwidth usage: {total_bandwidth_usage}")
# Also figure out how much bandwidth is left
total_bandwidth = sum(
    g[u][v]['bandwidth']
    for (u, v) in g.edges()
)
print(f"Total bandwidth available: {total_bandwidth}")

