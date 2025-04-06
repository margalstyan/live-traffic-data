import osmnx as ox

# Download drivable road network for Yerevan
graph = ox.graph_from_place("Yerevan, Armenia", network_type="drive")

# Save as OSM XML for SUMO
ox.save_graphml(graph, "yerevan.graphml")
ox.save_graphml(ox.project_graph(graph), "yerevan_projected.graphml")
print("✅ Yerevan road network downloaded and saved as OSM XML.")