# === Updated main.py ===

import os
from sumo.utils import  *

junctions_csv = "../data/junctions.csv"
points_csv = "../data/points.csv"
routes_csv = "../data/routes.csv"

nodes = parse_csv_to_nodes(junctions_csv)
visual_nodes, visual_edges = generate_visual_edges_and_nodes(junctions_csv, points_csv)

write_nodes(nodes + visual_nodes)
write_edges(visual_edges)


print(nodes + visual_nodes)
print(visual_edges)

os.system(f"bash netconvert.sh")
os.system(f"python generate_connections.py")
os.system(f"bash run_sumo.sh")
