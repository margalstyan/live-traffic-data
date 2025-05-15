import pandas as pd
import random
from lxml import etree
import os

# === Configuration ===
SUMO_CONFIG = "osm.sumocfg"
ROUTE_CSV = "traffic_calibration/road_load.csv"
ROUTE_FILE = "generated_flows_1.rou.xml"
FLOW_DURATION = 60
MAX_RANDOM_START = 90
TOTAL_VEHICLES_PER_DEST = 60  # you can tune this per to_edge if needed

# === Load Route CSV ===
df = pd.read_csv(ROUTE_CSV)

# === Group by Destination ===
to_edge_groups = df.groupby("to_edge")

# === XML Root for Flows ===
routes_el = etree.Element("routes")
flow_id_counter = 0

for to_edge, group in to_edge_groups:
    durations = group["duration_seconds"].tolist()
    from_edges = group["from_edge"].tolist()

    inv_durations = [1 / d for d in durations]
    inv_sum = sum(inv_durations)
    proportions = [inv / inv_sum for inv in inv_durations]
    vehicle_counts = [round(p * TOTAL_VEHICLES_PER_DEST) for p in proportions]

    for from_edge, vehicle_count in zip(from_edges, vehicle_counts):
        begin = random.randint(0, MAX_RANDOM_START)
        end = begin + FLOW_DURATION

        flow_el = etree.Element("flow", {
            "id": f"route_{flow_id_counter}",
            "from": from_edge,
            "to": to_edge,
            "begin": str(begin),
            "end": str(end),
            "number": str(vehicle_count)
        })

        routes_el.append(flow_el)
        flow_id_counter += 1

# === Save XML ===
tree = etree.ElementTree(routes_el)
tree.write(ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")

print(f"✅ Generated {flow_id_counter} flow entries in {ROUTE_FILE}")
