import pandas as pd
from lxml import etree
import traci

# BPR parameters
a = 0.15
b = 4
C = 1800  # vehicles/hour
FLOW_DURATION = 300  # seconds (1 hour)

JUNCTION_IDS_TO_PROCESS = [3]  # Filter list

def compute_volume(t_f, t_obs):
    if t_obs <= t_f:
        return 0
    return C * ((t_obs / t_f - 1) / a) ** (1 / b)

def generate_flow_route_file_bpr(df, route_cache, timestamp_column="duration_20250327_1730", route_file="generated_bpr.rou.xml", begin_time=0):
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

    # Filter by Junction_id
    df = df[df["Junction_id"].isin(JUNCTION_IDS_TO_PROCESS)]

    for idx, row in df.iterrows():
        rid = f"route_{idx+1}"
        from_edge = row["from_edge"]
        to_edge = row["to_edge"]

        key = (from_edge, to_edge)
        if key not in route_cache:
            try:
                route_cache[key] = traci.simulation.findRoute(from_edge, to_edge).edges
            except Exception as e:
                print(f"Error finding route from {from_edge} to {to_edge}: {e}")
                continue

    for idx, row in df.iterrows():
        rid = f"route_{idx+1}"
        from_edge = row["from_edge"]
        to_edge = row["to_edge"]
        t_f = row["duration_without_traffic"]
        t_obs = row[timestamp_column]

        vehicle_count = compute_volume(t_f, t_obs)
        int_vehicle_count = int(vehicle_count*FLOW_DURATION/3600)

        edges = route_cache.get((from_edge, to_edge), [])
        if not edges or int_vehicle_count == 0:
            continue

        etree.SubElement(root, "route", id=rid, edges=" ".join(edges))
        etree.SubElement(root, "flow", id=rid, type="car", route=rid, begin=str(begin_time),
                         end=str(begin_time + FLOW_DURATION), number=str(int_vehicle_count),
                         departPos="random", arrivalPos="random")

    tree = etree.ElementTree(root)
    tree.write(route_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")


SUMO_BINARY = "sumo"
SUMO_CONFIG = "config/osm.sumocfg"
traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "-r", "generated_bpr.rou.xml",
             "--tripinfo-output", "tripinfo.xml",
             "--start", "--step-length", str(1)])

df = pd.read_csv("data/final_with_all_data.csv")
route_cache = {}
generate_flow_route_file_bpr(df, route_cache)
traci.close()