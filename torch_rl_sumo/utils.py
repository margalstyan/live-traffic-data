import pandas as pd
import traci
from lxml import etree
import os
import torch

SUMO_BINARY = "sumo"
SUMO_CONFIG_TEMPLATE = "osm.sumocfg"
ROUTE_FILE_TEMPLATE = "flows/generated_flows_{}.rou.xml"
FLOW_DURATION = 30
STEP_LENGTH = 1

class RouteDataset:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.edge_vocab = list(pd.concat([self.df["from_edge"], self.df["to_edge"]]).unique())
        self.edge_to_idx = {edge: i for i, edge in enumerate(self.edge_vocab)}

    def sample_batch(self, batch_size):
        sampled = self.df.sample(batch_size)
        batch = []
        for _, row in sampled.iterrows():
            from_idx = self.edge_to_idx[row["from_edge"]]
            to_idx = self.edge_to_idx[row["to_edge"]]
            duration = row["duration_seconds"]
            batch.append((from_idx, to_idx, duration, row["from_edge"], row["to_edge"]))
        return batch

def scale_action_to_vehicle_count(action_tensor):
    action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
    scaled = (((action_tensor + 1) / 2) * 299 + 1)
    safe_scaled = torch.clamp(scaled, min=1.0, max=300.0)  # safer bounds
    return safe_scaled.int()

def write_flows_and_run_sumo(batch, vehicle_counts, route_cache, run_id=0):
    route_file = ROUTE_FILE_TEMPLATE.format(run_id)
    config_file = SUMO_CONFIG_TEMPLATE.replace(".sumocfg", f"_{run_id}.sumocfg")

    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5",
                     maxSpeed="16.6", sigma="0.5")
    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CONFIG_TEMPLATE,  # optionally replace with config_file if you create temp ones
        "-r", route_file,
        "--start",
        "--step-length", str(STEP_LENGTH)
    ])
    for i, (count, (from_idx, to_idx, duration, from_edge, to_edge)) in enumerate(zip(vehicle_counts, batch)):
        route_key = (from_edge, to_edge)
        if route_key not in route_cache:
            route = traci.simulation.findRoute(from_edge, to_edge)
            route_cache[route_key] = route.edges

        edges = route_cache[route_key]
        route_id = f"route_{i}"
        etree.SubElement(root, "route", id=route_id, edges=" ".join(edges))
        etree.SubElement(root, "flow",
                         id=f"{route_id}_flow",
                         type="car",
                         route=route_id,
                         begin="0",
                         end=str(FLOW_DURATION),
                         number=str(count.item()),
                         departPos="random",
                         arrivalPos="random",
                         departSpeed="max",
                         departLane="best")

    etree.ElementTree(root).write(route_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")



    vehicle_start_times = {}
    vehicle_end_times = {}

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        for veh_id in traci.vehicle.getIDList():
            if veh_id not in vehicle_start_times:
                vehicle_start_times[veh_id] = traci.simulation.getTime()
            vehicle_end_times[veh_id] = traci.simulation.getTime()

    traci.close()

    route_durations = [0.0] * len(batch)
    route_counts = [0] * len(batch)

    for veh_id in vehicle_start_times:
        if veh_id in vehicle_end_times:
            for i in range(len(batch)):
                if f"route_{i}" in veh_id:
                    dur = vehicle_end_times[veh_id] - vehicle_start_times[veh_id]
                    route_durations[i] += dur
                    route_counts[i] += 1
                    break

    avg_durations = [
        route_durations[i] / route_counts[i] if route_counts[i] > 0 else 9999.0
        for i in range(len(batch))
    ]

    # print average durations for each route
    for i, duration in enumerate(avg_durations):
        print(f"Route {i}: Average Duration: {duration:.2f} seconds | Target Duration: {batch[i][2]} seconds | Vehicle Count: {vehicle_counts[i].item()}")
    return avg_durations
