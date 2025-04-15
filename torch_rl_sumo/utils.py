import numpy as np
import pandas as pd
import torch
import traci
from lxml import etree

SUMO_BINARY = "sumo"
SUMO_CONFIG = "osm.sumocfg"
ROUTE_FILE = "generated_flows.rou.xml"
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
            duration = row["duration"]
            distance = row["distance"]
            batch.append((from_idx, to_idx, duration, distance, row["from_edge"], row["to_edge"]))
        return batch


def scale_action_to_vehicle_count(action_tensor):
    return (((action_tensor + 1) / 2) * 299 + 1).int()


def write_flows_and_run_sumo(batch, vehicle_counts, route_cache):
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "-r", ROUTE_FILE, "--start", "--step-length", str(STEP_LENGTH)])
    vehicle_counts = vehicle_counts.unsqueeze(0)
    for i, (from_idx, to_idx, duration, distance, from_edge, to_edge) in enumerate(batch):
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
                         number=str(max(1, int(vehicle_counts[0][i]))),
                         departPos="random",
                         arrivalPos="random",
                         departSpeed="max",
                         departLane="best")

    etree.ElementTree(root).write(ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    vehicle_start_times = {}
    vehicle_end_times = {}
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        for veh_id in traci.vehicle.getIDList():
            if veh_id not in vehicle_start_times:
                vehicle_start_times[veh_id] = traci.simulation.getTime()
            vehicle_end_times[veh_id] = traci.simulation.getTime()
    traci.close()

    total_durations = []
    for i in range(len(batch)):
        route_id = f"route_{i}"
        route_duration = 0
        route_vehicle_count = 0
        for veh_id in vehicle_start_times:
            if route_id in veh_id and veh_id in vehicle_end_times:
                route_duration += vehicle_end_times[veh_id] - vehicle_start_times[veh_id]
                route_vehicle_count += 1
        total_durations.append(torch.tensor(route_duration/route_vehicle_count) if route_vehicle_count > 0 else 9999.0)
    return torch.stack(total_durations)
