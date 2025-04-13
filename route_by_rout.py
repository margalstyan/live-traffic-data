import pandas as pd
import traci
import logging
from lxml import etree

# === CONFIGURATION ===
SUMO_BINARY = "sumo"
SUMO_CONFIG = "osm.sumocfg"
ROUTE_CSV = "traffic_calibration/road_load.csv"
ROUTE_FILE = "generated_flows.rou.xml"
STEP_LENGTH = 1
FLOW_DURATION = 100

# === Calibration Parameters ===
MAX_ATTEMPTS_PER_STEP = 50
MAX_VEHICLE_THRESHOLD = 100
TOLERANCE = 0.15
LARGE_DIFF_THRESHOLD = 20
TOO_LARGE_DIFF_THRESHOLD = 30
TOO_LARGE_ADJUSTMENT = 5
LARGE_ADJUSTMENT = 2
SMALL_ADJUSTMENT = 1

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.info("🚀 Starting stepwise traffic simulation calibration...")

# === Load CSV
df = pd.read_csv(ROUTE_CSV)
routes = {}
route_cache = {}

for idx, row in df.iterrows():
    route_id = f"route_{idx}"
    routes[route_id] = {
        "origin": row["Origin"],
        "destination": row["Destination"],
        "from_edge": row["from_edge"],
        "to_edge": row["to_edge"],
        "target_duration": row["duration_seconds"],
        "vehicle_count": 30,
        "last_duration": None,
        "converged": False
    }
    logging.info(f"🔧 Initialized {route_id}: {row['Origin']} → {row['Destination']}, Target={row['duration_seconds']}s")

# === Helper: Generate .rou.xml file
def generate_flow_route_file(active_route_ids):
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

    for rid in active_route_ids:
        info = routes[rid]
        if info["converged"]:
            logging.info(f"⏩ Skipping converged route {rid}")
        key = (info["from_edge"], info["to_edge"])
        if key not in route_cache:
            logging.info(f"🔍 Finding route edges for {rid}: {key}")
            try:
                if traci.isLoaded():
                    traci.close()
                traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(STEP_LENGTH), "--no-warnings"])
                route_cache[key] = traci.simulation.findRoute(info["from_edge"], info["to_edge"]).edges
                traci.close()
                logging.info(f"✅ Route edges for {rid}: {' → '.join(route_cache[key])}")
            except Exception as e:
                logging.error(f"❌ Route {rid} lookup failed: {e}")
                continue

        edges = route_cache[key]
        etree.SubElement(root, "route", id=rid, edges=" ".join(edges))

        veh_count = max(1, info["vehicle_count"])
        etree.SubElement(root, "flow",
                         id=rid,
                         type="car",
                         route=rid,
                         begin="0",
                         end=str(FLOW_DURATION),
                         number=str(veh_count),
                         departPos="random",
                         arrivalPos="random")

        logging.info(f"📤 Added flow for {rid}: {veh_count} vehicles")

    etree.ElementTree(root).write(ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    logging.info(f"📄 Wrote flow route file with {len(active_route_ids)} active routes.")

# === Stepwise Calibration
for step in range(1, len(routes) + 1):
    active_route_ids = [f"route_{i}" for i in range(step)]
    logging.info(f"\n🔷 Starting Step {step}/{len(routes)} | Routes: {active_route_ids}")

    for attempt in range(1, MAX_ATTEMPTS_PER_STEP + 1):
        logging.info(f"\n🔁 Attempt {attempt} for Step {step}")
        generate_flow_route_file(active_route_ids)

        traci.start([
            SUMO_BINARY,
            "-c", SUMO_CONFIG,
            "-r", ROUTE_FILE,
            "--start",
            "--step-length", str(STEP_LENGTH),
            "--no-warnings"
        ])

        vehicle_data = {}
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            for veh_id in traci.vehicle.getIDList():
                if veh_id not in vehicle_data:
                    vehicle_data[veh_id] = {
                        "start": traci.simulation.getTime(),
                        "route_id": traci.vehicle.getRouteID(veh_id)
                    }
            for veh_id in list(vehicle_data):
                if veh_id not in traci.vehicle.getIDList() and vehicle_data[veh_id].get("end") is None:
                    vehicle_data[veh_id]["end"] = traci.simulation.getTime()

        traci.close()

        # === Analyze results
        durations_by_route = {}
        for veh_id, vdata in vehicle_data.items():
            if "start" in vdata and "end" in vdata:
                rid = vdata["route_id"]
                dur = vdata["end"] - vdata["start"]
                durations_by_route.setdefault(rid, []).append(dur)

        converged_count = 0
        for rid in active_route_ids:
            if rid not in durations_by_route:
                logging.warning(f"⚠️ No data collected for {rid}")
                continue
            durs = durations_by_route[rid]
            avg_duration = sum(durs) / len(durs)
            routes[rid]["last_duration"] = avg_duration
            target = routes[rid]["target_duration"]
            diff = avg_duration - target
            lower = target * (1 - TOLERANCE)
            upper = target * (1 + TOLERANCE)

            logging.info(f"⏱ {rid}: Expected={target:.2f}s | Simulated={avg_duration:.2f}s | Diff={diff:.2f}s | Vehicles={routes[rid]['vehicle_count']}")

            if lower <= avg_duration <= upper:
                routes[rid]["converged"] = True
                logging.info(f"✅ {rid} converged.")
                converged_count += 1
            else:
                if abs(diff) >= (target * (TOO_LARGE_DIFF_THRESHOLD / 100)):
                    adjust = TOO_LARGE_ADJUSTMENT
                elif abs(diff) >= (target * (LARGE_DIFF_THRESHOLD / 100)):
                    adjust = LARGE_ADJUSTMENT
                else:
                    adjust = SMALL_ADJUSTMENT

                if diff < 0:
                    if routes[rid]["vehicle_count"] >= MAX_VEHICLE_THRESHOLD:
                        logging.warning(f"🚦 {rid} reached MAX_VEHICLE_THRESHOLD ({routes[rid]['vehicle_count']}), penalizing it")
                        routes[rid]["vehicle_count"] = 20
                        same_dest_routes = [
                            alt_rid for alt_rid, alt in routes.items()
                            if alt_rid != rid and alt["to_edge"] == routes[rid]["to_edge"] and not alt["converged"]
                        ]
                        logging.info(f"Alternative route options {same_dest_routes}")

                        if same_dest_routes:
                            for route_id in same_dest_routes:
                                if route_id not in active_route_ids:
                                    active_route_ids.append(route_id)
                                    split_target = route_id
                                    routes[split_target]["vehicle_count"] += adjust
                                    logging.info(f"🔀 Redirected {adjust} vehicles to {split_target} (also ends at {routes[rid]['to_edge']})")
                        else:
                            logging.info(f"⚠️ No alternative routes to {routes[rid]['to_edge']} found. Vehicle count remains capped.")
                    else:
                        routes[rid]["vehicle_count"] += adjust
                        logging.info(f"📈 Increasing {rid} vehicles by {adjust} → {routes[rid]['vehicle_count']}")
                else:
                    routes[rid]["vehicle_count"] = max(1, routes[rid]["vehicle_count"] - adjust)
                    logging.info(f"📉 Decreasing {rid} vehicles by {adjust} → {routes[rid]['vehicle_count']}")

        logging.info(f"✅ Converged routes in this step: {converged_count}/{len(active_route_ids)}")
        if all(routes[rid]["converged"] for rid in active_route_ids):
            logging.info(f"🎉 Step {step} finished. All included routes converged.\n")
            break
    else:
        logging.warning(f"❌ Step {step} failed to converge after {MAX_ATTEMPTS_PER_STEP} attempts.")

# Save final results
results_df = pd.DataFrame([
    {
        "route_id": rid,
        "origin": info["origin"],
        "destination": info["destination"],
        "real_duration": info["target_duration"],
        "simulated_duration": info["last_duration"],
        "vehicle_count": info["vehicle_count"],
        "converged": info["converged"]
    }
    for rid, info in routes.items()
])
results_df.to_csv("simulated_vs_real_stepwise.csv", index=False)
logging.info("📁 Final results saved to simulated_vs_real_stepwise.csv")
