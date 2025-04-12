import pandas as pd
import traci
import logging
from lxml import etree

# === CONFIGURATION ===
SUMO_BINARY = "sumo-gui"  # or "sumo"
SUMO_CONFIG = "osm.sumocfg"
ROUTE_CSV = "traffic_calibration/road_load.csv"
ROUTE_FILE = "generated_flows.rou.xml"
STEP_LENGTH = 1
FLOW_DURATION = 1  # seconds over which vehicles are generated

# === Calibration Parameters ===
MAX_ATTEMPTS = 100
TOLERANCE = 0.15
LARGE_DIFF_THRESHOLD = 20
TOO_LARGE_DIFF_THRESHOLD = 30
TOO_LARGE_ADJUSTMENT = 5
LARGE_ADJUSTMENT = 2
SMALL_ADJUSTMENT = 1

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.info("🚀 Starting traffic simulation calibration script...")

# === Load CSV
logging.info(f"📥 Loading routes from '{ROUTE_CSV}'...")
df = pd.read_csv(ROUTE_CSV)
routes = {}
route_cache = {}

logging.info(f"🔢 Preparing initial route data for {len(df)} rows...")
for idx, row in df.iterrows():
    route_id = f"route_{idx}"
    target_duration = row["duration_seconds"]
    vehicle_count = 30
    logging.info(f"🛣 Initialized {route_id} | Duration: {target_duration}s | Vehicles: {vehicle_count}")
    routes[route_id] = {
        "origin": row["Origin"],
        "destination": row["Destination"],
        "from_edge": row["from_edge"],
        "to_edge": row["to_edge"],
        "target_duration": target_duration,
        "vehicle_count": vehicle_count,
        "last_duration": None,
        "converged": False
    }

# === Helper: Generate .rou.xml with flows
def generate_flow_route_file(routes, route_cache):
    logging.info("📝 Generating flow-based .rou.xml file with random start/end positions...")
    root = etree.Element("routes")
    etree.SubElement(root, "vType", id="car", accel="1.0", decel="4.5", length="5", maxSpeed="16.6", sigma="0.5")

    for rid, info in routes.items():
        if info["converged"]:
            logging.info(f"⏩ Skipping converged route: {rid}")
        key = (info["from_edge"], info["to_edge"])
        if key not in route_cache:
            try:
                logging.info(f"🧭 Calculating route path for {rid} from {key[0]} to {key[1]}")
                route_cache[key] = traci.simulation.findRoute(info["from_edge"], info["to_edge"]).edges
            except Exception as e:
                logging.error(f"❌ Route calculation failed for {rid}: {e}")
                continue

        edges = route_cache[key]
        etree.SubElement(root, "route", id=rid, edges=" ".join(edges))

        etree.SubElement(root, "flow",
                         id=rid,
                         type="car",
                         route=rid,
                         begin="0",
                         end=str(FLOW_DURATION),
                         period=f"exp({info["vehicle_count"]})",
                         departPos="random",
                         arrivalPos="random")
        logging.info(f"🧩 Added flow for {rid} with {info['vehicle_count']} vehicles")

    tree = etree.ElementTree(root)
    tree.write(ROUTE_FILE, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    logging.info(f"✅ Flow route file written to '{ROUTE_FILE}'")

# === Calibration Loop
for attempt in range(1, MAX_ATTEMPTS + 1):
    logging.info(f"\n========================")
    logging.info(f"🔁 GLOBAL ATTEMPT #{attempt}")
    logging.info(f"========================")

    logging.info("📌 Running dry simulation for route generation...")
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(STEP_LENGTH)])
    generate_flow_route_file(routes, route_cache)
    traci.close()

    logging.info("🚦 Starting SUMO simulation with route flows...")
    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CONFIG,
        "-r", ROUTE_FILE,
        "--start",
        "--step-length", str(STEP_LENGTH)
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
    logging.info("🛑 SUMO simulation ended.")

    # === Analyze results
    durations_by_route = {}
    total_difference = 0

    for veh_id, vdata in vehicle_data.items():
        if "start" in vdata and "end" in vdata:
            rid = vdata["route_id"]
            dur = vdata["end"] - vdata["start"]
            durations_by_route.setdefault(rid, []).append(dur)

    converged_count = 0
    for rid, durs in durations_by_route.items():
        avg_duration = sum(durs) / len(durs)
        routes[rid]["last_duration"] = avg_duration
        target = routes[rid]["target_duration"]
        diff = avg_duration - target
        lower = target * (1 - TOLERANCE)
        upper = target * (1 + TOLERANCE)

        logging.info(f"⏱ {rid} | Simulated: {avg_duration:.2f}s | Target: {target:.2f}s | Vehicles: {routes[rid]['vehicle_count']}")
        total_difference += abs(diff)
        if lower <= avg_duration <= upper:
            logging.info(f"✅ {rid} converged.")
            routes[rid]["converged"] = True
            converged_count += 1
        else:
            if abs(diff) >= (target * (TOO_LARGE_DIFF_THRESHOLD / 100)):
                adjust = TOO_LARGE_ADJUSTMENT
            elif abs(diff) >= (target * (LARGE_DIFF_THRESHOLD / 100)):
                adjust = LARGE_ADJUSTMENT
            else:
                adjust = SMALL_ADJUSTMENT

            if diff < 0:
                routes[rid]["vehicle_count"] += adjust
                logging.info(f"📈 Increasing vehicles for {rid} by {adjust}")
            else:
                new_count = max(1, routes[rid]["vehicle_count"] - adjust)
                logging.info(f"📉 Decreasing vehicles for {rid} by {adjust}")
                routes[rid]["vehicle_count"] = new_count

    # Save intermediate results

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
    output_file = f"simulated_vs_real_attempt_{attempt}.csv"
    results_df.to_csv(output_file, index=False)

    logging.info(f"💾 Intermediate results saved to {output_file}")
    logging.info(f"✅ Converged: {converged_count} / {len(routes)}")
    logging.info(f"🔍 Total Difference: {total_difference:.2f}")

    if all(info["converged"] for info in routes.values()):
        logging.info("🎉 All routes converged. Stopping calibration.")
        break
else:
    logging.warning("⚠️ Max attempts reached. Some routes did not converge.")

# Final results
final_file = "simulated_vs_real_global_calibrated.csv"
results_df.to_csv(final_file, index=False)
logging.info(f"📁 Final results saved to '{final_file}'")
