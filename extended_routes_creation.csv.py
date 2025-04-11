import time

import pandas as pd
import traci
import logging
import os

from networkx import difference

# === CONFIGURATION ===
SUMO_BINARY = "sumo-gui"  # or "sumo-gui"
SUMO_CONFIG = "osm.sumocfg"
ROUTE_CSV = "prepared_routes_1689f7ce.csv"
STEP_LENGTH = 1

# === Calibration Parameters ===
MAX_ATTEMPTS = 100
VEHICLE_START_COUNT = 25
TOLERANCE = 0.15
LARGE_DIFF_THRESHOLD = 20
TOO_LARGE_DIFF_THRESHOLD = 50
TOO_LARGE_ADJUSTMENT = 10
LARGE_ADJUSTMENT = 5
SMALL_ADJUSTMENT = 2

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

logging.info("📥 Loading routes from CSV...")
df = pd.read_csv(ROUTE_CSV)

logging.info("🔧 Initializing route calibration state...")
routes = {}


for idx, row in df.iterrows():
    route_id = f"route_{idx}"
    target_duration = row["duration_seconds"]
    vehicle_count = int(target_duration / 2) if target_duration > 100 else int(target_duration)
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

# === Global Calibration Loop ===
route_cache = {}  # cache declared outside the loop

for attempt in range(1, MAX_ATTEMPTS + 1):
    logging.info(f"========================")
    logging.info(f"🔁 GLOBAL ATTEMPT #{attempt}")
    logging.info(f"========================")

    logging.info("🚦 Starting SUMO simulation...")
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(STEP_LENGTH)])
    time.sleep(10)
    vehicle_data = {}

    logging.info("📦 Caching route paths and adding vehicles...")
    for route_id, info in routes.items():
        if info["converged"]:
            continue
        try:
            key = (info["from_edge"], info["to_edge"])
            if key not in route_cache:
                route_cache[key] = traci.simulation.findRoute(info["from_edge"], info["to_edge"]).edges
            route_edges = route_cache[key]
            traci.route.add(route_id, route_edges)
            logging.info(f"➕ Added route {route_id} with {info['vehicle_count']} vehicles")

            for i in range(info["vehicle_count"]):
                veh_id = f"{route_id}_veh_{i}_a{attempt}"
                traci.vehicle.add(veh_id, route_id)
                vehicle_data[veh_id] = {
                    "route_id": route_id,
                    "start": None,
                    "end": None,
                }
        except Exception as e:
            logging.error(f"❌ Failed to add route {route_id}: {e}")
            routes[route_id]["converged"] = True

    logging.info("▶️ Entering simulation step loop...")
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        for veh_id in vehicle_data:
            if vehicle_data[veh_id]["start"] is None and veh_id in traci.vehicle.getIDList():
                vehicle_data[veh_id]["start"] = traci.simulation.getTime()
            elif (
                vehicle_data[veh_id]["end"] is None
                and veh_id not in traci.vehicle.getIDList()
                and vehicle_data[veh_id]["start"] is not None
            ):
                vehicle_data[veh_id]["end"] = traci.simulation.getTime()

    logging.info("🛑 Closing SUMO simulation...")
    traci.close()

    logging.info("📊 Analyzing simulation results...")
    durations_by_route = {}
    total_difference = 0
    total_routes = len(routes)
    converged_count = 0

    for veh_id, vdata in vehicle_data.items():
        if vdata["start"] is not None and vdata["end"] is not None:
            route_id = vdata["route_id"]
            duration = vdata["end"] - vdata["start"]
            durations_by_route.setdefault(route_id, []).append(duration)

    for route_id, durations in durations_by_route.items():
        avg_duration = sum(durations) / len(durations)
        target = routes[route_id]["target_duration"]
        diff = avg_duration - target
        lower = target * (1 - TOLERANCE)
        upper = target * (1 + TOLERANCE)

        logging.info(f"⏱ {route_id} | Simulated: {avg_duration:.2f}s | Target: {target:.2f}s")

        routes[route_id]["last_duration"] = avg_duration
        total_difference += abs(diff)

        if lower <= avg_duration <= upper:
            logging.info(f"✅ {route_id} converged.")
            routes[route_id]["converged"] = True
            converged_count += 1
        else:
            if abs(diff) >= (target * (TOO_LARGE_DIFF_THRESHOLD / 100)):
                adjust = TOO_LARGE_ADJUSTMENT
            elif abs(diff) >= (target * (LARGE_DIFF_THRESHOLD / 100)):
                adjust = LARGE_ADJUSTMENT
            else:
                adjust = SMALL_ADJUSTMENT
            if diff < 0:
                routes[route_id]["vehicle_count"] += adjust
                logging.info(f"⬆️ Increasing vehicles on {route_id} by {adjust} to {routes[route_id]['vehicle_count']}")
            else:
                new_count = max(1, routes[route_id]["vehicle_count"] - adjust)
                logging.info(f"⬇️ Decreasing vehicles on {route_id} by {adjust} to {new_count}")
                routes[route_id]["vehicle_count"] = new_count

    convergence_percentage = (converged_count / total_routes) * 100
    logging.info(f"📈 Convergence: {convergence_percentage:.2f}%")
    logging.info(f"🔍 Total Difference: {total_difference:.2f}")

    # Save intermediate CSV
    logging.info("💾 Saving intermediate results to CSV...")
    results_df = pd.DataFrame([
        {
            "route_id": rid,
            "origin": info["origin"],
            "destination": info["destination"],
            "real_duration": info["target_duration"],
            "simulated_duration": info["last_duration"],
            "vehicle_count": info["vehicle_count"],
            "converged": info["converged"],
        }
        for rid, info in routes.items()
    ])
    output_file = f"simulated_vs_real_attempt_{attempt}.csv"
    results_df.to_csv(output_file, index=False)
    logging.info(f"💾 Intermediate results saved to '{output_file}'")

    if all(info["converged"] for info in routes.values()):
        logging.info("✅ All routes converged. Calibration complete.")
        break
else:
    logging.warning("⏹ Max attempts reached. Some routes may not have converged.")

# Final save
logging.info("📁 Saving final results...")
final_file = "simulated_vs_real_global_calibrated.csv"
results_df.to_csv(final_file, index=False)
logging.info(f"📁 Final results saved to '{final_file}'")
