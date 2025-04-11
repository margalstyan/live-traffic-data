import pandas as pd
import traci

# === CONFIGURATION ===
SUMO_BINARY = "sumo"  # or "sumo"
SUMO_CONFIG = "osm.sumocfg"
ROUTE_CSV = "prepared_routes_1689f7ce.csv"
STEP_LENGTH = 1

# === Calibration Parameters ===
MAX_ATTEMPTS = 10
VEHICLE_START_COUNT = 10
VEHICLE_STEP = 5
TOLERANCE = 0.05

# === Load routes
df = pd.read_csv(ROUTE_CSV)

# === Initialize route calibration state
routes = {}
for idx, row in df.iterrows():
    route_id = f"route_{idx}"
    routes[route_id] = {
        "origin": row["Origin"],
        "destination": row["Destination"],
        "from_edge": row["from_edge"],
        "to_edge": row["to_edge"],
        "target_duration": row["duration_seconds"],
        "vehicle_count": VEHICLE_START_COUNT,
        "last_duration": None,
        "converged": False
    }

# === Global Calibration Loop
for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\n========================")
    print(f"🔁 GLOBAL ATTEMPT #{attempt}")
    print(f"========================")

    # Start SUMO for this round
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--step-length", str(STEP_LENGTH)])

    vehicle_data = {}

    # Add routes and vehicles
    for route_id, info in routes.items():
        if info["converged"]:
            continue

        try:
            route_edges = traci.simulation.findRoute(info["from_edge"], info["to_edge"]).edges
            traci.route.add(route_id, route_edges)

            for i in range(info["vehicle_count"]):
                veh_id = f"{route_id}_veh_{i}_a{attempt}"
                traci.vehicle.add(veh_id, route_id)
                vehicle_data[veh_id] = {
                    "route_id": route_id,
                    "start": None,
                    "end": None,
                }

        except Exception as e:
            print(f"❌ Failed to add route {route_id}: {e}")
            routes[route_id]["converged"] = True  # mark as done to skip

    # Run simulation
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

    traci.close()

    # Analyze and update vehicle counts
    durations_by_route = {}

    for veh_id, vdata in vehicle_data.items():
        if vdata["start"] is not None and vdata["end"] is not None:
            route_id = vdata["route_id"]
            duration = vdata["end"] - vdata["start"]
            durations_by_route.setdefault(route_id, []).append(duration)

    for route_id, durations in durations_by_route.items():
        avg_duration = sum(durations) / len(durations)
        target = routes[route_id]["target_duration"]
        lower = target * (1 - TOLERANCE)
        upper = target * (1 + TOLERANCE)

        print(f"⏱ {route_id} | Simulated: {avg_duration:.2f}s | Target: {target:.2f}s")

        routes[route_id]["last_duration"] = avg_duration

        if lower <= avg_duration <= upper:
            print(f"✅ {route_id} converged.")
            routes[route_id]["converged"] = True
        elif avg_duration < lower:
            routes[route_id]["vehicle_count"] += VEHICLE_STEP
        else:
            routes[route_id]["vehicle_count"] = max(1, routes[route_id]["vehicle_count"] - VEHICLE_STEP)

    # Check if all converged
    if all(info["converged"] for info in routes.values()):
        print("\n✅ All routes converged. Calibration complete.")
        break
else:
    print("\n⏹ Max attempts reached. Some routes may not have converged.")

# === Save results
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

results_df.to_csv("simulated_vs_real_global_calibrated.csv", index=False)
print("\n📁 Results saved to 'simulated_vs_real_global_calibrated.csv'")
