import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree

# STEP 1: Load the data
df_durations = pd.read_csv("data/result_10_minutes.csv", sep=';')
df_edges = pd.read_csv("data/routes_with_edges.csv")

# STEP 2: Prepare duration columns and lookup
duration_cols = [col for col in df_durations.columns if col.startswith('duration_')]
df = df_durations[['Origin', 'Destination', 'Direction'] + duration_cols]

edge_lookup = {
    (row['Origin'], row['Destination']): [row['from_edge'], row['to_edge']]
    for _, row in df_edges.iterrows()
}

origin_map = {}
for _, row in df.iterrows():
    origin_map.setdefault(row['Origin'], []).append((row['Destination'], row))

# STEP 3: Build chains
routes = []

def build_chains(start, path, visited, max_depth=3):
    if len(path) >= max_depth:
        return
    for next_dest, next_row in origin_map.get(start, []):
        if next_dest in visited:
            continue
        new_path = path + [next_row]
        routes.append(new_path)
        build_chains(next_dest, new_path, visited | {next_dest}, max_depth)

for origin in origin_map:
    build_chains(origin, [], {origin})

# STEP 4: Create chained route entries
chained_data = []

for path in routes:
    if len(path) < 2:
        continue
    try:
        segments = [(r['Origin'], r['Destination']) for r in path]
        edge_ids = []
        for o, d in segments:
            edge_pair = edge_lookup.get((o, d))
            if not edge_pair:
                raise ValueError(f"Missing edge for ({o}, {d})")
            edge_ids.extend(edge_pair)

        chained_row = {
            "from": path[0]['Origin'],
            "to": path[-1]['Destination'],
            "intermediate_nodes": [r['Destination'] for r in path[:-1]],
            "length": len(path),
            "segments": segments,
            "edges": edge_ids
        }

        for col in duration_cols:
            chained_row[col] = sum([r[col] for r in path if pd.notna(r[col])])

        chained_data.append(chained_row)

    except Exception as e:
        print(f"Skipping route: {e}")

# Save to chained_routes.csv
chained_df = pd.DataFrame(chained_data)
chained_df.to_csv("data/chained_routes.csv", index=False)
print("[✓] chained_routes.csv created.")


# STEP 5: Generate chained_routes.rou.xml for SUMO
def generate_sumo_flows_from_chained_df(df, output_file="chained_routes.rou.xml", car_ratio=0.9, headway=2.5):
    time_columns = [col for col in df.columns if col.startswith("duration_")]
    routes = Element("routes")

    for time_index, time_col in enumerate(time_columns):
        begin_time = time_index * 600  # 10-minute windows

        for i, row in df.iterrows():
            try:
                duration = float(row[time_col])
                distance = duration * 10  # Estimate distance (10 m/s average)

                flow_id = f"{time_col}_flow{i+1}"
                dist_id = f"mixed_{flow_id}"
                car_id = f"car_{flow_id}"
                bus_id = f"bus_{flow_id}"

                avg_speed = distance / duration
                car_speed = round(avg_speed * 1.1, 2)
                bus_speed = round(avg_speed * 0.95, 2)
                vehicle_count = max(1, int(duration / headway))

                # vTypeDistribution
                vtype_dist = SubElement(routes, "vTypeDistribution", {"id": dist_id})
                SubElement(vtype_dist, "vType", {
                    "id": car_id,
                    "accel": "1.0",
                    "decel": "4.5",
                    "sigma": "0.5",
                    "length": "5",
                    "maxSpeed": str(car_speed),
                    "guiShape": "passenger",
                    "probability": str(car_ratio)
                })
                SubElement(vtype_dist, "vType", {
                    "id": bus_id,
                    "accel": "0.8",
                    "decel": "4.0",
                    "sigma": "0.5",
                    "length": "12",
                    "maxSpeed": str(bus_speed),
                    "guiShape": "bus",
                    "probability": str(1 - car_ratio)
                })

                # Route edge string
                edge_str = ' '.join(eval(row['edges']))

                SubElement(routes, "flow", {
                    "id": flow_id,
                    "type": dist_id,
                    "begin": str(begin_time),
                    "end": str(begin_time + duration),
                    "number": str(vehicle_count),
                    "route": edge_str,
                    "departPos": "random",
                    "arrivalPos": "random"
                })

            except Exception as e:
                print(f"Skipping flow {time_col}_flow{i+1}: {e}")

    tree = ElementTree(routes)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"[✓] Generated {output_file}")

# Run the flow generation
generate_sumo_flows_from_chained_df(chained_df, output_file="chained_routes.rou.xml")
