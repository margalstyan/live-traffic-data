import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree


def add_flow_from_csv_row(routes_elem, flow_id, from_edge, to_edge, distance, duration, begin, car_ratio=0.9, headway=2.5):
    car_id = f"car_{flow_id}"
    bus_id = f"bus_{flow_id}"
    dist_id = f"mixed_{flow_id}"

    avg_speed = float(distance) / float(duration)
    car_speed = round(avg_speed * 1.1, 2)
    bus_speed = round(avg_speed * 0.95, 2)

    vehicle_count = max(1, int(float(duration) / headway))

    car_length = 5
    bus_length = 12

    vtype_dist = SubElement(routes_elem, "vTypeDistribution", {"id": dist_id})

    SubElement(vtype_dist, "vType", {
        "id": car_id,
        "accel": "1.0",
        "decel": "4.5",
        "sigma": "0.5",
        "length": str(car_length),
        "maxSpeed": str(car_speed),
        "guiShape": "passenger",
        "probability": str(car_ratio)
    })

    SubElement(vtype_dist, "vType", {
        "id": bus_id,
        "accel": "0.8",
        "decel": "4.0",
        "sigma": "0.5",
        "length": str(bus_length),
        "maxSpeed": str(bus_speed),
        "guiShape": "bus",
        "probability": str(1 - car_ratio)
    })

    SubElement(routes_elem, "flow", {
        "id": flow_id,
        "type": dist_id,
        "begin": str(begin),
        "end": str(begin + duration),
        "number": str(vehicle_count),
        "from": from_edge,
        "to": to_edge,
        "departPos": "random",
        "arrivalPos": "random"
    })


def create_combined_routes(csv_path="data/result_10_minutes.csv", distance_map_path="data/routes_with_edges.csv",
                           output_file="all_routes.rou.xml", car_ratio=0.9, headway=3.5):
    traffic_df = pd.read_csv(csv_path, sep=';')
    edge_df = pd.read_csv(distance_map_path)

    merged_df = traffic_df.merge(edge_df[['Origin', 'Destination', 'from_edge', 'to_edge', 'distance']],
                                 how='left', on=['Origin', 'Destination'])

    # Identify all columns that match duration pattern
    duration_cols = [col for col in merged_df.columns if col.startswith("duration_2025")]

    if not duration_cols:
        print("[!] No valid duration columns found.")
        return

    routes_elem = Element("routes")

    # Iterate over each time step (every 10 minutes = 600s)
    for step_index, col in enumerate(duration_cols):
        begin_time = step_index * 600

        for i, row in merged_df.iterrows():
            try:
                duration = float(row[col])
                if pd.isna(duration) or pd.isna(row['distance']):
                    continue

                flow_id = f"{col}_flow{i+1}"

                add_flow_from_csv_row(
                    routes_elem=routes_elem,
                    flow_id=flow_id,
                    from_edge=row["from_edge"],
                    to_edge=row["to_edge"],
                    distance=float(row["distance"]),
                    duration=duration,
                    begin=begin_time,
                    car_ratio=car_ratio,
                    headway=headway
                )
            except Exception as e:
                print(f"Skipping row {i}, time {col}, error: {e}")

    tree = ElementTree(routes_elem)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"[✓] Created single file: {output_file} with {len(duration_cols)} time intervals.")


# Example usage
if __name__ == "__main__":
    create_combined_routes()
