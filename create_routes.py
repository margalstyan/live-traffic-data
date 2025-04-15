import csv
from xml.etree.ElementTree import Element, SubElement, ElementTree


def add_flow_from_csv_row(routes_elem, flow_id, from_edge, to_edge, distance, duration, car_ratio=0.9, headway=2.5/3.0):
    """
    Adds a vTypeDistribution and flow for a given row of CSV with distance/duration.
    Speed is calculated from distance and duration.
    Number of vehicles is estimated from headway.
    """
    car_id = f"car_{flow_id}"
    bus_id = f"bus_{flow_id}"
    dist_id = f"mixed_{flow_id}"

    # Calculate average speed
    avg_speed = float(distance) / float(duration)  # in m/s
    car_speed = round(avg_speed * 1.1, 2)          # cars go slightly faster
    bus_speed = round(avg_speed * 0.95, 2)          # buses a bit slower

    # Estimate number of vehicles based on headway
    vehicle_count = max(1, int(float(duration) / headway))  # ensure at least 1

    # Default lengths
    car_length = 5
    bus_length = 12

    # Create vTypeDistribution
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

    # Add flow
    SubElement(routes_elem, "flow", {
        "id": flow_id,
        "type": dist_id,
        "begin": "0",
        "end": str(duration),
        "number": str(vehicle_count),
        "from": from_edge,
        "to": to_edge,
        "departPos": "random",
        "arrivalPos": "random"
    })


def create_routes_from_csv(csv_path="traffic_calibration/road_load.csv", output_file="routes.rou.xml", car_ratio=0.9, headway=2.5):
    routes = Element("routes")

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            flow_id = f"flow{i+1}"
            add_flow_from_csv_row(
                routes_elem=routes,
                flow_id=flow_id,
                from_edge=row["from_edge"],
                to_edge=row["to_edge"],
                distance=float(row["distance"]),
                duration=float(row["duration"]),
                car_ratio=car_ratio,
                headway=headway
            )

    tree = ElementTree(routes)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"[✓] Created {output_file} based on {csv_path}")


# Example usage
if __name__ == "__main__":
    create_routes_from_csv()
