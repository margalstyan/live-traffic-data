import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree

# Load road distances with durations
df = pd.read_csv("road_distances.csv")

# Calculate speeds in m/s (distance / duration), avoid divide-by-zero
df["speed"] = (df["Distance_m"] / df["Duration_s"]).clip(lower=1)

# Create XML root
edges_root = Element("edges")

for _, row in df.iterrows():
    SubElement(edges_root, "edge", attrib={
        "id": f"{row['Origin']}__to__{row['Destination']}",
        "from": row["Origin"],
        "to": row["Destination"],
        "priority": "1",
        "type": "car",
        "speed": str(round(row["speed"], 2)),
        "length": str(round(row["Distance_m"], 1))
    })

# Write to edges.xml
tree = ElementTree(edges_root)
tree.write("edges.xml", encoding="utf-8", xml_declaration=True)
print("✅ edges.xml generated successfully")
