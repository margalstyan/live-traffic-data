import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree
from pyproj import Transformer

# Load the junctions data from Google API
junctions_df = pd.read_csv("data/junctions_from_google.csv")

# Convert WGS84 lat/lon to UTM (for Yerevan: EPSG:32638)
transformer = Transformer.from_crs("epsg:4326", "epsg:32638", always_xy=True)

# Transform coordinates
junctions_df["x"], junctions_df["y"] = transformer.transform(
    junctions_df["lon_junction"].values,
    junctions_df["lat_junction"].values
)

# Create the XML root element
nodes_root = Element("nodes")

# Add each junction as a traffic light node
for _, row in junctions_df.iterrows():
    SubElement(
        nodes_root,
        "node",
        id=row["junction_id"],
        x=str(row["x"]),
        y=str(row["y"]),
        type="traffic_light"
    )

# Save to XML file
output_path = "xml/nodes_from_google.xml"
tree = ElementTree(nodes_root)
tree.write(output_path, encoding="utf-8", xml_declaration=True)

print(f"✅ Saved nodes file to {output_path}")
