import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree
from pyproj import Transformer

# Load points.csv
points_df = pd.read_csv("data/points.csv")

# Extract lat/lon and convert to UTM
points_df[['latitude', 'longitude']] = points_df['coordinate'].str.split(',', expand=True).astype(float)
transformer = Transformer.from_crs("epsg:4326", "epsg:32638", always_xy=True)
points_df['x'], points_df['y'] = transformer.transform(points_df['longitude'].values, points_df['latitude'].values)

# Build XML
nodes_root = Element("nodes")
for _, row in points_df.iterrows():
    SubElement(
        nodes_root,
        "node",
        id=row["key"],
        x=str(row["x"]),
        y=str(row["y"]),
        type="priority"  # We'll treat them as regular points for now
    )

# Save
tree = ElementTree(nodes_root)
tree.write("xml/nodes.xml", encoding="utf-8", xml_declaration=True)
print("✅ nodes.xml generated")
