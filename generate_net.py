import os
import subprocess

# Set input/output filenames
nodes_file = "nodes_utm.xml"
edges_file = "edges_utm.xml"
output_file = "xml/network.net.xml"
# Path to SUMO's netconvert (adjust this if needed)
netconvert_cmd = "netconvert"

# Build command
cmd = [
    netconvert_cmd,
    "--node-files", nodes_file,
    "--edge-files", edges_file,
    "--output-file", output_file,
    "--no-turnarounds",        # Optional: avoids 180-degree U-turns
]

# Run
try:
    subprocess.run(cmd, check=True)
    print(f"✅ Generated: {output_file}")
except subprocess.CalledProcessError as e:
    print(f"❌ netconvert failed with error: {e}")
except FileNotFoundError:
    print("❌ netconvert not found. Make sure SUMO is installed and in your PATH.")
