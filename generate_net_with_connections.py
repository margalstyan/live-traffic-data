import subprocess

cmd = [
    "netconvert",
    "--node-files", "xml/nodes.xml",
    "--edge-files", "xml/edges.xml",
    "--output-file", "xml/network.net.xml",
    "--no-turnarounds"
]

try:
    subprocess.run(cmd, check=True)
    print("✅ network.net.xml and connections.xml generated.")
except Exception as e:
    print(f"❌ Error: {e}")
