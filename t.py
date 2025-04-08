import numpy as np

# Step 1: Input your known data
# Format: [lon, lat]
gps_coords = np.array([
    [44.514210, 40.186265],  # Point 1
    [44.516015, 40.187747],  # Point 2
])

# Format: [x, y] from SUMO shape midpoint
sumo_coords = np.array([
    [117.34, 1248.36],  # Corresponding SUMO x, y for Point 1
    [216.43, 1359.45],  # Corresponding SUMO x, y for Point 2
])

# Step 2: Add a bias term for affine transformation
A = np.hstack((gps_coords, np.ones((gps_coords.shape[0], 1))))
B = sumo_coords

# Step 3: Solve for transformation matrix (least squares)
X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

# Step 4: Extract coefficients
a11, a12, b1 = X[0, 0], X[1, 0], X[2, 0]
a21, a22, b2 = X[0, 1], X[1, 1], X[2, 1]

# Print the equations
print("Transformation equations:")
print(f"x = {a11:.2f} * lon + {a12:.2f} * lat + {b1:.2f}")
print(f"y = {a21:.2f} * lon + {a22:.2f} * lat + {b2:.2f}")


def gps_to_sumo(lon, lat):
    x = 542005.77 * lon + -593272.88 * lat + -285419.97
    y = 607420.01 * lon + -664846.91 * lat + -319859.49
    return x, y

x, y = gps_to_sumo(40.182868, 44.519888)
print(f"SUMO x, y = {x:.2f}, {y:.2f}")



import sumolib

net = sumolib.net.readNet("osm.net.xml")

edge = net.getEdge("137406408#0")  # replace with your actual edge ID
shape = edge.getShape()  # list of (x, y) tuples along the edge

# Calculate midpoint
mid_index = len(shape) // 2
mid_point = shape[mid_index]
print("Midpoint coordinates:", mid_point)
