import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import numpy as np

# Full list of atom data provided by user
raw_data = """
Na,0.00,0.00,0.00
Na,0.00,2.82,2.82
Na,2.82,0.00,2.82
Na,2.82,2.82,0.00
Na,0.00,0.00,5.64
Na,0.00,2.82,8.46
Na,2.82,0.00,8.46
Na,2.82,2.82,5.64
Na,0.00,5.64,0.00
Na,0.00,8.46,2.82
Na,2.82,5.64,2.82
Na,2.82,8.46,0.00
Na,0.00,5.64,5.64
Na,0.00,8.46,8.46
Na,2.82,5.64,8.46
Na,2.82,8.46,5.64
Cl,2.82,2.82,2.82
Cl,2.82,0.00,0.00
Cl,0.00,2.82,0.00
Cl,0.00,0.00,2.82
Cl,2.82,2.82,8.46
Cl,2.82,0.00,5.64
Cl,0.00,2.82,5.64
Cl,0.00,0.00,8.46
Cl,5.64,2.82,2.82
Cl,5.64,0.00,0.00
Cl,2.82,5.64,0.00
Cl,2.82,2.82,5.64
Cl,5.64,2.82,8.46
Cl,5.64,0.00,5.64
Cl,2.82,5.64,5.64
Cl,2.82,2.82,8.46
"""

# Parse data
elements = []
coords = []

for line in raw_data.strip().split('\n'):
    parts = line.split(',')
    elements.append(parts[0])
    coords.append([float(x) for x in parts[1:]])

coords = np.array(coords)

# Split into Na and Cl for connection purposes
na_coords = coords[:16]
cl_coords = coords[16:]

# Use KDTree to find Na-Cl pairs within 3 Å
tree_cl = cKDTree(cl_coords)
bonds = []
for na in na_coords:
    dists, idxs = tree_cl.query(na, k=6, distance_upper_bound=2.18)
    for dist, j in zip(dists, idxs):
        if dist < 3 and j < len(cl_coords):
            bonds.append((na, cl_coords[j]))

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot Na atoms
ax.scatter(*na_coords.T, color='blue', label='Na', s=50)
# Plot Cl atoms
ax.scatter(*cl_coords.T, color='green', label='Cl', s=50)

# Draw bonds
for start, end in bonds:
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='gray', lw=1)

ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")
ax.set_title("NaCl 2×2×2 Supercell with Na–Cl Bonds (< 3 Å)")
ax.legend()
plt.tight_layout()
plt.show()
