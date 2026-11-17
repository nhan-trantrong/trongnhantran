import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Provided values for the first surface
x1 = [5, 31, 153, 125, 92, 66, 4]
y1 = [4, 7, 6, 3, 5, 2, 4]
z1 = [31, 153, 6, 125, 92, 66, 4]

# Provided values for the second surface
x2 = [4, 21, 133, 115, 102, 76, 18]
y2 = [4, 16, 26, 13, 35, 12, 54]
z2 = [61, 53, 26, 155, 72, 16, 4]

# Create meshgrid for the first surface
xi1 = np.linspace(min(x1), max(x1), 50)
yi1 = np.linspace(min(y1), max(y1), 50)
X1, Y1 = np.meshgrid(xi1, yi1)
Z1 = griddata((x1, y1), z1, (X1, Y1), method='cubic')

# Create meshgrid for the second surface
xi2 = np.linspace(min(x2), max(x2), 50)
yi2 = np.linspace(min(y2), max(y2), 50)
X2, Y2 = np.meshgrid(xi2, yi2)
Z2 = griddata((x2, y2), z2, (X2, Y2), method='cubic')

# Create a new figure
fig = plt.figure()

# Add a subplot for the first surface
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(X1, Y1, Z1, cmap='viridis', alpha=0.7)
surf2 = ax.plot_surface(X2, Y2, Z2, cmap='plasma', alpha=0.7)

# Set axis labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Add a color bar which maps values to colors
fig.colorbar(surf1)
fig.colorbar(surf2)

# Display the plot
plt.show()
