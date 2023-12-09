import matplotlib.pyplot as plt

# Example scatter plot with different markers
markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'H', 'D', '+', 'x']

for i, marker in enumerate(markers):
    plt.scatter(i, i, marker=marker, label=f'Marker {marker}')

plt.legend()
#plt.title('Scatter Plot with Different Markers')
#plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Example scatter plot with varying marker sizes
x = np.arange(10)
y = np.arange(10)
sizes = np.arange(10, 110, 10)

plt.scatter(x, y, s=sizes, marker='o', label='Markers with Varying Size')
plt.legend()
plt.title('Scatter Plot with Varying Marker Size')
plt.show()
