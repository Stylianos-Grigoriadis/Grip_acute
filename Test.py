import numpy as np
import matplotlib.pyplot as plt

# Example: your original signal (100 points)
x_original = np.linspace(0, 10, 100)       # Original x values
y_original = np.sin(x_original)            # Example signal (replace with your own)

# Create new x values (1000 points)
x_new = np.linspace(x_original[0], x_original[-1], 1000)

# Perform linear interpolation
y_new = np.interp(x_new, x_original, y_original)
print(len(y_new))
print(len(y_original))
# Plot to check
plt.figure(figsize=(8, 4))
plt.scatter(x_original, y_original, label='Original (100 pts)', s=100)
plt.scatter(x_new, y_new, label='Upsampled (1000 pts)', s=1)
plt.legend()
plt.xlabel("X")
plt.ylabel("Signal")
plt.title("Linear Interpolation (Resampling from 100 → 1000 points)")
plt.show()
