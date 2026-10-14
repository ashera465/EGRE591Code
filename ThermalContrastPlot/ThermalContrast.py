import numpy as np
import matplotlib.pyplot as plt

# Constants
epsilon = 1.0          # Emissivity
sigma = 5.670374419e-8 # Stefan-Boltzmann constant [W/m^2/K^4]

# Scene temperature range [K]
T = np.linspace(200, 350, 500)

# dM/dt = 4 * epsilon * sigma * T^3
dM_dT = 4 * epsilon * sigma * T**3

# Plot
plt.figure(figsize=(8, 5))

plt.plot(T, dM_dT, linewidth=2)

plt.xlabel("Scene Temperature (K)")
plt.ylabel(r"$dM/dT$ (W m$^{-2}$ K$^{-1}$)")
plt.title(r"Thermal Sensitivity vs. Scene Temperature: $dM/dT = 4\epsilon\sigma T^3$")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()