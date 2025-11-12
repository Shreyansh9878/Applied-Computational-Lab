import numpy as np
import matplotlib.pyplot as plt

def read_xvg_file(file_path):
    data = np.loadtxt(file_path, comments=['#', '@'])
    return data[:, 1]

T = 300
k_B = 1.380649e-23

Ps = [0.1, 1, 10, 100, 1000, 10000]
Kps = []
Cvs = []

for P in Ps:
    volume_data = read_xvg_file(f'npt_fluctuations/npt{P}/volume.xvg')
    energy_data = read_xvg_file(f'npt_fluctuations/npt{P}/energy.xvg')

    avg_volume = np.mean(volume_data)
    avg_volume_2 = np.mean((volume_data-avg_volume)**2)

    avg_energy = np.mean(energy_data)
    avg_energy_2 = np.mean((energy_data-avg_energy)**2)

    Kp = avg_volume_2/((avg_volume**2) * k_B * T)
    Cv = avg_energy_2 / (k_B * (T**2))

    Kps.append(Kp)
    Cvs.append(Cv)

# Plot Compressibility vs Pressure
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(Ps, Kps, label='Compressibility (κ)', marker='o', color='b')
plt.xscale('log')  # Log scale for pressure
plt.xlabel('Pressure (P)')
plt.ylabel('Compressibility (κ)')
plt.title(f'Compressibility vs Pressure at {T}K')
plt.grid(True)

# Plot Specific Heat vs Pressure
plt.subplot(1, 2, 2)
plt.plot(Ps, Cvs, label='Specific Heat (C_V)', marker='s', color='r')
plt.xscale('log')  # Log scale for pressure
plt.xlabel('Pressure (P)')
plt.ylabel('Specific Heat (C_V)')
plt.title(f'Specific Heat vs Pressure at {T}K')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()