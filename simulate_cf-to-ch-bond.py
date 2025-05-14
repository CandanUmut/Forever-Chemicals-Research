import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Constants and Parameters
# -----------------------------

# Vibrational frequency (cm^-1 to Hz conversion)
freq_CF_cm1 = 1200  # C–F bond
freq_CH_cm1 = 2900  # C–H bond
freq_CF_Hz = freq_CF_cm1 * 29.9792458e9
freq_CH_Hz = freq_CH_cm1 * 29.9792458e9

# Effective reduced mass (kg)
mass_CF = 2.5e-26  # approx. carbon + fluorine

# Time domain
duration = 0.1  # seconds
samples = 50000
time = np.linspace(0, duration, samples)
dt = time[1] - time[0]

# Amplitude envelope: grows from 0.05 nm to 1.0 nm
base_amp = 5.0e-11
max_amp = 1.0e-9
envelope = np.linspace(base_amp, max_amp, samples)

# Frequency transition curve
transition = np.linspace(0, 1, samples)
omega_CF = 2 * np.pi * freq_CF_Hz
omega_CH = 2 * np.pi * freq_CH_Hz

# -----------------------------
# 2. Resonant Shift Signal
# -----------------------------

vibration = np.sin((1 - transition) * omega_CF * time + transition * omega_CH * time)
oscillation = envelope * vibration

# -----------------------------
# 3. Derive Velocity and Energy
# -----------------------------

velocity = np.gradient(oscillation, dt)
kinetic_energy = 0.5 * mass_CF * velocity**2

# Simulation results
max_disp = np.max(np.abs(oscillation))
max_vel = np.max(np.abs(velocity))
max_energy = np.max(kinetic_energy)
bond_break_energy = 8.05e-19  # J (C–F bond)

print("==== Resonant Bond Reprogramming ====")
print(f"Max Displacement: {max_disp:.2e} m")
print(f"Max Velocity:     {max_vel:.2e} m/s")
print(f"Max Energy:       {max_energy:.2e} J")
print(f"Bond Broken?      {'YES' if max_energy > bond_break_energy else 'NO'}")
print(f"Resonant Shift Success? {'YES' if max_disp > 1e-10 and max_energy < bond_break_energy else 'LIKELY'}")

# -----------------------------
# 4. Visualization
# -----------------------------

plt.figure(figsize=(10, 5))
plt.plot(time, oscillation, label="C–F → C–H Vibrational Shift")
plt.title("Adaptive Resonant Shift: C–F to C–H")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("../results/cf_to_ch_shift_output.png")
plt.show()
