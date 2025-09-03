import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import matplotlib as mpl
import scienceplots
plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]
plt.rcParams["text.usetex"] = False

mpl.rc("font", size=15)

# --- Constants ---
C = 50000  # Capacitance [F]
T_eclipse = 10  # Eclipse duration [s] (30 minutes)
LC_slow = (T_eclipse * 0.1)**2  # Make LC >> eclipse duration
L = LC_slow / C  # Inductance [H]
R0 = 0.1         # Baseline resistance [Ohm]
delta_R = 0.2    # Eclipse-induced increase in resistance
L, C = 50, 4

# Initial conditions
V0 = 100e3  # Initial voltage across capacitor [V]
I0 = 60e3  # Initial inductor current [A]

# --- Eclipse-modulated resistance R(t) ---
def R_t(t, ecl=True):
    if ecl:
        if (t<=-0.5*T_eclipse) or (t>=0.5*T_eclipse):
            return R0
        else:
            return R0 + delta_R * np.cos(np.pi * t / T_eclipse)**2
    else: return R0

# --- ODE system ---
# y[0] = i(t), y[1] = v_C(t)
def rlc_dynamics(t, y):
    i, vc = y
    Rt = R_t(t)
    di_dt = (vc - Rt * i) / L
    dvc_dt = -i / C
    return [di_dt, dvc_dt]

# --- Time span and solver ---
t_span = (-1*T_eclipse, 1*T_eclipse)
t_eval = np.linspace(*t_span, 2*T_eclipse+1)
y0 = [I0, V0]
# print(t_eval, [R_t(t) for t in t_eval])

sol = solve_ivp(rlc_dynamics, t_span, y0, t_eval=t_eval, method='RK45')

# --- Extract solutions ---
t = sol.t
i = sol.y[0]
vc = sol.y[1]
R_vals = np.array([R_t(tx) for tx in t])
vr = R_vals * i
vl = L * np.gradient(i, t)
power_r = vr * i  # Instantaneous power dissipated

# --- Plot Vc and VR ---
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
# Current
axs[0].plot(t / 60, i / 1e3, color='purple')
axs[0].set_ylabel('Current i(t) [kA]')
axs[0].set_title('Current, Voltages, Resistance, and Power Dissipation in Eclipse-Modulated RLC')
axs[0].grid(True)

# Voltages
axs[1].plot(t / 60, vc / 1e3, label='Vc(t)', color='blue')
# axs[1].plot(t / 60, vr / 1e3, label='Vr(t)', color='green')
# axs[1].plot(t / 60, vl / 1e3, label='Vl(t)', color='orange')
axs[1].set_ylabel('Voltage [kV]')
axs[1].legend()
axs[1].grid(True)

# Resistance
axs[2].plot(t / 60, R_vals, color='black')
axs[2].set_ylabel('R(t) [Ω]')
axs[2].grid(True)

# Power Dissipation
axs[3].plot(t / 60, power_r / 1e6, color='red')
axs[3].set_ylabel('Power in R(t) [MW]')
axs[3].set_xlabel('Time [minutes]')
axs[3].grid(True)


plt.tight_layout()
plt.savefig("figures_2021_Special/model.png")