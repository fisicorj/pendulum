import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st
from io import BytesIO
from matplotlib.colors import Normalize

# === Page configuration ===
st.set_page_config(page_title="Pendulum Simulator", layout="wide")
st.title("🎯 Interactive Simple Pendulum Simulator")

# === Sidebar parameters ===
st.sidebar.header("Pendulum Parameters")
g = st.sidebar.slider("Gravity (m/s²)", 1.0, 20.0, 9.81, 0.01)
L = st.sidebar.slider("Rod length (m)", 0.1, 5.0, 1.0, 0.1)
m = 1.0

# Initial conditions
theta0_deg = st.sidebar.slider("Initial angle θ₀ (degrees)", -180, 180, 30)
omega0_deg = st.sidebar.slider("Initial angular velocity ω₀ (degrees/s)", -360, 360, 0)
theta0 = np.radians(theta0_deg)
omega0 = np.radians(omega0_deg)
t_eval = np.linspace(0, 10, 200)  # Reduced number of frames for memory efficiency

# === Cached solver ===
@st.cache_resource
def solve_pendulum(g, L, theta0, omega0, t_eval):
    def pendulum(t, y):
        return [y[1], - (g / L) * np.sin(y[0])]
    return solve_ivp(pendulum, (0, 10), [theta0, omega0], t_eval=t_eval)

# === Solve ODE ===
sol = solve_pendulum(g, L, theta0, omega0, t_eval)
theta = sol.y[0]
omega = sol.y[1]

# === Harmonic solution ===
omega_natural = np.sqrt(g / L)
theta_harm = theta0 * np.cos(omega_natural * t_eval)

# === Layout with columns ===
col1, col2 = st.columns(2)

# === θ(t) plot ===
with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(t_eval, np.degrees(theta), label='Numerical solution')
    ax1.plot(t_eval, np.degrees(theta_harm), '--', label='Harmonic approximation')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle θ (degrees)')
    ax1.set_title('θ(t)')
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

# === Phase space with separatrix, vectors, and energy ===
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 5))

    # Vector field
    theta_vals = np.linspace(-180, 180, 40)
    omega_vals = np.linspace(-360, 360, 40)
    T, W = np.meshgrid(np.radians(theta_vals), np.radians(omega_vals))
    dT = W
    dW = - (g / L) * np.sin(T)
    magnitude = np.hypot(dT, dW)
    dT_unit = dT / magnitude
    dW_unit = dW / magnitude
    ax2.quiver(np.degrees(T), np.degrees(W), dT_unit, dW_unit, magnitude,
               cmap='coolwarm', scale=30, alpha=0.6, width=0.003)

    # Separatrix
    theta_sep = np.linspace(-np.pi, np.pi, 500)
    omega_sep = np.sqrt(2 * g / L * (1 - np.cos(theta_sep)))
    ax2.plot(np.degrees(theta_sep), np.degrees(omega_sep), 'r--', lw=2, label="Separatrix")
    ax2.plot(np.degrees(theta_sep), -np.degrees(omega_sep), 'r--', lw=2)

    # Current trajectory + energy
    E = 0.5 * m * (L**2) * omega0**2 + m * g * L * (1 - np.cos(theta0))
    ax2.plot(np.degrees(theta), np.degrees(omega), lw=2, label=f'Trajectory (E={E:.2f} J)')

    ax2.set_xlabel('Angle θ (degrees)')
    ax2.set_ylabel('Angular velocity ω (degrees/s)')
    ax2.set_title('Complete Phase Space')
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-360, 360)
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

# === Download charts ===
with st.expander("💾 Download Charts"):
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png", dpi=300)
    st.download_button("⬇️ Download θ(t)", buf1.getvalue(), file_name="theta_t.png", mime="image/png")

    buf2 = BytesIO()
    fig2.savefig(buf2, format="png", dpi=300)
    st.download_button("⬇️ Download Phase Space", buf2.getvalue(), file_name="phase_space.png", mime="image/png")