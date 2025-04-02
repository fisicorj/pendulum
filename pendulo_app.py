import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st
from io import BytesIO

# === Page Configuration ===
st.set_page_config(page_title="Pendulum Simulator", layout="wide")
st.title("🎯 Interactive Simple Pendulum Simulator")

# === Interactive Parameters ===
st.sidebar.header("Pendulum Parameters")
g = st.sidebar.slider("Gravity (m/s²)", 1.0, 20.0, 9.81, 0.01)
L = st.sidebar.slider("Rod Length (m)", 0.1, 5.0, 1.0, 0.1)
theta0_deg = st.sidebar.slider("Initial Angle θ₀ (degrees)", -180, 180, 30)
omega0_deg = st.sidebar.slider("Initial Angular Velocity ω₀ (degrees/s)", -360, 360, 0)

# Convert to radians
theta0 = np.radians(theta0_deg)
omega0 = np.radians(omega0_deg)
t_eval = np.linspace(0, 10, 1000)

# === Differential Equation ===
def pendulum(t, y):
    return [y[1], - (g / L) * np.sin(y[0])]

# === Solve ODE ===
sol = solve_ivp(pendulum, (0, 10), [theta0, omega0], t_eval=t_eval)
theta = sol.y[0]
omega = sol.y[1]

# === Harmonic Solution ===
omega_natural = np.sqrt(g / L)
theta_harm = theta0 * np.cos(omega_natural * t_eval)

# === Layout with Columns ===
col1, col2 = st.columns(2)

# === Plot θ(t) ===
with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(t_eval, np.degrees(theta), label='Numerical Solution')
    ax1.plot(t_eval, np.degrees(theta_harm), '--', label='Harmonic Approximation')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle θ (degrees)')
    ax1.set_title('θ(t)')
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

# === Phase Space Plot ===
with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(np.degrees(theta), np.degrees(omega), label='Trajectory in Phase Space')
    ax2.set_xlabel('Angle θ (degrees)')
    ax2.set_ylabel('Angular Velocity ω (degrees/s)')
    ax2.set_title('Phase Space')
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# === Download Graphs as Images ===
with st.expander("💾 Download Graphs"):
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png", dpi=300)
    st.download_button("⬇️ Download θ(t)", buf1.getvalue(), file_name="theta_t.png", mime="image/png")

    buf2 = BytesIO()
    fig2.savefig(buf2, format="png", dpi=300)
    st.download_button("⬇️ Download Phase Space", buf2.getvalue(), file_name="phase_space.png", mime="image/png")