import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st
from io import BytesIO
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter, FFMpegWriter
import tempfile
import os

# === Page configuration ===
st.set_page_config(page_title="Pendulum Simulator", layout="wide")
st.title("🎯 Interactive Simple Pendulum Simulator")

st.markdown("""
This application simulates the motion of a simple pendulum and visualizes its behavior through various representations:

- **θ(t)**: The angular displacement over time compared to the harmonic approximation.
- **Phase space**: The angular velocity versus angular displacement, including the separatrix and energy levels.
- **Pendulum animation**: A real-time visual representation of the pendulum swing in physical space.

The governing differential equation of motion is:
""")

st.latex(r"\frac{d^2\theta}{dt^2} + \frac{g}{L} \sin(\theta) = 0")

st.markdown("""
Where:
- \( \ttheta \) is the angle from the vertical,
- \( g \) is the gravitational acceleration,
- \( L \) is the length of the pendulum rod.
""")

# === Sidebar parameters ===
st.sidebar.header("Pendulum Parameters")
g = st.sidebar.number_input("Gravity (m/s²)", min_value=1.0, max_value=20.0, value=9.81, step=0.01)
L = st.sidebar.number_input("Rod length (m)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
m = 1.0

theta0_deg = st.sidebar.number_input("Initial angle θ₀ (degrees)", min_value=-180, max_value=180, value=30)
omega0_deg = st.sidebar.number_input("Initial angular velocity ω₀ (degrees/s)", min_value=-360, max_value=360, value=0)
theta0 = np.radians(theta0_deg)
omega0 = np.radians(omega0_deg)
t_eval = np.linspace(0, 10, 200)

# === Cached ODE solver ===
@st.cache_resource
def solve_pendulum(g, L, theta0, omega0, t_eval):
    def pendulum(t, y):
        return [y[1], - (g / L) * np.sin(y[0])]
    return solve_ivp(pendulum, (0, 10), [theta0, omega0], t_eval=t_eval)

sol = solve_pendulum(g, L, theta0, omega0, t_eval)
theta = sol.y[0]
omega = sol.y[1]

# === Energies ===
KE = 0.5 * m * (L * omega)**2  # Kinetic Energy
PE = m * g * (L - L * np.cos(theta))  # Potential Energy
TE = KE + PE  # Total Energy

# === Harmonic solution ===
omega_natural = np.sqrt(g / L)
theta_harm = theta0 * np.cos(omega_natural * t_eval)

# === Layout with columns ===
col1, col2 = st.columns(2)

# === θ(t) plot ===
with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(t_eval, np.degrees(theta), label='Numerical solution', color='blue')
    ax1.plot(t_eval, np.degrees(theta_harm), '--', label='Harmonic approximation', color='orange')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle θ (degrees)')
    ax1.set_title('Angular Displacement θ(t)')
    ax1.grid(True)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    st.pyplot(fig1)
    plt.close(fig1)

# === Phase space with separatrix, vectors, and energy ===
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 5))

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

    theta_sep = np.linspace(-np.pi, np.pi, 500)
    omega_sep = np.sqrt(2 * g / L * (1 - np.cos(theta_sep)))
    ax2.plot(np.degrees(theta_sep), np.degrees(omega_sep), 'r--', lw=2, label="Separatrix")
    ax2.plot(np.degrees(theta_sep), -np.degrees(omega_sep), 'r--', lw=2)

    E = 0.5 * m * (L**2) * omega0**2 + m * g * L * (1 - np.cos(theta0))
    ax2.plot(np.degrees(theta), np.degrees(omega), lw=2, label=f'Trajectory (E={E:.2f} J)', color='green')

    ax2.set_xlabel('Angle θ (degrees)')
    ax2.set_ylabel('Angular velocity ω (degrees/s)')
    ax2.set_title('Phase Space with Separatrix and Energy')
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-360, 360)
    ax2.grid(True)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    st.pyplot(fig2)
    plt.close(fig2)

# === Pendulum animation ===
with st.expander("🎥 Pendulum Animation"):
    st.markdown("""
    This animation shows the pendulum in real space, swinging back and forth under gravity based on the chosen parameters.
    It is computed from the numerical solution of the non-linear differential equation governing pendulum motion.

    The scale of the animation is physically accurate. One unit on the screen corresponds to one real meter. The pendulum string has length L (in meters), and its motion follows the real-time solution of the second-order differential equation.
    """)

    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.set_xlim(-1.2*L, 1.2*L)
    ax3.set_ylim(-1.2*L, 0.2)
    ax3.set_aspect('equal')
    ax3.grid()
    ax3.plot([-0.5, 0.5], [-1.25*L, -1.25*L], 'k-', lw=2)
    ax3.text(0, -1.28*L, '1 meter', ha='center', fontsize=9)

    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    line, = ax3.plot([], [], 'o-', lw=2)
    time_text = ax3.text(0.05, 0.9, '', transform=ax3.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        line.set_data([0, x[frame]], [0, y[frame]])
        time_text.set_text(f"t = {t_eval[frame]:.2f}s")
        return line, time_text

    ani = animation.FuncAnimation(fig3, update, frames=len(t_eval), init_func=init, interval=10, blit=True)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        ani.save(tmpfile.name, writer=PillowWriter(fps=30))
        tmpfile.seek(0)
        st.image(tmpfile.name, caption="Pendulum Animation", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as mp4file:
        ani.save(mp4file.name, writer=FFMpegWriter(fps=30))
        st.download_button("⬇️ Download MP4 Animation", mp4file.read(), file_name="pendulum.mp4", mime="video/mp4")
    plt.close(fig3)

# === Energy plots ===
st.markdown("### ⚡ Energy over Time")
fig4, ax4 = plt.subplots()
ax4.plot(t_eval, KE, label='Kinetic Energy (J)', color='blue')
ax4.plot(t_eval, PE, label='Potential Energy (J)', color='orange')
ax4.plot(t_eval, TE, label='Total Energy (J)', color='green')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Energy (Joules)')
ax4.set_title('Energy Analysis of the Pendulum')
ax4.grid(True)
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
st.pyplot(fig4)
plt.close(fig4)

# === Download charts ===
with st.expander("💾 Download Charts"):
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png", dpi=300)
    st.download_button("⬇️ Download θ(t)", buf1.getvalue(), file_name="theta_t.png", mime="image/png")

    buf2 = BytesIO()
    fig2.savefig(buf2, format="png", dpi=300)
    st.download_button("⬇️ Download Phase Space", buf2.getvalue(), file_name="phase_space.png", mime="image/png")

    buf3 = BytesIO()
    fig4.savefig(buf3, format="png", dpi=300)
    st.download_button("⬇️ Download Energy Plot", buf3.getvalue(), file_name="energy_plot.png", mime="image/png")
