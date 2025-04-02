import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st
from io import BytesIO
from matplotlib.colors import Normalize
import matplotlib.animation as animation

# === Configuração da página ===
st.set_page_config(page_title="Simulador de Pêndulo", layout="wide")
st.title("🎯 Simulador Interativo de Pêndulo Simples")

# === Parâmetros interativos ===
st.sidebar.header("Parâmetros do Pêndulo")
g = st.sidebar.slider("Gravidade (m/s²)", 1.0, 20.0, 9.81, 0.01)
L = st.sidebar.slider("Comprimento do fio (m)", 0.1, 5.0, 1.0, 0.1)
m = 1.0

# Condições iniciais
theta0_deg = st.sidebar.slider("Ângulo inicial θ₀ (graus)", -180, 180, 30)
omega0_deg = st.sidebar.slider("Velocidade angular inicial ω₀ (graus/s)", -360, 360, 0)
theta0 = np.radians(theta0_deg)
omega0 = np.radians(omega0_deg)
t_eval = np.linspace(0, 10, 1000)

# === Equação diferencial ===
def pendulo(t, y):
    return [y[1], - (g / L) * np.sin(y[0])]

# === Resolver EDO ===
sol = solve_ivp(pendulo, (0, 10), [theta0, omega0], t_eval=t_eval)
theta = sol.y[0]
omega = sol.y[1]

# === Solução harmônica ===
omega_natural = np.sqrt(g / L)
theta_harm = theta0 * np.cos(omega_natural * t_eval)

# === Layout com colunas ===
col1, col2 = st.columns(2)

# === Gráfico θ(t) ===
with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(t_eval, np.degrees(theta), label='Solução Numérica')
    ax1.plot(t_eval, np.degrees(theta_harm), '--', label='Aproximação Harmônica')
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Ângulo θ (graus)')
    ax1.set_title('θ(t)')
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

# === Espaço de Fase com Separatriz, Vetores e Energia ===
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 5))

    # Campo vetorial
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

    # Separatriz
    theta_sep = np.linspace(-np.pi, np.pi, 500)
    omega_sep = np.sqrt(2 * g / L * (1 - np.cos(theta_sep)))
    ax2.plot(np.degrees(theta_sep), np.degrees(omega_sep), 'r--', lw=2, label="Separatriz")
    ax2.plot(np.degrees(theta_sep), -np.degrees(omega_sep), 'r--', lw=2)

    # Trajetória atual + energia
    E = 0.5 * m * (L**2) * omega0**2 + m * g * L * (1 - np.cos(theta0))
    ax2.plot(np.degrees(theta), np.degrees(omega), lw=2, label=f'Trajetória (E={E:.2f} J)')

    ax2.set_xlabel('Ângulo θ (graus)')
    ax2.set_ylabel('Velocidade Angular ω (graus/s)')
    ax2.set_title('Espaço de Fase Completo')
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-360, 360)
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# === Animação do pêndulo ===
with st.expander("🎥 Animação do Pêndulo"):
    from matplotlib import animation
    import tempfile

    # Configuração do gráfico
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.set_xlim(-1.2*L, 1.2*L)
    ax3.set_ylim(-1.2*L, 0.2)
    ax3.set_aspect('equal')
    ax3.grid()

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

    ani = animation.FuncAnimation(fig3, update, frames=len(t_eval), init_func=init,
                                  interval=10, blit=True)

    # Salvar como GIF temporário
    tmpfile = BytesIO()
    ani.save(tmpfile, format='gif', writer='pillow')
    tmpfile.seek(0)

    # Exibir no Streamlit
    st.image(tmpfile, caption="Animação do Pêndulo", use_column_width=True)


# === Download dos gráficos ===
with st.expander("💾 Baixar gráficos"):
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png", dpi=300)
    st.download_button("⬇️ Baixar θ(t)", buf1.getvalue(), file_name="theta_t.png", mime="image/png")

    buf2 = BytesIO()
    fig2.savefig(buf2, format="png", dpi=300)
    st.download_button("⬇️ Baixar Espaço de Fase", buf2.getvalue(), file_name="espaco_fase.png", mime="image/png")
