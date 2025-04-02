# 🌀 Interactive Simple Pendulum Simulator

This project is a web-based interactive simulator for the **simple pendulum**, developed using [Streamlit](https://streamlit.io). It allows users to explore the physics of a pendulum by adjusting parameters and viewing results in real-time.

---

## 🚀 Features

- Adjustable parameters via sidebar:
  - Initial angle \( \theta_0 \)
  - Initial angular velocity \( \omega_0 \)
  - Gravity \( g \)
  - Rod length \( L \)
- Numerical solution of the pendulum equation using `scipy.integrate.solve_ivp`
- Harmonic approximation comparison
- Time evolution plot \( \theta(t) \)
- Phase space plot \( \theta \times \omega \)
- High-resolution graph downloads

---

## 📦 Requirements

To run this app locally, install the required packages:

```bash
pip install streamlit numpy matplotlib scipy
```

---

## ▶️ Running the App

```bash
streamlit run pendulo_app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) to interact with the simulator.

---

## 🌐 Deploying on Streamlit Cloud

You can deploy this app online using [Streamlit Cloud](https://streamlit.io/cloud):

1. Fork or clone this repository.
2. Make sure `pendulo_app.py` and `requirements.txt` are in the root directory.
3. Go to Streamlit Cloud and click **New App**.
4. Select the repo and file name.
5. Click **Deploy**.

---

## 📁 File Structure

```
pendulum-streamlit/
├── pendulo_app.py       # Main Streamlit app
├── requirements.txt     # Python dependencies
├── README.md            # Project info and instructions
```

---

## 🧠 Physics Behind It

This simulator numerically solves the nonlinear second-order differential equation of a simple pendulum:

\[ \ddot{\theta} + \frac{g}{L} \sin(\theta) = 0 \]

It also shows the linearized harmonic solution for small angles:

\[ \theta(t) \approx \theta_0 \cos\left(\sqrt{\frac{g}{L}}t\right) \]

---

## 👨‍💻 Author

Developed with ❤️ by Manoel Moraes

Feel free to use, share, or contribute!
