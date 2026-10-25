# Naca_4-WindTunnel
A Python simulator that computes and visualises potential flow around any NACA 4-digit airfoil using the vortex panel method.

# Working Mechanism
When the program is provided a NACA 4-digit code and an angle of attack, it:
1. Generates the airfoil from standard NACA equations.
2. Discretizes the surface into panels with control/collocation points.
3. Solves for vortex sheet strength on each panel.
4. Integrates streamlines through resulting velocity field.
5. Plots the visualization of airflow.

# Vortex Panel Method
The airfoil surface is split into N panels. Each panel carries a linearly varying vortex sheet strength γ. Enforcing the no-penetration condition (V·n̂ = 0) at every control point, plus the Kutta condition at the trailing edge (γ_first + γ_last = 0), gives a linear system Aγ = b that is solved with numpy.linalg.solve.

Streamlines are computed by treating the velocity field as an ordinary differential equation (ODE) and integrating with scipy.integrate.solve_ivp. Integration stops automatically when a streamline enters the airfoil.

# Explanation: NACA Code
For 2412:

• 2 → max camber = 2% of chord

• 4 → max camber located at 40% of chord

• 12 → max thickness = 12% of chord

# Usage
Run:

python app.py

# Dependencies
pip install numpy matplotlib scipy
