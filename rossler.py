import numpy as np
from scipy.integrate import solve_ivp

def rossler_deriv(t, state, f1, f2, c):
    x1, y1, z1, x2, y2, z2 = state
    dx1 = -2*np.pi*f1*y1 - z1 + c*(x2 - x1)
    dy1 =  2*np.pi*f1*x1 + 0.15*y1
    dz1 =  0.2 + z1*(x1 - 10)
    dx2 = -2*np.pi*f2*y2 - z2 + c*(x1 - x2)
    dy2 =  2*np.pi*f2*x2 + 0.15*y2
    dz2 =  0.2 + z2*(x2 - 10)
    return [dx1, dy1, dz1, dx2, dy2, dz2]

def simulate_rossler(f1, f2, c, T, fs, init_state):
    t_eval = np.linspace(0, T, int(T*fs))
    sol = solve_ivp(rossler_deriv, [0, T], init_state, args=(f1, f2, c), t_eval=t_eval)
    return sol.y, t_eval
