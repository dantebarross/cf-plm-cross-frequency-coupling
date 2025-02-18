import numpy as np

def kuramoto_simulation(T, dt, freqs, k, tau):
    N = len(freqs)
    num_steps = int(T/dt)
    theta = np.zeros((N, num_steps))
    theta[:,0] = np.random.uniform(0, 2*np.pi, size=N)
    delay_steps = int(tau/dt)
    for t in range(1, num_steps):
        for n in range(N):
            sum_term = 0
            for p in range(N):
                idx = t - delay_steps if t - delay_steps >= 0 else 0
                sum_term += np.sin(theta[p, idx] - theta[n, t-1])
            dtheta = 2*np.pi*freqs[n] + (k/N)*sum_term
            theta[n, t] = theta[n, t-1] + dtheta*dt
    return theta
