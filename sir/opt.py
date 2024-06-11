import numpy as np

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return ([dS_dt, dI_dt, dR_dt])

from scipy.integrate import odeint

###
data = np.array([3, 8, 28, 76, 176, 325, 473, 530, 495, 385]) / 1000
I0 = data[0]
S0 = 1 - I0
R0 = 0
###

def loss(params_to_fit, t, data):
    size = len(data)
    beta, gamma = params_to_fit
    solution = odeint(sir_model, [S0, I0, R0], t, args=(beta, gamma))
    return np.sqrt(np.mean((solution[:,1] - data)**2))

from scipy.optimize import minimize

# initial guess for the parameters to fit (beta and gamma)
p0 = [0.5, 0.5]

# time points
t = np.linspace(0, len(data), len(data))

# fit the model
optimal = minimize(loss, p0, args=(t, data), method='L-BFGS-B')

print(optimal.x)

beta_optimal, gamma_optimal = optimal.x

# Predict and compare with data
# Integrate the SIR equations over the time grid, t.
solution_optimal = odeint(sir_model, [S0, I0, R0], t, args=(beta_optimal, gamma_optimal))

from plot import plot_sir

plot_sir(solution_optimal, t, data)