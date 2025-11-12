import numpy as np
import matplotlib.pyplot as plt

def f(x, y, k2):
    y1, y2 = y
    return np.array([y2, -k2 * y1])

def TISE_solver(E, m, L, h=1e-3):
    hbar = 6.626e-34 / (2 * np.pi)
    k2 = (2 * m * E) / (hbar**2)

    x_vals = np.arange(0, L + h, h)
    psi_vals = []

    y = np.array([0.0, 1.0])

    for x in x_vals:
        psi_vals.append(y[0])

        K1 = f(x, y, k2)
        K2 = f(x + h/2, y + h*K1/2, k2)
        K3 = f(x + h/2, y + h*K2/2, k2)
        K4 = f(x + h, y + h*K3, k2)

        y = y + (h/6) * (K1 + 2*K2 + 2*K3 + K4)

    psi_vals = np.array(psi_vals)

    norm = np.trapezoid(psi_vals**2, x_vals)
    psi_vals = psi_vals / np.sqrt(norm)

    return np.array(x_vals), psi_vals

def Energy(m, L, n):
    hbar = 6.626e-34 / (2 * np.pi)
    return (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)

def expected_psi(n, L, x):
    return np.sqrt(2 / L) * np.sin((n * np.pi * x) / L)

mass = 1.0
length = 1.0

for n in [1, 2, 3, 4]:
    x, psi_num = TISE_solver(Energy(mass, length, n), mass, length)
    psi_exp = expected_psi(n, length, x)

    plt.plot(x, np.abs(psi_num)**2, label=f"Numerical |ψ{n}|²")
    plt.plot(x, np.abs(psi_exp)**2, "--", label=f"Analytical |ψ{n}|²")
    plt.title(f"Wavefunction squared for n={n}")
    plt.xlabel("x")
    plt.ylabel("|ψ|²")
    plt.legend()
    plt.show()
