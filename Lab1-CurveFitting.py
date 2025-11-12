import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def linear_fit(x,y):
    n = len(x)
    xi = yi = xiyi = xi_sq = 0

    for i in range(n):
        xi += x[i]
        yi += y[i]
        xiyi += x[i]*y[i]
        xi_sq += x[i]**2

    a = ((n*xiyi) - (xi*yi))/((n*xi_sq) - (xi**2))
    b = (yi - (a*xi))/n

    return a,b

def quadratic_fit(x,y):
    n = len(x)
    xi = yi = xiyi = xi_sq = xi_3 = xi_4 = yixi_2 = 0

    for i in range(n):
        xi += x[i]
        yi += y[i]
        xiyi += x[i]*y[i]
        xi_sq += x[i]**2
        xi_3  += x[i]**3
        xi_4 += x[i]**4
        yixi_2 += y[i]*(x[i]**2)

    A = np.array([[xi_4, xi_3, xi_sq],
                 [xi_3, xi_sq, xi],
                 [xi_sq, xi, n]])
    A_inv = np.linalg.inv(A)
    B = np.array([yixi_2, xiyi, yi])

    X = A_inv @ B

    return X[0], X[1], X[2]

def exp_fit(x,y, eta = 0.1, iterations = 10000):
    A,B,C = 1,1,0
    n = len(x)
    E_prev = np.inf

    for it in range(iterations):
        exp_term = np.exp(np.clip(B * x, -1e6, 1e6))
        y_pred = A * exp_term + C
        error = y - y_pred
        E = np.mean(error**2)

        dA = -2 * np.mean(error * exp_term)
        dB = -2 * np.mean(error * A * exp_term * x)
        dC = -2 * np.mean(error)

        A -= eta * dA
        B -= eta * dB
        C -= eta * dC

        if np.abs(E - E_prev) <= 1e-8:
            break

        E_prev = E

    return A, B, C

def rectangular_integration(low, high, a,b,c=0, curve="linear"):
    if curve=="linear":
        def f(x):
            return a*x+b
    elif curve=="quadratic":
        def f(x):
            return a*(x**2) + b*x + c
    if curve=="exp":
        def f(x):
            return a*np.exp(b*x)+c
    
    h = 0.1
    n = int((high-low)/h)
    h = (high-low)/n

    y = 0
    x = low
    for _ in range(n):
        y += f((x+x+h)/2)
        x+=h

    return h*y    

def trapezoidal_integration(low, high, a,b,c=0, curve="linear"):
    if curve=="linear":
        def f(x):
            return a*x+b
    elif curve=="quadratic":
        def f(x):
            return a*(x**2) + b*x + c
    if curve=="exp":
        def f(x):
            return a*np.exp(b*x)+c
    
    h = 0.1
    n = int((high-low)/h)
    h = (high-low)/n

    y = 0
    x = low
    for _ in range(n-1):
        y += f(x)+f(x+h)
        x+=h

    return h*y/2  

data = pd.read_csv("Simple pendulum data.csv")

x = np.array(data["length(l)"])
y = np.array(data["time(t)"])

result = {"linear":{"a": 0, "b": 0}, "quadratic":{"a": 0, "b": 0, "c": 0}, "exp":{"a": 0, "b": 0, "c": 0}}

a,b = linear_fit(x,y)
result["linear"]["a"] = a
result["linear"]["b"] = b
y_pred = a*x+b

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', alpha=0.6, edgecolor='k', label='Data Points')
plt.plot(x, y_pred, color='red', linewidth=2, label='Fitted Curve')
plt.xlabel("X values", fontsize=12)
plt.ylabel("Y values", fontsize=12)
plt.title("Curve Fitting Result", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

a,b,c = quadratic_fit(x,y)
result["quadratic"]["a"] = a
result["quadratic"]["b"] = b
result["quadratic"]["c"] = c
y_pred = a*(x**2) + b*x + c

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', alpha=0.6, edgecolor='k', label='Data Points')
plt.plot(x, y_pred, color='red', linewidth=2, label='Fitted Curve')
plt.xlabel("X values", fontsize=12)
plt.ylabel("Y values", fontsize=12)
plt.title("Curve Fitting Result", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

a,b,c = exp_fit(x,y)
result["exp"]["a"] = a
result["exp"]["b"] = b
result["exp"]["c"] = c
y_pred = a*np.exp(b*x) + c

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', alpha=0.6, edgecolor='k', label='Data Points')
plt.plot(x, y_pred, color='red', linewidth=2, label='Fitted Curve')
plt.xlabel("X values", fontsize=12)
plt.ylabel("Y values", fontsize=12)
plt.title("Curve Fitting Result", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

print(f"Area under curve using rectangular integration: {rectangular_integration(min(x), max(x), result['quadratic']['a'], result['quadratic']['b'], result['quadratic']['c'], 'quadratic')}")
print(f"Area under curve trapezoidal  integration: {rectangular_integration(min(x), max(x), result['quadratic']['a'], result['quadratic']['b'], result['quadratic']['c'], 'quadratic')}")