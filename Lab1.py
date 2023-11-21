import matplotlib.pyplot as plt
import numpy as np
from cmath import sqrt
from typing import Callable
from typing import Union

def f(x):
    return np.power(x, 5) - 7*np.power(x, 2) + 2*x +1

def diff_f(x):
    return 5*np.power(x, 4) - 14*x + 2

def diff2_f(x):
    return 20*np.power(x, 3) - 14

def g(x):
    return x - np.sign(5*x**4 - 14*x + 2)*f(x)*0.1


def maxiter_fixed_point(x0, eps):
    q = func_norm(g, x0)
    print(q)
    if(q >= 1):
        print("q = {}>= 1".format(q))
        raise ValueError
    if(abs(g(x0)-x0) > abs(1-q)*0.05):
        print("Second condition is not satisfied")
        raise ValueError
    n = np.floor(np.log(abs(x0 - g(x0))/eps*(1-q))/np.log(1/q))+1
    n = int(n)
    return n

def func_norm(f, x0):
    norm = max(abs((f(x+1e-14)-f(x))/1e-14) for x in np.linspace(x0-0.05, x0+0.05, 50))
    return norm


#fixed-point iteration
def find_fixed_point(x0, eps, maxiter):
    x = x0
    for i in range(maxiter):
        x = g(x)
        print_iteration_result(i, x, abs(x - x0), f(x))
        if abs(x - x0) < eps and abs(f(x)) < eps:
            return x
        x0 = x
    return x

def print_iteration_result(i, x, dist, f_x):
    print("i = {:<4} x_i = {:<14.10f} x_i - x_i-1 = {:<14.10f}  f(x_i) = {:<14.10f}".format(i, x, dist, f_x))


def fixed_point(x0, eps, result: dict):
    maxiter = maxiter_fixed_point(x0, eps)
    root = find_fixed_point(x0, eps, maxiter)
    result["fixed_point"] = root
    result["maxiter_fixed_point"] = maxiter
    return result

def find_secant(x0, x1, eps, maxiter):
    x = x0
    for i in range(maxiter):
        x = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
        print_iteration_result(i, x, abs(x - x0), f(x))
        if abs(x - x0) < eps and abs(f(x)) < eps:
            return x
        if diff_f(x)*diff2_f(x) < 0:
            x0 = x1
            x1 = x
        else:
            x0 = x
    return x

def secant(x0, x1, eps, result: dict):
    maxiter = 100
    root = find_secant(x0, x1, eps, maxiter)
    result["secant"] = root
    result["maxiter_secant"] = maxiter
    return result


Num = Union[float, complex]
Func = Callable[[Num], Num]

def div_diff(f: Func, xs: list[Num]):
    """Calculate the divided difference f[x0, x1, ...]."""
    if len(xs) == 2:
        a, b = xs
        return (f(a) - f(b)) / (a - b)
    else:
        return (div_diff(f, xs[1:]) - div_diff(f, xs[0:-1])) / (xs[-1] - xs[0])

def mullers_method(f: Func, xs: (Num, Num, Num), iterations: int, eps) -> float:
    """Return the root calculated using Muller's method."""
    x0, x1, x2 = xs
    for i in range(iterations):
        w = div_diff(f, (x2, x1)) + div_diff(f, (x2, x0)) - div_diff(f, (x2, x1))
        s_delta = sqrt(w ** 2 - 4 * f(x2) * div_diff(f, (x2, x1, x0)))
        denoms = [w + s_delta, w - s_delta]
        # Take the higher-magnitude denominator
        x3 = x2 - 2 * f(x2) / max(denoms, key=abs)
        print_iteration_result(i, x3, abs(x3 - x2), f(x3))
        if abs(x3 - x2) < eps and abs(f(x3)) < eps:
            return x3
        # Advance
        x0, x1, x2 = x1, x2, x3
    return x3




def muller(x0, x1, x2, eps, result: dict):
    maxiter = 100
    root = mullers_method(f, [x0, x1, x2], maxiter, eps)
    root = float(root.real)
    result["muller"] = root
    result["maxiter_muller"] = maxiter
    return result

def result(method: str, x0 = 0.55, eps = 1e-5):
    result = {}
    if method == "fixed_point":
        result = fixed_point(x0, eps, result)
        print("fixed_point: ", result["fixed_point"])
        print("maxiter_fixed_point: ", result["maxiter_fixed_point"])
    elif method == "secant":
        result = secant(x0, x0+0.1, eps, result)
        print("secant: ", result["secant"])
        print("maxiter_secant: ", result["maxiter_secant"])
    elif method == "muller":
        result = muller(x0, x0+0.1, x0+0.2, eps, result)
        print("muller: ", result["muller"])
        print("maxiter_muller: ", result["maxiter_muller"])



def plotter():
    x = np.linspace(-1, 2, 50)
    plt.plot(x, f(x))
    roots = np.roots([1, 0, 0, -7, 2, 1])
    plt.scatter(roots, f(roots), color='red')
    result = {}
    result = fixed_point(0.5, 0.00001, result)
    x1 = result["fixed_point"]
    print(x1)
    print(f(x1))
    plt.scatter(x1, f(x1), color='green')

    plt.xlim(-1, 2)  # Set the x-axis limits
    plt.ylim(-1, 2)  # Set the y-axis limits

    plt.grid()
    plt.show()

#plotter()

result("muller")


