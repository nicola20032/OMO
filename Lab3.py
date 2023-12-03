#Variant 66
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def chebyshev_nodes(a, b, n):
    return np.array([(a+b)/2 + (b-a)/2*np.cos((2*i+1)/(2*n+2)*np.pi) for i in range(n)])

def f(x):
    return abs(x*x-1)-x*x+abs(x-2)

def divided_difference_recursive(x, y):
    if len(y) == 1:
        return y[0]
    else:
        return (divided_difference_recursive(x[1:], y[1:]) - divided_difference_recursive(x[:-1], y[:-1]))/(x[-1] - x[0])


def newton_interpolation(x, y):
    n = len(x)
    p = 0
    for i in range(n):
        p += divided_difference_recursive(x[:i+1], y[:i+1])*np.prod([sp.Symbol('x') - x[j] for j in range(i)])
    return p

def plot(a,b, n):
    x = np.linspace(a,b,100)
    y = f(x)

    x1 = np.linspace(a,b,n)
    y1 = f(x1)

    x2 = chebyshev_nodes(a,b,n)
    y2 = f(x2)

    p1 = newton_interpolation(x1, y1)
    p2 = newton_interpolation(x2, y2)

    p1_func = sp.lambdify(sp.Symbol('x'), p1, 'numpy')
    p2_func = sp.lambdify(sp.Symbol('x'), p2, 'numpy')

    p1y = p1_func(x)
    p2y = p2_func(x)

    plt.subplot(2, 1, 1)
    plt.plot(x, y, label='f(x)', linewidth=2, linestyle='dashed')
    plt.plot(x, p1y, label=f'p_basic(x)', linewidth=1, linestyle='solid')
    plt.plot(x, p2y, label=f'p_chebyshev(x)', linewidth=3, linestyle='dashdot')

    # Add labels, legend, and other plot details
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Title of the Plot')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, y - p1y, label='f(x) - p_basic(x)', linewidth=2, linestyle='dashed')
    plt.plot(x, y -p2y, label='f(x) - p_chebyshev(x)', linewidth=2, linestyle='solid')
    plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Second Plot')
    plt.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    print(f'p_basic(x) = {sp.simplify(p1)}'
            f'\np_chebyshev(x) = {sp.simplify(p2)}')

    plt.show()


def main():
    a = -2
    b = 4
    n = 6
    plot(a, b, n)

if __name__ == "__main__":
    main()