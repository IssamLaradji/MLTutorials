import numpy as np
import pylab as plt

f = lambda x: x**2

t = 0.5 


x1 = -3
x2 = 4

t = 0.5


x = np.linspace(-5, 5, 100)
T = np.linspace(0, 1, 50)


plt.plot(x, f(x), label="$f(x) = x^2$")

plt.grid(True)
plt.scatter(x1, f(x1))
plt.scatter(x2, f(x2))

lhs = lambda t: f(t*x1 + (1-t)*x2)
rhs = lambda t: t*f(x1) + (1-t)*f(x2)
xvals = lambda t: t*x1 + (1-t)*x2
plt.title("$f(x) = x^2$, $x_1 = -3$, $x_2 = 4$, $t \in [0, 1]$")
plt.plot(xvals(T), lhs(T), label="$g(t) = f(t \cdot x_1 + (1-t) \cdot x_2)$")
plt.plot(xvals(T), rhs(T), label="$h(t) = t \cdot f(x_1) + (1-t) \cdot f(x_2) $")
plt.legend(loc="best")
plt.xlabel("$x$")

plt.show()