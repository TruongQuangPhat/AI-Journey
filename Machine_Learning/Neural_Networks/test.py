import numpy as np
import matplotlib.pyplot as plt

# Hàm và gradient
def f(x, y):
    return x**2 + 10*y**2

def grad(x, y):
    return np.array([2*x, 20*y])

# Batch Gradient Descent
def batch_gd(lr=0.1, steps=20, start=np.array([2.0, 2.0])):
    x = start.copy()
    path = [x.copy()]
    for _ in range(steps):
        g = grad(x[0], x[1])
        x -= lr * g
        path.append(x.copy())
    return np.array(path)

# Stochastic GD (giả lập bằng thêm noise vào gradient)
def sgd(lr=0.1, steps=20, start=np.array([2.0, 2.0]), noise=0.5):
    x = start.copy()
    path = [x.copy()]
    for _ in range(steps):
        g = grad(x[0], x[1]) + np.random.randn(2)*noise
        x -= lr * g
        path.append(x.copy())
    return np.array(path)

# SGD + Momentum
def sgd_momentum(lr=0.1, beta=0.9, steps=20, start=np.array([2.0, 2.0]), noise=0.5):
    x = start.copy()
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(steps):
        g = grad(x[0], x[1]) + np.random.randn(2)*noise
        v = beta * v + g
        x -= lr * v
        path.append(x.copy())
    return np.array(path)

# Quỹ đạo
path_batch = batch_gd()
path_sgd = sgd()
path_mom = sgd_momentum()

# Vẽ contour
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

plt.figure(figsize=(8, 8))
plt.contour(X, Y, Z, levels=30, cmap="jet")

plt.plot(path_batch[:,0], path_batch[:,1], "o--", color="blue", label="Batch GD")
plt.plot(path_sgd[:,0], path_sgd[:,1], "o--", color="red", label="SGD")
plt.plot(path_mom[:,0], path_mom[:,1], "o--", color="green", label="SGD + Momentum")

plt.legend()
plt.title("So sánh Batch GD, SGD và SGD + Momentum")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
