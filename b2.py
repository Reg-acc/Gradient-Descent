import numpy as np

# Đạo hàm của hàm f(x) = (1/3)*x^3 - x
def grad(x):
    return x**2 - 1

# Hàm chi phí f(x) = (1/3)*x^3 - x
def cost(x):
    return (1/3) * x**3 - x

# Thuật toán Gradient Descent
def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:  # Điều kiện dừng
            break
        x.append(x_new)
    return (x, it)

# Chạy thử nghiệm với các điểm khởi tạo x0 = 0 và x0 = 5, learning rate eta = 0.1
(x1, it1) = myGD1(0, .1)
(x2, it2) = myGD1(5, .1)

# In kết quả
print('Solution x1 = %f, cost = %f, after %d iterations' % (x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, after %d iterations' % (x2[-1], cost(x2[-1]), it2))