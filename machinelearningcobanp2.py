import numpy as np

# Đạo hàm f'(x) = 2*x + 5*cos(x)
def grad(theta):
    return 2 * theta + 5 * np.cos(theta)

# Hàm chi phí f(x) = x^2 + 5*sin(x)
def cost(theta):
    return theta**2 + 5 * np.sin(theta)

# Kiểm tra điều kiện hội tụ
def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new)) / len(theta_new) < 1e-3

# Thuật toán Gradient Descent với Momentum
def GD_momentum(theta_init, grad, eta, gamma):
    # Đưa theta_init về dạng np.array để dùng được len() và np.linalg.norm
    theta_init = np.array(theta_init, dtype=float)
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    
    for it in range(100):
        v_new = gamma * v_old + eta * grad(theta[-1])
        theta_new = theta[-1] - v_new
        if has_converged(theta_new, grad):
            break 
        theta.append(theta_new)
        v_old = v_new
        
    return theta, it

# Chạy thử nghiệm với 2 điểm khởi tạo theta_init = [-5.0] và [5.0]
# Learning rate eta = 0.1, hệ số momentum gamma = 0.9
eta = 0.1
gamma = 0.9

(path1, it1) = GD_momentum([-5.0], grad, eta, gamma)
(path2, it2) = GD_momentum([5.0], grad, eta, gamma)

sol1 = path1[-1][0]
sol2 = path2[-1][0]

# In kết quả
print('Solution x1 = %f, cost = %f, after %d iterations' % (sol1, cost(sol1), it1))
print('Solution x2 = %f, cost = %f, after %d iterations' % (sol2, cost(sol2), it2))