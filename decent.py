import numpy as np
from scipy.stats import ortho_group
from matplotlib import pyplot as plt

class quad_rand:
    def __init__(self, n, seed=135, max_iters=200):
        """
        intitialize random coefficients with given dimention
        """
        np.random.seed(seed)
        self.n = n
        self.b = np.matrix(2 * np.random.rand(n, 1) - 1)
        U = np.matrix(ortho_group.rvs(dim=n))
        D = np.matrix(np.diagflat(np.random.rand(n)))
        self.A = U.T * D * U
        self.max_iters = max_iters

    def f(self, x):
        """
        calculate f(x) with input x
        """
        return np.asscalar(1 / 2 * x.T * self.A * x + self.b.T * x)

    def df(self, x):

        """
        calculate f'(x) with input x
        """
        return self.A * x + self.b

    def grad_desc_const(self, alpha):
        """
        gradient descent with constant step size alpha
        """
        x = np.matrix(np.zeros((self.n, 1)))
        self.fun_values_const = [self.f(x)]
        for i in range(self.max_iters-1):
            # calculate derivative
            d = self.df(x)
            # update x
            x -= alpha * d
            # get new function value
            self.fun_values_const.append(self.f(x))

    def grad_desc_exact(self):
        """
        gradient descent with exact line minimization
        """
        x = np.matrix(np.zeros((self.n, 1)))
        self.fun_values_exact = [self.f(x)]
        for i in range(self.max_iters-1):
            # calculate derivative
            d = self.df(x)
            # calculate step size
            alpha = np.asscalar(d.T * d / (d.T * self.A * d))
            # update x
            x -= alpha * d
            # get new function value
            self.fun_values_exact.append(self.f(x))

    def grad_desc_armijo(self, alpha, beta=0.5, sigma=0.9):
        """
        gradient descent with Armijo step size rule
        """
        x = np.matrix(np.zeros((self.n, 1)))
        self.fun_values_armijo = [self.f(x)]
        for i in range(self.max_iters-1):
            # calculate derivative
            d = self.df(x)
            # backtracking line search
            cur_alpha = alpha
            cur_value = self.f(x - cur_alpha * d)
            while cur_value <= self.fun_values_exact[-1] + sigma * cur_alpha * d.T * d:
                cur_alpha *= beta
                cur_value = self.f(x - cur_alpha * d)
            # update x
            x -= cur_alpha * d
            # get new function value
            self.fun_values_armijo.append(cur_value)

# create my random function
my_fun = quad_rand(100)
# run graident descent
my_fun.grad_desc_const(alpha=0.3)
my_fun.grad_desc_exact()
my_fun.grad_desc_armijo(alpha=1)
# plot
plt.plot(range(200), my_fun.fun_values_const, label='Constant')
plt.plot(range(200), my_fun.fun_values_exact, label='Exact Min')
plt.plot(range(200), my_fun.fun_values_armijo, label='Armijo')
plt.legend(loc="best")
plt.xlabel("Iterations")
plt.ylabel("Function Value")
plt.show()
