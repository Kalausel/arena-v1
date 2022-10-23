#%%
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum
import torch

import utils

#%%

def DFT_1d(arr : np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Returns the DFT of the array `arr`, using the equation above.
    """
    # Solution: Could be done with np.outer without the for loop.
    matrix = np.empty((len(arr), len(arr)), dtype=complex)
    for ind_tuple, value in np.ndenumerate(matrix):
        matrix[ind_tuple] = np.exp(ind_tuple[0]*ind_tuple[1]*(-2)*np.pi*1j/len(arr))
    if inverse:
        matrix = np.conjugate(matrix)/len(arr)
    return matrix.dot(arr) # Solution uses @ instead of explicit dot()

utils.test_DFT_func(DFT_1d)

def test_DFT_1d(DFT_1d: Callable) -> None:
    # Test example with known DFT from https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Example
    test_arr = np.array([1, 2-1j, -1j, -1+2j])
    computed_dft = DFT_1d(test_arr)
    actual_dft = np.array([2, -2-2j, -2j, 4+4j])
    reconstructed_test_arr = DFT_1d(computed_dft, inverse=True)
    np.testing.assert_allclose(computed_dft, actual_dft, atol=1e-10, err_msg="DFT failed")
    np.testing.assert_allclose(reconstructed_test_arr, test_arr, atol=1e-10, err_msg="Inverse DFT failed")

test_DFT_1d(DFT_1d)

def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Calculates the approximation of the Riemann integral of the function `func`, 
    between the limits x0 and x1.

    You should use the Left Rectangular Approximation Method (LRAM).
    """

    x_samples, width = np.linspace(x0, x1, num=n_samples, endpoint=False, retstep=True)
    y_samples = func(x_samples)
    return np.sum(y_samples*width)

utils.test_integrate_function(integrate_function)

def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Computes the integral of the function x -> func1(x) * func2(x).
    """

    product_func =lambda x : func1(x) * func2(x)
    return integrate_function(product_func, x0, x1, n_samples)

utils.test_integrate_product(integrate_product)



def calculate_fourier_series(func: Callable, max_freq: int = 50):
    """
    Calculates the fourier coefficients of a function, 
    assumed periodic between [-pi, pi].

    Your function should return ((a_0, A_n, B_n), func_approx), where:
        a_0 is a float
        A_n, B_n are lists of floats, with n going up to `max_freq`
        func_approx is the fourier approximation, as described above
    """

    a_0 = 1/np.pi * integrate_function(func, -np.pi, np.pi)
    list_of_A_n = [integrate_product(func, lambda x : np.cos(n*x)/np.pi, -np.pi, np.pi) for n in range(1, max_freq+1)]
    list_of_B_n = [integrate_product(func, lambda x : np.sin(n*x)/np.pi, -np.pi, np.pi) for n in range(1, max_freq+1)]
    func_approx = lambda x : sum([a_0/2] + [A * np.cos((n+1)*x) for n, A in enumerate(list_of_A_n)] + [B * np.sin((n+1)*x) for n, B in enumerate(list_of_B_n)])
    return (a_0, list_of_A_n, list_of_B_n), np.vectorize(func_approx)


step_func = lambda x: 1 * (x > 0)
utils.create_interactive_fourier_graph(calculate_fourier_series, func = step_func)

#%%

NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = np.linspace(-np.pi, np.pi, 2000)
y = TARGET_FUNC(x)

x_cos = np.array([np.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = np.array([np.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = np.random.randn()
A_n = np.random.randn(NUM_FREQUENCIES)
B_n = np.random.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = a_0/2 + np.transpose(x_cos).dot(A_n) + np.transpose(x_sin).dot(B_n)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = np.square((y - y_pred)).sum()

    if step % 2 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)

    # TODO: compute gradients of coeffs with respect to `loss`
    grad_a0 = 2 * (y_pred - y).sum() * 1/2
    grad_A_n = 2 * x_cos.dot(y_pred - y)
    grad_B_n = 2 * x_sin.dot(y_pred - y)

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    a_0 -= LEARNING_RATE * grad_a0
    A_n -= LEARNING_RATE * grad_A_n
    B_n -= LEARNING_RATE * grad_B_n

utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%
import math

NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = torch.linspace(-math.pi, math.pi, 2000)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = torch.rand(1)
A_n = torch.rand(NUM_FREQUENCIES)
B_n = torch.rand(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = a_0/2 + torch.matmul(torch.transpose(x_cos, 0, 1), A_n) + torch.matmul(torch.transpose(x_sin, 0, 1), B_n)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = torch.square((y - y_pred)).sum()

    if step % 2 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.detach(), B_n.detach()])
        y_pred_list.append(y_pred)

    # TODO: compute gradients of coeffs with respect to `loss`
    grad_a0 = 2 * (y_pred - y).sum() * 1/2
    grad_A_n = 2 * torch.matmul(x_cos, (y_pred - y))
    grad_B_n = 2 * torch.matmul(x_sin, (y_pred - y))

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    a_0 -= LEARNING_RATE * grad_a0
    A_n -= LEARNING_RATE * grad_A_n
    B_n -= LEARNING_RATE * grad_B_n
print(coeffs_list)
utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%
