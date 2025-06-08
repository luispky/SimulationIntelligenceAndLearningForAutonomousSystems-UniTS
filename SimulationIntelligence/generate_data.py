"""
DATA GENERATION FUNCTIONS FOR THE FIVE FUNCTION-APPROXIMATION SCENARIOS
"""

import numpy as np


# SCENARIO 1: Continuous function with linear correlation.
def generate_data_scenario1_continuous_linear(
    n_low=11, n_high=4, noise=False, seed=None
):
    """
    Low-fidelity:  yl(x) = A*(6x - 2)^2 * sin(12x - 4) + B*(x - 0.5) + C
    High-fidelity: yh(x) = (6x - 2)^2 * sin(12x - 4)
    with A=0.5, B=10, C=-5, x in [0,1].

    This function generates low-fidelity and high-fidelity data
    based on a continuous function with linear correlation.

    The low-fidelity data is a noisy version of the high-fidelity data,
    which is a transformation of the Forrester function.
    The high-fidelity data is a clean version of the Forrester function.
    The function also generates a plot range for visualization.
    The noise can be added to both low-fidelity and high-fidelity data.
    Parameters:
    - n_low: Number of low-fidelity samples.
    - n_high: Number of high-fidelity samples.
    - noise: If True, adds Gaussian noise to the data.
    - seed: Random seed for reproducibility.
    Returns:
    - xl: Low-fidelity input samples. Shape (n_low, 1).
    - yl: Low-fidelity output samples. Shape (n_low, 1).
    - xh: High-fidelity input samples.  Shape (n_high, 1).
    - yh: High-fidelity output samples. Shape (n_high, 1).
    - x_plot: Input samples for plotting. Shape (200, 1).
    - yh_plot: High-fidelity output samples for plotting. Shape (200, 1).
    - yl_func: Function to compute low-fidelity outputs.
    - yh_func: Function to compute high-fidelity outputs.
    """
    A, B, C = 0.5, 10.0, -5.0
    if seed is not None:
        np.random.seed(seed)

    xl = np.linspace(0, 1, n_low).reshape(-1, 1)
    xh = np.array([0, 0.4, 0.6, 1.0]).reshape(-1, 1)
    if n_high != 4:
        # Fallback if n_high is changed
        xh = np.linspace(0, 1, n_high).reshape(-1, 1)

    x_plot = np.linspace(0, 1, 200).reshape(-1, 1)

    def yl_func(x_in):
        return A * (6 * x_in - 2) ** 2 * np.sin(12 * x_in - 4) + B * (x_in - 0.5) + C

    def yh_func(x_in):
        return (6 * x_in - 2) ** 2 * np.sin(12 * x_in - 4)

    yl = yl_func(xl)
    yh = yh_func(xh)

    if noise:
        yl += 0.01 * np.std(yl) * np.random.randn(*yl.shape)
        yh += 0.01 * np.std(yh) * np.random.randn(*yh.shape)

    yh_plot = yh_func(x_plot)

    return xl, yl, xh, yh, x_plot, yh_plot, yl_func, yh_func


# SCENARIO 2: Discontinuous function with linear correlation.
def generate_data_scenario2_discontinuous_linear(
    n_low=38, n_high=5, noise=False, seed=None
):
    """
    Low-fidelity:  yl(x) = piecewise "Forrester" with jump
    High-fidelity: yh(x) = piecewise linear transform of yl.
    Jump at x=0.5. Forrester: (6x-2)^2 sin(12x-4)
    """
    if seed is not None:
        np.random.seed(seed)

    xl = np.linspace(0, 1, n_low).reshape(-1, 1)
    xh = np.linspace(0, 1, n_high).reshape(-1, 1)

    def yl_func(x):
        # piecewise
        y = np.zeros_like(x)
        mask = (x <= 0.5)
        # left half
        y[mask] = 0.5 * (6 * x[mask] - 2) ** 2 * np.sin(12 * x[mask] - 4) + 10 * (x[mask] - 0.5) - 5
        # right half
        y[~mask] = 3.0 + 0.5 * (6 * x[~mask] - 2) ** 2 * np.sin(12 * x[~mask] - 4) + 10 * (x[~mask] - 0.5) - 5
        return y

    def yh_func(x):
        # piecewise, linear transformation of yl
        ylvals = yl_func(x)
        y = np.zeros_like(x)
        mask = (x <= 0.5)
        # left half
        y[mask] = 2 * ylvals[mask] - 20 * x[mask] + 20
        # right half
        y[~mask] = 4.0 + 2 * ylvals[~mask] - 20 * x[~mask] + 20
        return y

    yl = yl_func(xl)
    yh = yh_func(xh)

    if noise:
        yl += 0.01 * np.random.randn(*yl.shape)
        yh += 0.01 * np.random.randn(*yh.shape)

    # For plotting
    x_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    yh_plot = yh_func(x_plot)

    return xl, yl, xh, yh, x_plot, yh_plot, yl_func, yh_func


# SCENARIO 3: Continuous function with *nonlinear* correlation.
def generate_data_scenario3_continuous_nonlinear(
    n_low=51, n_high=14, noise=False, seed=None
):
    """
    Low-fidelity:  yl(x) = sin(8 pi x)
    High-fidelity: yh(x) = (x - sqrt(2)) * [yl(x)]^2
    on x in [0,1].
    """
    if seed is not None:
        np.random.seed(seed)

    xl = np.linspace(0, 1, n_low).reshape(-1, 1)
    xh = np.linspace(0, 1, n_high).reshape(-1, 1)

    def yl_func(x_in):
        return np.sin(8 * np.pi * x_in)

    def yh_func(x_in):
        return (x_in - np.sqrt(2)) * (yl_func(x_in)) ** 2

    yl = yl_func(xl)
    yh = yh_func(xh)
    if noise:
        yl += 0.01 * np.random.randn(*yl.shape)
        yh += 0.01 * np.random.randn(*yh.shape)

    x_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    yh_plot = yh_func(x_plot)

    return xl, yl, xh, yh, x_plot, yh_plot, yl_func, yh_func


# SCENARIO 4: Phase-shifted oscillations.
def generate_data_scenario4_phase_shift(
    n_low=51, n_high=16, noise=False, seed=None, use_embedding=True
):
    """
    yl(x) = sin(8 pi x)
    yh(x) = x^2 + sin^2(8 pi x + pi/10)
    """
    if seed is not None:
        np.random.seed(seed)

    xl = np.linspace(0, 1, n_low).reshape(-1, 1)
    xh = np.linspace(0, 1, n_high).reshape(-1, 1)

    def yl_func(x_in):
        return np.sin(8 * np.pi * x_in)

    def yh_func(x_in):
        return x_in**2 + (np.sin(8 * np.pi * x_in + np.pi / 10.0)) ** 2

    yl = yl_func(xl)
    yh = yh_func(xh)

    if noise:
        yl += 0.05 * np.std(yl) * np.random.randn(*yl.shape)
        yh += 0.02 * np.std(yh) * np.random.randn(*yh.shape)

    x_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    yh_plot = yh_func(x_plot)

    return xl, yl, xh, yh, x_plot, yh_plot, yl_func, yh_func


# SCENARIO 5: 20-dimensional function approximation.
def generate_data_scenario5_20D(n_low=30000, n_high=5000, seed=None):
    """
       yh(X) = (X1 - 1)^2 + sum_{i=2..20} [2 X_i^2 - X_{i-1}]^2
       yl(X) = 0.8 yh(X) - sum_{i=1..19}(0.4 X_i X_{i+1}) - 50
    with X_i in [-3, 3].
    """
    if seed is not None:
        np.random.seed(seed)

    Xl = -3 + 6 * np.random.rand(n_low, 20)
    Xh = -3 + 6 * np.random.rand(n_high, 20)

    def yh_func_20d(X_in):
        term1 = (X_in[:, 0] - 1) ** 2
        sum_terms = np.sum(
            [(2 * X_in[:, i] ** 2 - X_in[:, i - 1]) ** 2 for i in range(1, 20)], axis=0
        )
        return term1 + sum_terms

    def yl_func_20d(X_in):
        yh_val = yh_func_20d(X_in)
        sum_cross_terms = np.sum(
            [0.4 * X_in[:, i] * X_in[:, i + 1] for i in range(19)], axis=0
        )
        return 0.8 * yh_val - sum_cross_terms - 50

    yl_data = yl_func_20d(Xl).reshape(-1, 1)
    yh_data = yh_func_20d(Xh).reshape(-1, 1)

    return Xl, yl_data, Xh, yh_data, yl_func_20d, yh_func_20d
