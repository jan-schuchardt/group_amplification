from itertools import combinations_with_replacement
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from scipy.special import loggamma


def log_factorial(n: NDArray[np.int64]) -> NDArray[np.float64]:
    """Log factorial of an input array.

    Args:
        n (NDArray[np.int64]): Array of arbitrary shape.

    Returns:
        NDArray[np.float64]: Log factorial of same shape.
    """
    return loggamma(n+1)


def log_binomial_coefficient(n: NDArray[np.int64], k: NDArray[np.int64]) -> NDArray[np.float64]:
    """log(n choose k) of input arrays

    Args:
        n (NDArray[np.int64]): Array that is broadcastable to shape of k
        k (NDArray[np.int64]): Array that is broadcastable to shape of n

    Returns:
        NDArray[np.float64]: Log factorial of broadcasted shape
    """

    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)


def log_multinomial_coefficient(n: NDArray[np.int64], k: NDArray[np.int64]) -> NDArray[np.float64]:
    """log(n choose k[..., 0], k[..., 1],...,k[..., N])

    Args:
        n (NDArray[np.int64]): Array of shape S
        k (NDArray[np.int64]): Array of shape S x N

    Returns:
        NDArray[np.float64]: Log of multinomial coefficient with same shape as k
    """
    return log_factorial(n) - log_factorial(k).sum(axis=-1)


def fixed_size_compositions(n: int, k: int) -> NDArray[np.int64]:
    """All ways of writing n as a sum of k integers.

    Args:
        n (int): Non-negative integer
        k (int): Non-negative integer

    Returns:
        NDArray[np.int64]: Array of shape Nxk, where each row is one composition.
    """
    partition_positions = np.vstack(list(combinations_with_replacement(np.arange(n+1), k-1)))

    first_values = partition_positions[:, 0][:, np.newaxis]  # N x 1
    middle_values = np.diff(partition_positions, axis=1)  # N x (k - 2)
    last_values = n - first_values - middle_values.sum(axis=1, keepdims=True)  # N x 1

    return np.hstack([first_values, middle_values, last_values])


def get_privacy_spent(
    *, orders: list[float] | float, rdp: list[float] | float, delta: float
) -> tuple[float, float]:
    """
    This code is taken from Opacus
    https://github.com/pytorch/opacus/blob/main/opacus/accountants/analysis/rdp.py

    Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    if idx_opt == 0 or idx_opt == len(eps) - 1:
        extreme = "smallest" if idx_opt == 0 else "largest"
        warn(
            f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound."
        )
    return eps[idx_opt], orders_vec[idx_opt]