__all__ = ['rdp_traditional', 'rdp_traditional_group', 'rdp_tight_group']

import itertools
from warnings import warn

import mpmath as mpm
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import hypergeom

from group_amplification.privacy_analysis.base_mechanisms import (BaseMechanism, GaussianMechanism,
                                                                  RandomizedResponseMechanism)
from group_amplification.privacy_analysis.utils import log_binomial_coefficient


def rdp_traditional_group(alpha: int, base_mechanism: BaseMechanism,
                          dataset_size: int, batch_size: int,
                          group_size: int,
                          eval_method: str, eval_params:  dict) -> float:
    """Group privacy property of RDP applied to amplification by WOR subsampling.

    This corresponds to Proposition 2/11 in (Mironov, 2017) and Theorem 9/27 in (Wang et al., 2019)

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        group_size (int): Number of substituted elements.
        eval_method: Which flavor of group privacy to use.
            Options:
                - traditional: Proposition 2 from Mironov
                - recursive: Proposition 11 from Mironov.
                - gaussian: Assume group privacy property of Gaussian distribution
        eval_params (dict): Must specify "hoelders_exponent".
            This corresponds to p in Proposition 11 from (Mironov, 2017.)
            Can also specify "tight_base_guarantee", if you want to apply the
            traditional group privacy property to our tight subsampling guarantees.

    Returns:
        float: RDP parameter epsilon of the WOR subsampled mechanism w.r.t induced distance.
    """
    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    valid_methods = ['traditional', 'recursive', 'gaussian']

    eval_method = eval_method.lower()
    if eval_method not in valid_methods:
        raise ValueError(f'eval_method {eval_method} is not in {valid_methods}')

    tight_base_guarantee = eval_params.get('tight_base_guarantee', False)

    if eval_method == 'traditional':
        if not np.log2(group_size).is_integer:
            warn('Group size must be power of 2 for proper upper bound to hold.')
        if alpha < 2 * group_size:
            warn('Group size must be at least double of alpha for proper upper bound to hold.')

        epsilon_scale = 3 ** np.log2(group_size)

        if tight_base_guarantee:
            base_guarantee = _rdp_tight_group_quadrature(group_size * alpha, base_mechanism,
                                                         dataset_size, batch_size, 1, eval_params)
        else:
            base_guarantee = rdp_traditional(group_size * alpha, base_mechanism, dataset_size,
                                             batch_size, eval_params)
        return epsilon_scale * base_guarantee

    elif eval_method == 'recursive':
        if not np.log2(group_size).is_integer:
            raise ValueError('Group size must be power of 2 for recursive method.')

        if (eval_params is None) or ('hoelders_exponent' not in eval_params):
            raise ValueError('You must provide the \'hoelders_exponent\' parameter in eval_params.')

        p = eval_params['hoelders_exponent']
        if p <= 1:
            raise ValueError('hoelders_exponent must be > 1')
        q = p / (p - 1)

        if group_size == 1:
            if tight_base_guarantee:
                return _rdp_tight_group_quadrature(alpha, base_mechanism,
                                                   dataset_size, batch_size, 1, eval_params)
            else:
                return rdp_traditional(alpha, base_mechanism, dataset_size, batch_size, eval_params)
        else:
            res = (alpha - 1 / p) / (alpha - 1)
            res *= rdp_traditional_group(
                        p * alpha, base_mechanism, dataset_size, batch_size,
                        group_size // 2, eval_method, eval_params)

            res += rdp_traditional_group(
                        q * (alpha - 1 / p), base_mechanism, dataset_size, batch_size,
                        group_size // 2, eval_method, eval_params)

            return res

    elif eval_method == 'gaussian':
        epsilon_scale = group_size
        if tight_base_guarantee:
            base_guarantee = _rdp_tight_group_quadrature(group_size * alpha, base_mechanism,
                                                         dataset_size, batch_size, 1, eval_params)
        else:
            base_guarantee = rdp_traditional(group_size * alpha, base_mechanism, dataset_size,
                                             batch_size, eval_params)
        return epsilon_scale * base_guarantee

    else:
        assert False


def rdp_traditional(alpha: int, base_mechanism: BaseMechanism,
                    dataset_size: int, batch_size: int,
                    eval_params: dict) -> float:
    """Amplification by subsampling without replacement guarantee.

    This corresponds to Theorem 9/27 in (Wang et al., 2019).

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        eval_params (dict): Must specify "use_self_consistency".
                - True: Use Theorem 27
                - False: Use Theorem 9
            If True, must also specify "use_self_consistency_quadrature"
                - True: Evaluate Theorem 27 via quadrature
                - False. Evaluate Theorem 27 via series expansion

    Returns:
        float: RDP parameter epsilon of the WOR subsampled mechanism.
    """

    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    if 'use_self_consistency' not in eval_params:
        raise ValueError('YOu must specify "use_self_consistency" in eval_params!')

    if eval_params['use_self_consistency']:
        if not any([isinstance(base_mechanism, mechanism_type)
                    for mechanism_type in [GaussianMechanism]]):
            warn('"use_self_consistency" is only confirmed to be correct for Gaussian and Laplace.')

        if 'use_self_consistency_quadrature' not in eval_params:
            raise ValueError('You must provide the bool "use_self_consistency_quadrature" parameter')

    if not (
        eval_params['use_self_consistency'] and eval_params['use_self_consistency_quadrature']):
        return _rdp_traditional_expansion(
            alpha, base_mechanism, dataset_size, batch_size, eval_params
        )

    else:
        return _rdp_self_consistency_quadrature(
            alpha, base_mechanism, dataset_size, batch_size, eval_params
        )

def _rdp_traditional_expansion(
                    alpha: int, base_mechanism: BaseMechanism,
                    dataset_size: int, batch_size: int,
                    eval_params: dict) -> float:
    """Amplification by subsampling without replacement guarantee.

    This corresponds to Theorem 9/27 in (Wang et al., 2019).

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        eval_params (dict): Must specify "use_self_consistency".
                - True: Use Theorem 27
                - False: Use Theorem 9

    Returns:
        float: RDP parameter epsilon of the WOR subsampled mechanism.
    """

    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    if 'use_self_consistency' not in eval_params:
        raise ValueError('YOu must specify "use_self_consistency" in eval_params!')

    if eval_params['use_self_consistency']:
        if not any([isinstance(base_mechanism, mechanism_type)
                    for mechanism_type in [GaussianMechanism]]):
            warn('"use_self_consistency" is only confirmed to be correct for Gaussian and Laplace.')

    log_w = np.log(batch_size) - np.log(dataset_size)

    js = np.arange(0, alpha + 1)  # (alpha + 1)
    # The j=1 term is always 0
    js = np.delete(js, 1)  # (alpha)

    log_summands = log_binomial_coefficient(alpha, js)  # (alpha)
    log_summands += log_w * js

    base_mechanism_terms = base_mechanism.log_expectation(js[1:])  # (alpha - 1)
    base_mechanism_terms += np.minimum(
        np.log(2),
        np.log(np.exp(base_mechanism.epsilon_dp()) - 1) * js[1:]
    )

    # Special case j = 2
    base_mechanism_terms[0] = min(
        base_mechanism_terms[0],
        np.log(4 * (np.exp(base_mechanism.log_expectation(np.array(2))) - 1))
    )

    # Appendix B.5 for j >= 3
    if eval_params['use_self_consistency'] and js.max() >= 3:
        log_consistency_floor, sign_floor = _log_self_consistency_term(
                2 * np.floor(js[2:] / 2), base_mechanism)

        log_consistency_ceil, sign_ceil = _log_self_consistency_term(
                2 * np.floor(js[2:] / 2), base_mechanism)

        assert np.all(sign_floor == sign_ceil)

        base_mechanism_terms[1:] = np.minimum(
            base_mechanism_terms[1:],
            np.log(4) + (log_consistency_floor + log_consistency_ceil) / 2
        )

    log_summands[1:] += base_mechanism_terms

    return logsumexp(log_summands) / (alpha - 1)


def _log_self_consistency_term(ls: NDArray[np.int64],
                               base_mechanism: BaseMechanism
                               ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Function B(epsilon, l) from Appendix B.5 of Wang et al.

    Args:
        ls (NDArray[np.int64]): 1D array
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.

    Returns:
        _type_: Tuple of absolute log-value array and sign array, both with same shape as l
    """

    if ls.ndim != 1:
        raise ValueError('ls must be a 1D array')

    is_ = np.arange(ls.max() + 1)

    pairs = np.stack(np.meshgrid(ls, is_, indexing='ij'), axis=2) # len(ls) x (1 + max(ls)) x 2
    pairs = pairs.reshape(-1, 2)  # (len(ls) * (1 + max(ls))) x 2

    all_summands = np.zeros(pairs.shape[0], dtype=float)  # (len(ls) * (1 + max(ls)))
    nonzero_mask = (pairs[:, 1] <= pairs[:, 0])

    all_summands[nonzero_mask] = log_binomial_coefficient(
                                        pairs[nonzero_mask, 0],
                                        pairs[nonzero_mask, 1])

    all_summands[nonzero_mask] += base_mechanism.log_expectation(pairs[nonzero_mask, 1])

    all_summands = all_summands.reshape(len(ls), len(is_))
    nonzero_mask = nonzero_mask.reshape(len(ls), len(is_))

    return logsumexp(all_summands, b=((-1 * nonzero_mask) ** is_), axis=1, return_sign=True)


def _rdp_self_consistency_quadrature(
            alpha, base_mechanism, dataset_size, batch_size, eval_params):
    """Evaluates Theorem 27 via quadrature.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        group_size (int): Number of substituted elements.
        eval_params (None | dict): Extra parameters for the specific eval method.

    Returns:
        float: RDP parameter epsilon of the WOr subsampled mechanism w.r.t induced distance.
    """
    if isinstance(base_mechanism, RandomizedResponseMechanism):

        return _rdp_self_consistency_quadrature_randomized_response(
            alpha, base_mechanism, dataset_size, batch_size)

    elif isinstance(base_mechanism, GaussianMechanism):

        return _rdp_self_consistency_quadrature_gaussian(
            alpha, base_mechanism, dataset_size, batch_size, eval_params)

    else:
        raise NotImplementedError('Only Gaussian and Randomized Response is supported.')


def _rdp_self_consistency_quadrature_randomized_response(
        alpha, base_mechanism, dataset_size, batch_size):
    """Evaluates Theorem 27 via quadrature for randomized response.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        group_size (int): Number of substituted elements.
        eval_params (None | dict): Extra parameters for the specific eval method.

    Returns:
        float: RDP parameter epsilon of the WOr subsampled mechanism w.r.t induced distance.
    """
    w = batch_size / dataset_size

    theta = np.array([1 - base_mechanism.true_response_prob,
                      base_mechanism.true_response_prob])

    log_full_series = np.log(np.array([
        1 + w * np.abs(theta[1] / theta[0] - 1),
        1 + w * np.abs(theta[0] / theta[1] - 1)
    ]))

    log_full_series *= alpha
    log_full_series = logsumexp(log_full_series, b=theta)

    # Set j=1 summand to zero
    first_summand = np.log(np.array([
        np.abs(theta[1] / theta[0] - 1),
        np.abs(theta[0] / theta[1] - 1)
    ]))

    first_summand += np.log(w) + np.log(alpha)  # (alpha choose 1)
    first_summand = logsumexp(first_summand, b=theta)
    log_full_series = logsumexp([log_full_series, first_summand],
                                b=[1, -1])

    # Multiply all terms with 4, instead of just trailing ones
    log_full_series += np.log(4)
    # Eliminate unnecessary factor 4 in first summand
    log_full_series = logsumexp([log_full_series, np.log(3)],
                                b=[1, -1])

    # Replace j=2 summand, when pure DP bound is tighter
    log_second_summand = np.log(
        4 * (np.exp(base_mechanism.log_expectation(np.array(2))) - 1))

    log_second_summand += np.log(w) * 2 + log_binomial_coefficient(alpha, 2)
    log_second_summand = float(log_second_summand)

    log_alternative_second_summand = base_mechanism.log_expectation(np.array(2))

    log_alternative_second_summand += min(
        np.log(2),
        np.log(np.exp(base_mechanism.epsilon_dp()) - 1) * 2
    )

    log_alternative_second_summand += np.log(w) * 2 + log_binomial_coefficient(alpha, 2)
    log_alternative_second_summand = float(log_alternative_second_summand)

    if log_alternative_second_summand < log_second_summand:
        log_full_series = logsumexp(
            [log_full_series, log_second_summand, log_alternative_second_summand],
            b=[1, -1, 1])

    return log_full_series / (alpha - 1)


def _rdp_self_consistency_quadrature_gaussian(
        alpha, base_mechanism, dataset_size, batch_size, eval_params):
    """Evaluates Theorem 27 via quadrature for Gaussian mechanism.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        group_size (int): Number of substituted elements.
        eval_params (None | dict): Extra parameters for the specific eval method.

    Returns:
        float: RDP parameter epsilon of the WOr subsampled mechanism w.r.t induced distance.
    """

    if (eval_params is None) or ('dps' not in eval_params):
        raise ValueError('You must provide the \'dps\' parameter in eval_params.')

    w = batch_size / dataset_size

    def integrand_series(x):
        p = mpm.npdf(x, 0, base_mechanism.standard_deviation)
        q = mpm.npdf(x, 1, base_mechanism.standard_deviation)

        return ((1 + w * mpm.fabs(p / q - 1)) ** alpha) * q

    with mpm.workdps(eval_params['dps']):
        log_full_series = float(mpm.log(
            mpm.quad(integrand_series, [-np.inf, np.inf])))

    # Set j=1 summand to zero
    def integrand_first_summand(x):
        p = mpm.npdf(x, 0, base_mechanism.standard_deviation)
        q = mpm.npdf(x, 1, base_mechanism.standard_deviation)

        return (mpm.fabs(p / q - 1)) * q

    with mpm.workdps(eval_params['dps']):
        first_summand = float(mpm.log(
            mpm.quad(integrand_first_summand, [-np.inf, np.inf])))

    first_summand += np.log(w) + np.log(alpha)  # (alpha choose 1)
    log_full_series = logsumexp([log_full_series, first_summand],
                                b=[1, -1])

    # Multiply all terms with 4, instead of just trailing ones
    log_full_series += np.log(4)
    # Eliminate unnecessary factor 4 in j=0 summand
    log_full_series = logsumexp([log_full_series, np.log(3)],
                                b=[1, -1])

    # Replace j=2 summand, when pure DP bound is tighter
    log_second_summand = np.log(
        4 * (np.exp(base_mechanism.log_expectation(np.array(2))) - 1))

    log_second_summand += np.log(w) * 2 + log_binomial_coefficient(alpha, 2)
    log_second_summand = float(log_second_summand)

    log_alternative_second_summand = base_mechanism.log_expectation(np.array(2))

    log_alternative_second_summand += min(
        np.log(2),
        np.log(np.exp(base_mechanism.epsilon_dp()) - 1) * 2
    )

    log_alternative_second_summand += np.log(w) * 2 + log_binomial_coefficient(alpha, 2)
    log_alternative_second_summand = float(log_alternative_second_summand)

    if log_alternative_second_summand < log_second_summand:
        log_full_series = logsumexp(
            [log_full_series, log_second_summand, log_alternative_second_summand],
            b=[1, -1, 1])

    return log_full_series / (alpha - 1)



def rdp_tight_group(alpha: int, base_mechanism: BaseMechanism,
                    dataset_size: int, batch_size: int,
                    group_size: int,
                    eval_method: str, eval_params: None | dict) -> float:
    """Tight group privacy amplification via WOR subsampling.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        group_size (int): Number of substituted elements.
        eval_method: Which method to use for evaluating the divergence.
            Options:
                - quadrature: Numerical integration
                - directtransport: Upper bound using optimal transport without conditioning
        eval_params (None | dict): Extra parameters for the specific eval method.

    Returns:
        float: RDP parameter epsilon of the WOR subsampled mechanism w.r.t induced distance.
    """
    valid_methods = ['quadrature', 'directtransport']

    eval_method = eval_method.lower()
    if eval_method not in valid_methods:
        raise ValueError(f'eval_method {eval_method} is not in {valid_methods}')

    if eval_method == 'quadrature':
        return _rdp_tight_group_quadrature(alpha, base_mechanism, dataset_size, batch_size,
                                           group_size, eval_params)

    if eval_method == 'directtransport':
        return _rdp_tight_group_direct_transport(alpha, base_mechanism, dataset_size, batch_size,
                                                 group_size)

    else:
        assert False


def _rdp_tight_group_quadrature(
        alpha: int, base_mechanism: BaseMechanism, dataset_size: int, batch_size: int,
        group_size: int, eval_params: None | dict) -> float:
    """Tight group privacy amplification for WOR subsampling via quadrature.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        group_size (int): Number of substituted elements.
        eval_params (None | dict): Extra parameters for the specific mechanism.

    Returns:
        float: RDP parameter epsilon of the WOr subsampled mechanism w.r.t induced distance.
    """

    if isinstance(base_mechanism, RandomizedResponseMechanism):

        return _rdp_tight_group_quadrature_randomized_response(
            alpha, base_mechanism, dataset_size, batch_size, group_size)

    elif isinstance(base_mechanism, GaussianMechanism):

        return _rdp_tight_group_quadrature_gaussian(
            alpha, base_mechanism, dataset_size, batch_size, group_size, eval_params)

    else:
        raise NotImplementedError('Only Gaussian and Randomized Response is supported.')


def _rdp_tight_group_quadrature_randomized_response(
        alpha: int, base_mechanism: RandomizedResponseMechanism,
        dataset_size: int, batch_size: int,
        group_size: int) -> float:
    """Tight group privacy amplification for WOR subsampled randomized response via quadrature.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        group_size (int): Number of substituted elements.

    Returns:
        float: RDP parameter epsilon of the WOR subsampled mechanism w.r.t induced distance.
    """

    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    log_w = hypergeom.logpmf(np.arange(group_size + 1),
                             dataset_size, group_size, batch_size)  # (group_size + 1)

    log_theta = np.log(np.array([1 - base_mechanism.true_response_prob,
                       base_mechanism.true_response_prob]))  # (2)

    # (2^{2 * (group_size + 1})) x (2 * (group_size + 1))
    idx = np.array(list(itertools.product([0, 1], repeat=2 * (group_size + 1))))

    idx_p = idx[:, :(group_size + 1)]
    idx_q = idx[:, (group_size + 1):]
    idx_q[:, 0] = idx_p[:, 0]

    # 2 x (2^{2 * (group_size + 1)}) x (group_size + 1)
    ps = np.stack([log_theta[idx_p], log_theta[1 - idx_p]], axis=0)
    qs = np.stack([log_theta[idx_q], log_theta[1 - idx_q]], axis=0)

    res = logsumexp(log_w + ps, axis=2) * alpha  # 2 x (2^{2 * (group_size + 1)})
    res -= logsumexp(log_w + qs, axis=2) * (alpha - 1)

    res = logsumexp(res, axis=0)  # (2^{2 * (group_size + 1)})

    return res.max() / (alpha - 1)


def _rdp_tight_group_quadrature_gaussian(
        alpha: int, base_mechanism: GaussianMechanism,
        dataset_size: int, batch_size: int,
        group_size: int, eval_params: dict) -> float:
    """Tight group privacy amplification for WOR subsampled Gaussian via quadrature.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        group_size (int): Number of substituted elements.
        eval_params (None | dict): Dictionary of params that should contain bool "relax_distances",
            i.e., whether to double pairwise distance constraints between
            clean and perturbed mixture components from max(i, j) to 2 * max(i, j),
            which enables 1D numerical integration.

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if (eval_params is None) or ('dps' not in eval_params):
        raise ValueError('You must provide the \'dps\' parameter in eval_params.')
    if (eval_params is None) or ('relax_distances' not in eval_params):
        raise ValueError('You must provide the \'relax_distances\' parameter in eval_params.')

    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    w = hypergeom.pmf(np.arange(group_size + 1), dataset_size, group_size, batch_size)

    def integrand_relaxed(z):
        densities_p = np.array([mpm.npdf(z, k, base_mechanism.standard_deviation)
                                for k in range(group_size + 1)])

        densities_q = np.array([mpm.npdf(z, -k, base_mechanism.standard_deviation)
                                for k in range(group_size + 1)])

        p = w @ densities_p
        q = w @ densities_q
        ratio = p / q
        return (ratio ** alpha) * q

    def integrand_relaxed_one_mixture(z, mixture_p):
        if mixture_p:
            n_components_p = group_size
            n_components_q = 1
        else:
            n_components_p = 1
            n_components_q = group_size

        w_p = hypergeom.pmf(np.arange(n_components_p + 1), dataset_size, n_components_p, batch_size)
        w_q = hypergeom.pmf(np.arange(n_components_q + 1), dataset_size, n_components_q, batch_size)

        densities_p = np.array([mpm.npdf(z, k, base_mechanism.standard_deviation)
                                for k in range(n_components_p + 1)])

        densities_q = np.array([mpm.npdf(z, k, base_mechanism.standard_deviation)
                                for k in range(n_components_q + 1)])

        p = w_p @ densities_p
        q = w_q @ densities_q
        ratio = p / q
        return (ratio ** alpha) * q

    def integrand_two_mixtures(z_1, z_2):
        densities_p = np.array([mpm.npdf(z_1, k, base_mechanism.standard_deviation)
                                for k in range(group_size + 1)])

        densities_p *= mpm.npdf(z_2, 0, base_mechanism.standard_deviation)

        densities_q = np.array([mpm.npdf(z_1, k / 2, base_mechanism.standard_deviation)
                                for k in range(group_size + 1)])

        densities_q *= np.array([mpm.npdf(z_2, k * np.sqrt(0.75), base_mechanism.standard_deviation)
                                for k in range(group_size + 1)])

        p = w @ densities_p
        q = w @ densities_q
        ratio = p / q
        return (ratio ** alpha) * q

    with mpm.workdps(eval_params['dps']):
        relax_distances = eval_params['relax_distances']

        if relax_distances:
            res = mpm.log(mpm.quad(integrand_relaxed, [-mpm.inf, mpm.inf])) / (alpha - 1)
        else:

            res_mixture_both = mpm.quad(integrand_two_mixtures,
                                        [-mpm.inf, mpm.inf], [-mpm.inf, mpm.inf])

            res_mixture_p = mpm.quad(lambda z: integrand_relaxed_one_mixture(z, True),
                                     [-mpm.inf, mpm.inf])

            res_mixture_q = mpm.quad(lambda z: integrand_relaxed_one_mixture(z, False),
                                     [-mpm.inf, mpm.inf])

            res = max(res_mixture_both, res_mixture_p, res_mixture_q)
            res = mpm.log(res) / (alpha - 1)

        return float(res)


def _rdp_tight_group_direct_transport(
        alpha: int, base_mechanism: BaseMechanism,
        dataset_size: int, batch_size: int,
        group_size: int) -> float:
    """Group privacy amplification for WOR subsampled randomized response via transport.

    Corresponds to Theorem 9 / Lemma 5 of (Daigavane et al, 2022).


    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        dataset_size (float): Number of elemetns in the dataset.
        batch_size (int): Number of elements in each WOR-sampled batch.
        group_size (int): Number of substituted elements.

    Returns:
        float: RDP parameter epsilon of the WOR subsampled mechanism w.r.t induced distance.
    """

    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    n_sampled = np.arange(group_size + 1)  # (group_size + 1)

    log_w = np.array([hypergeom.logpmf(k, dataset_size, group_size, batch_size)
                      for k in n_sampled])

    res = log_w

    if isinstance(base_mechanism, RandomizedResponseMechanism):
        res[1:] += base_mechanism.log_expectation(np.array(alpha))

    elif isinstance(base_mechanism, GaussianMechanism):

        res += base_mechanism.log_expectation(
            np.full_like(n_sampled, alpha, dtype=float),
            n_sampled.astype(float))

    else:
        raise NotImplementedError('Currently only support RandomizedResponse and GaussianMechanism')

    return logsumexp(res) / (alpha + 1)
