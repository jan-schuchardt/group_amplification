__all__ = ['rdp_traditional', 'rdp_traditional_group', 'rdp_tight_group']

import itertools
from warnings import warn

import mpmath as mpm
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

from group_amplification.privacy_analysis.base_mechanisms import (BaseMechanism, GaussianMechanism,
                                                                  LaplaceMechanism,
                                                                  RandomizedResponseMechanism)
from group_amplification.privacy_analysis.utils import (fixed_size_compositions,
                                                        log_binomial_coefficient,
                                                        log_multinomial_coefficient)


def rdp_traditional(alpha: int, base_mechanism: BaseMechanism, subsampling_rate: float,
                    eval_params: None | dict) -> float:
    """Amplification by Poisson subsampling guarantee.

    This corresponds to Theorem 6 in (Zhu & Wang, 2019).

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        subsampling_rate (float): Probability of adding an element to the batch.

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism.
    """
    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    if not any([isinstance(base_mechanism, mechanism_type)
                for mechanism_type in [GaussianMechanism, LaplaceMechanism,
                                       RandomizedResponseMechanism]]):
        warn('Guarantee only holds for non-negative Pearson-Vajda pseudo-divergence.')

    if int(alpha) != alpha:
        res_del = _rdp_tight_group_quadrature(alpha, base_mechanism, subsampling_rate, 0, 1, eval_params)
        res_ins = _rdp_tight_group_quadrature(alpha, base_mechanism, subsampling_rate, 1, 0, eval_params)
        warn('Using quadrature for traditional guarantee, because non-integer alpha.')
        return np.maximum(res_del, res_ins)

    log_w = np.log(np.array([1 - subsampling_rate, subsampling_rate]))

    # Pairs of non-negative integers that sum up to alpha
    compositions = fixed_size_compositions(alpha, 2)  # (alpha + 1) x 2

    log_summands = log_binomial_coefficient(alpha, compositions[:, 1])  # (alpha + 1)
    log_summands += (log_w * compositions).sum(axis=1)
    log_summands += base_mechanism.log_expectation(compositions[:, 1])

    return logsumexp(log_summands) / (alpha - 1)


def rdp_traditional_group(alpha: int, base_mechanism: BaseMechanism, subsampling_rate: float,
                          group_size: int,
                          eval_method: str, eval_params: None | dict) -> float:
    """Group privacy property of RDP applied to amplification by Poisson subsampling.

    This corresponds to Proposition 2/11 in (Mironov, 2017) and Theorem 6 in (Zhu & Wang, 2019)

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
            Should have non-negative Pearson-Vajda pseudo-divergence, see Theorem 8.
        subsampling_rate (float): Probability of adding an element to the batch.
        group_size (int): Number of inserted / removed elements.
        eval_method: Which flavor of group privacy to use.
            Options:
                - traditional: Proposition 2 from Mironov
                - recursive: Proposition 11 from Mironov.
                - gaussian: Assume group privacy property of Gaussian distribution
        eval_params (None | dict): Must specify "hoelders_exponent".
            This corresponds to p in Proposition 11 from (Mironov, 2017.)

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism w.r.t induced distance.
    """
    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    valid_methods = ['traditional', 'recursive', 'gaussian']

    eval_method = eval_method.lower()
    if eval_method not in valid_methods:
        raise ValueError(f'eval_method {eval_method} is not in {valid_methods}')

    if eval_method == 'traditional':
        if not np.log2(group_size).is_integer:
            warn('Group size must be power of 2 for proper upper bound to hold.')
        if alpha < 2 * group_size:
            warn('Group size must be at least double of alpha for proper upper bound to hold.')

        epsilon_scale = 3 ** np.log2(group_size)
        return epsilon_scale * rdp_traditional(group_size * alpha, base_mechanism, subsampling_rate, eval_params)

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
            return rdp_traditional(alpha, base_mechanism, subsampling_rate, eval_params)
        else:
            res = (alpha - 1 / p) / (alpha - 1)
            res *= rdp_traditional_group(
                        p * alpha, base_mechanism, subsampling_rate,
                        group_size // 2, eval_method, eval_params)

            res += rdp_traditional_group(
                        q * (alpha - 1 / p), base_mechanism, subsampling_rate,
                        group_size // 2, eval_method, eval_params)

            return res

    elif eval_method == 'gaussian':
        epsilon_scale = group_size
        return epsilon_scale * rdp_traditional(group_size * alpha, base_mechanism, subsampling_rate, eval_params)

    else:
        assert False


def rdp_tight_group(alpha: int, base_mechanism: BaseMechanism, subsampling_rate: float,
                    insertions: int, deletions: int,
                    eval_method: str, eval_params: None | dict) -> float:
    """Tight group privacy amplification via Poisson subsampling.

    This corresponds to Theorem 3.7 in our paper.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_method: Which method to use for evaluating the divergence.
            Options:
                - quadrature: Numerical integration
                - expansion: Multinomial expansion
        eval_params (None | dict): Extra parameters for the specific eval method.

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism w.r.t induced distance.
    """
    valid_methods = ['quadrature', 'expansion', 'maximalcoupling']

    eval_method = eval_method.lower()
    if eval_method not in valid_methods:
        raise ValueError(f'eval_method {eval_method} is not in {valid_methods}')

    if eval_method == 'quadrature':
        return _rdp_tight_group_quadrature(alpha, base_mechanism, subsampling_rate,
                                           insertions, deletions, eval_params)

    elif eval_method == 'expansion':
        if insertions > 0:
            raise ValueError('Multinomial expansion can only be used for deletions.')

        return _rdp_tight_group_expansion(alpha, base_mechanism, subsampling_rate, deletions)

    elif eval_method == 'maximalcoupling':
        return _rdp_maximal_coupling_group(alpha, base_mechanism, subsampling_rate,
                                           insertions, deletions, eval_params)

    else:
        assert False


def _rdp_tight_group_quadrature(
        alpha: int, base_mechanism: BaseMechanism, subsampling_rate: float,
        insertions: int, deletions: int, eval_params: None | dict) -> float:
    """Tight group privacy amplification for Poisson subsampling via quadrature.

    This corresponds to Theorem 3.8 in our paper.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_params (None | dict): Extra parameters for
            _rdp_tight_group_quadrature_gaussian and _rdp_tight_group_quadrature_gaussian.

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if isinstance(base_mechanism, GaussianMechanism):

        return _rdp_tight_group_quadrature_gaussian(
            alpha, base_mechanism, subsampling_rate, insertions, deletions, eval_params)

    elif isinstance(base_mechanism, LaplaceMechanism):
        return _rdp_tight_group_quadrature_laplace(
            alpha, base_mechanism, subsampling_rate, insertions, deletions, eval_params)

    elif isinstance(base_mechanism, RandomizedResponseMechanism):
        return _rdp_tight_group_quadrature_randomized_response(
            alpha, base_mechanism, subsampling_rate, insertions, deletions)

    else:
        raise NotImplementedError('Only Gaussian, Laplace, and Randomized Respons mechanism '
                                  'are supported.')


def _rdp_tight_group_quadrature_gaussian(
        alpha: int, base_mechanism: GaussianMechanism, subsampling_rate: float,
        insertions: int, deletions: int, eval_params: dict) -> float:
    """Tight group privacy amplification for Poisson subsampled Gaussian via quadrature.

    This corresponds to Theorem 3.8 in our paper.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_params (None | dict): Dictionary of parameters that should contain "dps",
            i.e., decimal precision for mpmath.

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if (eval_params is None) or ('dps' not in eval_params):
        raise ValueError('You must provide the \'dps\' parameter in eval_params.')

    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    w_p = binom.pmf(np.arange(deletions + 1), deletions, subsampling_rate)
    w_q = binom.pmf(np.arange(insertions + 1), insertions, subsampling_rate)

    def integrand(z):
        densities_p = np.array([mpm.npdf(z, k, base_mechanism.standard_deviation)
                                for k in range(deletions + 1)])

        densities_q = np.array([mpm.npdf(z, -1 * k, base_mechanism.standard_deviation)
                                for k in range(insertions + 1)])

        p = w_p @ densities_p
        q = w_q @ densities_q
        ratio = p / q

        return (ratio ** alpha) * q

    with mpm.workdps(eval_params['dps']):
        res = mpm.log(mpm.quad(integrand, [-mpm.inf, mpm.inf])) / (alpha - 1)
        return float(res)


def _rdp_tight_group_quadrature_laplace(
        alpha: int, base_mechanism: LaplaceMechanism, subsampling_rate: float,
        insertions: int, deletions: int, eval_params: dict) -> float:
    """Tight group privacy amplification for Poisson subsampled Laplace via quadrature.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_params (None | dict): Dictionary of parameters with entries:
            - dps (int): Decimal precision for mpmath
            - subdivide (bool): If True, partition integration region between
                discontinuities.
        that should contain "dps",
            i.e., decimal precision for mpmath.
            Should also contain the boolean "subdi

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if (eval_params is None) or ('dps' not in eval_params):
        raise ValueError('You must provide the \'dps\' parameter in eval_params.')
    if (eval_params is None) or ('subdivide' not in eval_params):
        raise ValueError('You must provide the \'subdivide\' parameter in eval_params.')

    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    w_p = binom.pmf(np.arange(deletions + 1), deletions, subsampling_rate)
    w_q = binom.pmf(np.arange(insertions + 1), insertions, subsampling_rate)

    def lpdf(z, loc, scale):
        return mpm.exp(-1 * mpm.fabs(z - loc) / scale) / (2 * scale)

    def integrand(z):
        densities_p = np.array([lpdf(z, k, base_mechanism.scale)
                                for k in range(deletions + 1)])

        densities_q = np.array([lpdf(z, -1 * k, base_mechanism.scale)
                                for k in range(insertions + 1)])

        p = w_p @ densities_p
        q = w_q @ densities_q
        ratio = p / q

        return (ratio ** alpha) * q

    with mpm.workdps(eval_params['dps']):

        if eval_params['subdivide']:

            discontinuities = np.arange(-1 * insertions, deletions + 1)

            try:
                res = mpm.quad(integrand, [-mpm.inf, discontinuities[0]])
                for x in discontinuities[:-1]:
                    res += mpm.quad(integrand, [x, x+1])
                res += mpm.quad(integrand, [discontinuities[-1], mpm.inf])
            except:
                return np.inf

        else:
            try:
                res = mpm.quad(integrand, [-mpm.inf, mpm.inf])
            except:
                return np.inf

        return float(mpm.log(res) / (alpha - 1))


def _rdp_tight_group_quadrature_randomized_response(
        alpha: int, base_mechanism: RandomizedResponseMechanism, subsampling_rate: float,
        insertions: int, deletions: int) -> float:
    """Tight group privacy amplification for Poisson subsampled randomized response via quadrature.

    This corresponds to Theorem 5.1 in our paper.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    log_w_p = binom.logpmf(np.arange(deletions + 1),
                           deletions, subsampling_rate)  # (deletions + 1)

    log_w_q = binom.logpmf(np.arange(insertions + 1),
                           insertions, subsampling_rate)  # (insertions + 1)

    log_theta = np.log(np.array([1 - base_mechanism.true_response_prob,
                       base_mechanism.true_response_prob]))  # (2)

    group_size = insertions + deletions
    # (2^{group_size + 2}) x (group_size + 1)
    idx = np.array(list(
            itertools.product([0, 1], repeat=(group_size + 2))))

    # (2^{group_size + 2}) x (deletions + 1)
    idx_p = idx[:, :(deletions + 1)]
    # (2^{group_size + 2}) x (insertions + 1)
    idx_q = idx[:, (deletions + 1):]
    idx_q[:, 0] = idx_p[:, 0]

    # 2 x (2^{group_size + 2}) x (deletions + 1)
    pmfs_p = np.stack([log_theta[idx_p], log_theta[1 - idx_p]], axis=0)
    # 2 x (2^{group_size + 2}) x (insertions + 1)
    pmfs_q = np.stack([log_theta[idx_q], log_theta[1 - idx_q]], axis=0)

    res = logsumexp(log_w_p + pmfs_p, axis=2) * alpha  # 2 x (2^{group_size + 2})
    res -= logsumexp(log_w_q + pmfs_q, axis=2) * (alpha - 1)  # 2 x (2^{group_size + 2})
    res = logsumexp(res, axis=0)  # (2^{group_size + 2})

    return res.max() / (alpha - 1)


def _rdp_tight_group_expansion(alpha: int, base_mechanism: BaseMechanism, subsampling_rate: float,
                               deletions: int) -> float:
    """Tight group privacy amplification for Poisson subsampling, via multinomial expansion.

    This corresponds to Appendix M.4 from our paper.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        deletions (int): Number of inserted elements.

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism w.r.t K inserts/deletes.
    """

    if not isinstance(base_mechanism, GaussianMechanism):
        raise NotImplementedError(
                'Multinomial expansion currently only supported for GaussianMechanism.')

    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    if not any([isinstance(base_mechanism, mechanism_type)
                for mechanism_type in [GaussianMechanism, RandomizedResponseMechanism]]):
        warn('Guarantee only holds for non-negative Pearson-Vajda pseudo-divergence.')

    log_w = binom.logpmf(np.arange(deletions + 1), deletions, subsampling_rate)

    # C tuples of length [group_size + 1] with non-negative integers that sum up to alpha
    compositions = fixed_size_compositions(alpha, deletions + 1)  # C x (group_size + 1)

    # Multinomial term
    log_summands = log_multinomial_coefficient(np.full(len(compositions), alpha),
                                               compositions)  # C

    # Probability term
    log_summands += (log_w * compositions).sum(axis=1)

    # The inner expectation
    original_means = np.arange(0, deletions + 1)  # (group_size + 1)
    interpolated_means = (compositions / alpha) @ original_means  # C
    interpolated_squared_means = (compositions / alpha) @ (original_means ** 2)  # C

    log_summands += base_mechanism.log_expectation(
        np.full_like(interpolated_means, alpha),
        interpolated_means)

    # Rescaling of the inner expectation, so that we have  valid normal distributions
    std = base_mechanism.standard_deviation

    log_summands += alpha / (2 * (std ** 2)) * (interpolated_means ** 2)
    log_summands -= alpha / (2 * (std ** 2)) * interpolated_squared_means

    return logsumexp(log_summands) / (alpha - 1)


def _rdp_tight_group_jensen(alpha: int, base_mechanism: GaussianMechanism, subsampling_rate: float,
                            group_size: int, eval_params: dict) -> float:
    """Upper bound on group amplification by Poisson subsampling, via Jensen's inequality.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        group_size (int): Number of inserted / removed elements.
        eval_params (dict): Dictionary of parameters that should contain "extract_squared_means"
            If True: Multiply exp(-1/(2 * sigma^2) mu_k^2) with weights before applying inequality.
            If False: Apply the inequality using only the mixture weights.

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism w.r.t K inserts/deletes.
    """

    if not isinstance(base_mechanism, GaussianMechanism):
        raise ValueError(
                'Jensens inequality only works with  GaussianMechanism.')

    if alpha <= 1:
        raise ValueError('Only alpha > 1 is supported.')

    if (eval_params is None) or ('extract_squared_means' not in eval_params):
        raise ValueError('You must provide the \'extract_squared_means\' parameter in eval_params.')

    extract_squared_means = eval_params['extract_squared_means']

    original_means = np.arange(0, group_size+1)  # (group_size + 1)
    std = base_mechanism.standard_deviation
    log_w = binom.logpmf(np.arange(group_size + 1), group_size, subsampling_rate)

    if extract_squared_means:
        log_w -= 1 / (2 * (std ** 2)) * (original_means ** 2)

    log_normalizer = logsumexp(log_w)
    log_w -= log_normalizer
    w = np.exp(log_w)  # (group_size + 1)

    interpolated_mean = w @ original_means
    res = base_mechanism.log_expectation(np.array([alpha]), np.array([interpolated_mean]))[0]

    # Normalize, so that we have a proper normal distribution in denominator
    res -= (alpha - 1) / (2 * (std ** 2)) * (interpolated_mean ** 2)

    if not extract_squared_means:
        interpolated_squared_means = w @ (original_means ** 2)
        res += (alpha - 1) / (2 * (std ** 2)) * (interpolated_squared_means)

    res -= log_normalizer * (alpha - 1)

    return res / (alpha - 1)


def _rdp_maximal_coupling_group(alpha: int, base_mechanism: BaseMechanism, subsampling_rate: float,
                               insertions: int, deletions: int,
                               eval_params: None | dict) -> float:
    """Group privacy amplification via Poisson subsampling.

    This is analyzed via maximal coupling and thus weaker.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_params (None | dict): Extra parameters for the specific eval method.

    Returns:
        float: RDP parameter epsilon of the Poisson subsampled mechanism w.r.t induced distance.
    """
    if (insertions > 0) and (deletions > 0):
        raise ValueError('Maximal coupling currently only supports insertion XOR deletion')
    if (insertions <= 0) and (deletions <= 0):
        raise ValueError('Either insertions or deletions must be greater 0')
    if (insertions < 0) or (deletions < 0):
        raise ValueError('Insertions and deletions must not be negative')

    if deletions > 0:
        group_size = deletions
    else:
        group_size = insertions

    w = binom.pmf(np.arange(1, group_size + 1), group_size, subsampling_rate)
    p_not_zero = 1 - binom.pmf(0, group_size, subsampling_rate)
    w /= p_not_zero

    if isinstance(base_mechanism, GaussianMechanism):
        rdps = np.array([
            rdp_traditional(alpha, GaussianMechanism(base_mechanism.standard_deviation / k),
                            p_not_zero, eval_params) for k in range(1, group_size+1)
        ])

    elif isinstance(base_mechanism, LaplaceMechanism):
        rdps = np.array([
            rdp_traditional(alpha, LaplaceMechanism(base_mechanism.scale / k),
                            p_not_zero, eval_params) for k in range(1, group_size+1)
        ])

    elif isinstance(base_mechanism, RandomizedResponseMechanism):
        rdps = np.array([
            rdp_traditional(alpha, base_mechanism,
                            p_not_zero, eval_params) for k in range(1, group_size+1)
        ])

    else:
        raise ValueError('Only support Gaussian, Laplace, and Randomized Response Mechanisms')

    log_expectations = rdps * (alpha - 1)

    return logsumexp(log_expectations, b=w) / (alpha - 1)
