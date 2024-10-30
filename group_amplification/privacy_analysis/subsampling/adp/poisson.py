__all__ = ['adp_traditional', 'adp_traditional_group', 'adp_tight_group']

import mpmath as mpm
import numpy as np
from scipy.stats import bernoulli, binom

from group_amplification.privacy_analysis.base_mechanisms import (BaseMechanism, GaussianMechanism,
                                                                  LaplaceMechanism,
                                                                  RandomizedResponseMechanism)

from group_amplification.privacy_analysis.composition.pld.accounting import (
    DoubleMixtureGaussianPrivacyLoss, DoubleMixtureLaplacePrivacyLoss
)


def adp_traditional(eps: float, base_mechanism: BaseMechanism, subsampling_rate: float) -> float:
    """Amplification by Poisson subsampling guarantee.

    This corresponds to Theorem 8 in Balle et al.

    Args:
        alpha (int): First ADP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.

    Returns:
        float: ADP parameter delta of the Poisson subsampled mechanism.
    """
    if eps < 0:
        raise ValueError('Only eps >= 0 is supported.')

    # See Proof of Proposition 30 in Characteristic Function Accounting paper
    individual_alpha = 1 + (np.exp(eps) - 1) / subsampling_rate

    return subsampling_rate * float(base_mechanism.adp(np.array(individual_alpha)))


def adp_traditional_group(eps: int, base_mechanism: BaseMechanism, subsampling_rate: float,
                          group_size: int,
                          eval_method: str) -> float:
    """Group privacy property of adp applied to amplification by Poisson subsampling.

    This corresponds to Lemma 2.2 in (Vadhan) and Theorem 8 in (Balle et al.)

    Args:
        eps (int): First ADP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        group_size (int): Number of inserted / removed elements.
        eval_method: Which flavor of group privacy to use.
            Options:
                - simple: Last inequality in Lemma 2.2
                - improved: Second-to-last inequality in Lemma 2.2

    Returns:
        float: ADP parameter delta of the Poisson subsampled mechanism w.r.t induced distance.
    """
    if eps < 0:
        raise ValueError('Only eps >= 0 is supported.')

    individual_eps = eps / group_size
    base_delta = adp_traditional(individual_eps, base_mechanism, subsampling_rate)

    if eval_method not in ['simple', 'improved']:
        raise ValueError('Eval method must  be in ["simple", "improved"]')

    if eval_method == 'simple':
        delta_scale = group_size * np.exp(group_size * individual_eps)
    else:
        delta_scale = np.sum(np.exp(np.arange(group_size) * individual_eps))

    return base_delta * delta_scale


def adp_tight_group(eps: int, base_mechanism: BaseMechanism, subsampling_rate: float,
                    insertions: int, deletions: int,
                    eval_method: str, eval_params: None | dict) -> float:
    """Tight group privacy amplification via Poisson subsampling.

    Args:
        eps (int): First ADP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_method: Which method to use for evaluating the divergence.
            Options:
                - quadrature: Mechanism-specific
                - maximalcoupling: Mechanism-agnostic
        eval_params (None | dict): Extra parameters for the specific eval method.

    Returns:
        float: ADP parameter delta of the Poisson subsampled mechanism w.r.t induced distance.
    """
    valid_methods = ['quadrature', 'maximalcoupling', 'bisection']

    eval_method = eval_method.lower()
    if eval_method not in valid_methods:
        raise ValueError(f'eval_method {eval_method} is not in {valid_methods}')

    if eval_method == 'quadrature':
        return _adp_tight_group_quadrature(eps, base_mechanism, subsampling_rate,
                                           insertions, deletions, eval_params)

    elif eval_method == 'maximalcoupling':
        return _adp_maximal_coupling_group(eps, base_mechanism, subsampling_rate,
                                           insertions, deletions)

    elif eval_method == 'bisection':
        return _adp_tight_group_bisection(eps, base_mechanism, subsampling_rate,
                                          insertions, deletions, eval_params)

    else:
        assert False


def _adp_tight_group_quadrature(
        eps: int, base_mechanism: BaseMechanism, subsampling_rate: float,
        insertions: int, deletions: int, eval_params: None | dict) -> float:
    """Tight group privacy amplification for Poisson subsampling via quadrature.

    Args:
        eps (int): First adp parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_params (None | dict): Extra parameters for
            _adp_tight_group_quadrature_gaussian and _adp_tight_group_quadrature_gaussian.

    Returns:
        float: adp parameter epsilon of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if isinstance(base_mechanism, GaussianMechanism):

        return _adp_tight_group_quadrature_gaussian(
            eps, base_mechanism, subsampling_rate, insertions, deletions, eval_params)

    elif isinstance(base_mechanism, LaplaceMechanism):
        return _adp_tight_group_quadrature_laplace(
            eps, base_mechanism, subsampling_rate, insertions, deletions, eval_params)

    elif isinstance(base_mechanism, RandomizedResponseMechanism):
        return _adp_tight_group_quadrature_randomized_response(
            eps, base_mechanism, subsampling_rate, insertions, deletions)

    else:
        raise NotImplementedError('Only Gaussian, Laplace, and Randomized Respons mechanism '
                                  'are supported.')


def _adp_tight_group_quadrature_gaussian(
        eps: int, base_mechanism: GaussianMechanism, subsampling_rate: float,
        insertions: int, deletions: int, eval_params: dict) -> float:
    """Tight group privacy amplification for Poisson subsampled Gaussian via quadrature.

    This corresponds to Theorem 3.8 in our paper.

    Args:
        eps (int): First adp parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_params (None | dict): Dictionary of parameters that should contain "dps",
            i.e., decimal precision for mpmath.

    Returns:
        float: adp parameter delta of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if (eval_params is None) or ('dps' not in eval_params):
        raise ValueError('You must provide the \'dps\' parameter in eval_params.')

    if eps < 0:
        raise ValueError('Only eps >= 0 is supported.')
    alpha = np.exp(eps)

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

        if ratio <= alpha:
            return 0
        else:
            return p - q * alpha

    with mpm.workdps(eval_params['dps']):
        res = mpm.quad(integrand, [-mpm.inf, mpm.inf])
        return float(res)


def _adp_tight_group_quadrature_laplace(
        eps: int, base_mechanism: LaplaceMechanism, subsampling_rate: float,
        insertions: int, deletions: int, eval_params: dict) -> float:
    """Tight group privacy amplification for Poisson subsampled Laplace via quadrature.

    Args:
        eps (int): First adp parameter.
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

    Returns:
        float: adp parameter delta of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if (eval_params is None) or ('dps' not in eval_params):
        raise ValueError('You must provide the \'dps\' parameter in eval_params.')
    if (eval_params is None) or ('subdivide' not in eval_params):
        raise ValueError('You must provide the \'subdivide\' parameter in eval_params.')

    if eps < 0:
        raise ValueError('Only eps >= 0 is supported.')
    alpha = np.exp(eps)

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

        if ratio <= alpha:
            return 0
        else:
            return p - q * alpha

    with mpm.workdps(eval_params['dps']):
        with mpm.workdps(eval_params['dps']):

            if eval_params['subdivide']:

                discontinuities = np.arange(-1 * insertions, deletions + 1)

                res = mpm.quad(integrand, [-mpm.inf, discontinuities[0]])
                for x in discontinuities[:-1]:
                    res += mpm.quad(integrand, [x, x+1])
                res += mpm.quad(integrand, [discontinuities[-1], mpm.inf])

            else:

                res = mpm.quad(integrand, [-mpm.inf, mpm.inf])

            return float(res)


def _adp_tight_group_quadrature_randomized_response(
        eps: int, base_mechanism: RandomizedResponseMechanism, subsampling_rate: float,
        insertions: int, deletions: int) -> float:
    """Tight group privacy amplification for Poisson subsampled randomized response via quadrature.

    Args:
        alpha (int): First adp parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.

    Returns:
        float: adp parameter delta of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if eps < 0:
        raise ValueError('Only eps >= 0 is supported.')
    alpha = np.exp(eps)

    w_p = 1 - binom.pmf(0,
                        deletions, subsampling_rate)  # (deletions + 1)

    w_q = 1 - binom.pmf(0,
                        insertions, subsampling_rate)  # (insertions + 1)

    true_response_prob = base_mechanism.true_response_prob

    def integrand(z: int, forward=True):
        if forward:
            p = ((1 - w_p) * bernoulli.pmf(z, true_response_prob)
                 + w_p * bernoulli.pmf(z, 1 - true_response_prob))

            q = bernoulli.pmf(z, true_response_prob)
        else:
            p = bernoulli.pmf(z, true_response_prob)

            q = ((1 - w_q) * bernoulli.pmf(z, true_response_prob)
                 + w_q * bernoulli.pmf(z, 1 - true_response_prob))

        return np.maximum(p - q * alpha, 0)

    div_forward = integrand(0, True) + integrand(1, True)
    div_backward = integrand(0, False) + integrand(1, False)

    return max(div_forward, div_backward)


def _adp_maximal_coupling_group(eps: int, base_mechanism: BaseMechanism, subsampling_rate: float,
                                insertions: int, deletions: int) -> float:
    """Group privacy amplification via Poisson subsampling.

    This is analyzed via maximal coupling and thus weaker.

    Args:
        eps (int): First adp parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_params (None | dict): Extra parameters for the specific eval method.

    Returns:
        float: eps parameter delta of the Poisson subsampled mechanism w.r.t induced distance.
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

    sensitivities = np.arange(1, group_size + 1)
    # See Proof of Proposition 30 in Characteristic Function Accounting paper
    individual_alpha = 1 + (np.exp(eps) - 1) / p_not_zero
    individual_alphas = np.full_like(sensitivities, individual_alpha, dtype='float')

    if isinstance(base_mechanism, GaussianMechanism):
        adps = base_mechanism.adp(individual_alphas, sensitivities)

    elif isinstance(base_mechanism, LaplaceMechanism):
        adps = base_mechanism.adp(individual_alphas, sensitivities)

    elif isinstance(base_mechanism, RandomizedResponseMechanism):
        adps = base_mechanism.adp(individual_alphas)

    else:
        raise ValueError('Only support Gaussian, Laplace, and Randomized Response Mechanisms')

    return p_not_zero * w @ adps


def _adp_tight_group_bisection(
        eps: int, base_mechanism: BaseMechanism, subsampling_rate: float,
        insertions: int, deletions: int, eval_params: None | dict) -> float:
    """Tight group privacy amplification for Poisson subsampling via bisection.

    Args:
        eps (int): First adp parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_params (None | dict): Extra parameters for
            _adp_tight_group_bisection_gaussian and _adp_tight_group_bisection_gaussian.

    Returns:
        float: adp parameter epsilon of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if isinstance(base_mechanism, GaussianMechanism):
        return _adp_tight_group_bisection_gaussian(
            eps, base_mechanism, subsampling_rate, insertions, deletions, eval_params)

    elif isinstance(base_mechanism, LaplaceMechanism):
        return _adp_tight_group_bisection_laplace(
            eps, base_mechanism, subsampling_rate, insertions, deletions, eval_params)

    else:
        raise NotImplementedError('Only Gaussian and Laplace mechanism '
                                  'are supported.')


def _adp_tight_group_bisection_gaussian(
        eps: int, base_mechanism: GaussianMechanism, subsampling_rate: float,
        insertions: int, deletions: int, eval_params: dict) -> float:
    """Tight group privacy amplification for Poisson subsampled Gaussian via bisection.

    This corresponds to Theorem 3.8 in our paper.

    Args:
        eps (int): First adp parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_params (None | dict): Params for DoubleMixtureGaussianMechanism

    Returns:
        float: adp parameter delta of the Poisson subsampled mechanism w.r.t induced distance.
    """

    if eps < 0:
        raise ValueError('Only eps >= 0 is supported.')

    sensitivities_upper = np.arange(deletions + 1)
    sampling_probs_upper = binom.pmf(sensitivities_upper, deletions, subsampling_rate)

    sensitivities_lower = np.arange(insertions + 1)
    sampling_probs_lower = binom.pmf(sensitivities_lower, insertions, subsampling_rate)

    privacy_loss = DoubleMixtureGaussianPrivacyLoss(
        base_mechanism.standard_deviation,
        sensitivities_upper, sensitivities_lower,
        sampling_probs_upper, sampling_probs_lower,
        **eval_params
    )

    return privacy_loss.get_delta_for_epsilon(eps)


def _adp_tight_group_bisection_laplace(
        eps: int, base_mechanism: LaplaceMechanism, subsampling_rate: float,
        insertions: int, deletions: int, eval_params: dict) -> float:
    """Tight group privacy amplification for Poisson subsampled Laplace via bisection.

    Args:
        eps (int): First adp parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        subsampling_rate (float): Probability of adding an element to the batch.
        insertions (int): Number of inserted elements.
            insertions+1 is number of mixture components in denominator.
        deletions (int): Number of deleted elements.
            deletions+1 is number of mixture components in numerator.
        eval_params (None | dict): Params for DoubleMixtureLaplaceMechanism

    Returns:
        float: adp parameter delta of the Poisson subsampled mechanism w.r.t induced distance.
    """

    sensitivities_upper = np.arange(deletions + 1)
    sampling_probs_upper = binom.pmf(sensitivities_upper, deletions, subsampling_rate)

    sensitivities_lower = np.arange(insertions + 1)
    sampling_probs_lower = binom.pmf(sensitivities_lower, insertions, subsampling_rate)

    privacy_loss = DoubleMixtureLaplacePrivacyLoss(
        base_mechanism.scale,
        sensitivities_upper, sensitivities_lower,
        sampling_probs_upper, sampling_probs_lower,
        **eval_params
    )

    return privacy_loss.get_delta_for_epsilon(eps)
