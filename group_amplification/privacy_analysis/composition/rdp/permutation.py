__all__ = ['rdp_direct_transport', 'rdp_tight']

import mpmath as mpm
import numpy as np

from group_amplification.privacy_analysis.base_mechanisms import BaseMechanism, GaussianMechanism

from itertools import product


def rdp_direct_transport(alpha: float, base_mechanism: BaseMechanism,
                         n_chunks: int, n_iterations: int):
    """Loose composed privacy amplified via permutation + WOR Subsampling

    This corresponds to Appendix P.2 from our paper.
    It uses optimal transport without conditioning, so very simple.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        n_chunks (float): Number of initial batches that are permuted.
        n_iterations (int): Number of batches that are subsampled WOR.

    Returns:
        float: RDP parameter epsilon of the composed mechanism.
    """

    if n_iterations != 2:
        raise NotImplementedError('Currently only support exactly 2 iterations')

    if n_iterations != n_chunks:
        raise NotImplementedError('Subsampling+Permutation not implemented yet')

    return float(base_mechanism.rdp(np.array(alpha)))


def rdp_tight(alpha: float, base_mechanism: BaseMechanism,
              n_chunks: int, n_iterations: int, eval_params: dict):
    """Tight composed privacy amplified via permutation + WOR Subsampling

    This corresponds to Appendix P.3 from our paper.
    Currently assumes that we have Gaussian mechanism applied to function
    with codomain {0, 1}.

    Args:
        alpha (int): First RDP parameter.
        base_mechanism (BaseMechanism): Base mechanisms that is being amplified.
        n_chunks (float): Number of initial batches that are permuted.
        n_iterations (int): Number of batches that are subsampled WOR.
        eval_params (dict: Dictionary of parameters that should contain "dps",
            i.e., decimal precision for mpmath.

    Returns:
        float: RDP parameter epsilon of the composed mechanism.
    """

    if n_iterations != 2:
        raise NotImplementedError('Currently only support exactly 2 iterations')

    if n_iterations != n_chunks:
        raise NotImplementedError('Subsampling+Permutation not implemented yet')

    if not isinstance(base_mechanism, GaussianMechanism):
        raise NotImplementedError('Currently only support Gaussian mechanism')

    def integrand(z_1, z_2, loc_p, loc_q, loc_r):
        p_1 = mpm.npdf(z_1, loc_p, base_mechanism.standard_deviation)  # Element sampled clean
        p_2 = mpm.npdf(z_2, loc_p, base_mechanism.standard_deviation)

        q_1 = mpm.npdf(z_1, loc_q, base_mechanism.standard_deviation)  # Element not sampled
        q_2 = mpm.npdf(z_2, loc_q, base_mechanism.standard_deviation)

        r_1 = mpm.npdf(z_1, loc_r, base_mechanism.standard_deviation)  # Element sampled perturbed
        r_2 = mpm.npdf(z_2, loc_r, base_mechanism.standard_deviation)

        numerator = (p_1 * q_2 + q_1 * p_2) / 2
        denominator = (r_1 * q_2 + q_1 * r_2) / 2

        ratio = numerator / denominator

        return (ratio ** alpha) * denominator

    def integral(loc_p, loc_q, loc_r):
        with mpm.workdps(eval_params['dps']):
            res = mpm.quad(
                (lambda z_1, z_2: integrand(z_1, z_2, loc_p, loc_q, loc_r)),
                [-np.inf, np.inf], [-np.inf, np.inf]
            )

            res = mpm.log(res) / (alpha - 1)
            return float(res)

    best_res = 0

    for loc_p, loc_q, loc_r in product([0, 1], repeat=3):
        if (loc_p == loc_q) and (loc_q == loc_r):
            continue

        new_res = integral(loc_p, loc_q, loc_r)
        best_res = max(best_res, new_res)

    return best_res
