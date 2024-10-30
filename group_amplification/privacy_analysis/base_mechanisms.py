from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import laplace, norm


class BaseMechanism(ABC):

    @abstractmethod
    def epsilon_dp(self) -> float:
        """Return smallest epsilon s.t. mechanisms is epsilon-DP

        Returns:
            float: Epsilon parameter of tradtional DP.
        """

    @abstractmethod
    def adp(self, alpha: NDArray) -> NDArray[np.float64]:
        """Return smallest delta s.t. mechanisms is (log(alpha), epsilon)-ADP.

        Args:
            alpha (NDArray): Array of arbitrary shape

        Returns:
            NDArray[np.float64]: Epsilons of same shape as alpha
        """

    @abstractmethod
    def rdp(self, alpha: NDArray) -> NDArray[np.float64]:
        """Return smallest epsilon s.t. mechanisms is (alpha, epsilon)-RDP.

        Args:
            alpha (NDArray): Array of arbitrary shape

        Returns:
            NDArray[np.float64]: Epsilons of same shape as alpha
        """

    def log_expectation(self, alpha: NDArray, *args, **kwargs) -> NDArray[np.float64]:
        """Return worst-case log(Phi_alpha), with Phi_alpha defined in our paper.

        Args:
            alpha (NDArray): Array of arbitrary shape
            *args: Variable length argument list passed to rdp()
            **kwargs: Arbitrary keyword arguments passed to rdp()

        Returns:
            NDArray[np.float64]: log(Phi_alpha)s of same shape as alpha
        """
        if np.any((alpha < 1) & (alpha != 0)):
            raise ValueError('Log expectation only considers alpha in {0} and [1, infty)')
        else:
            res = np.zeros_like(alpha, dtype=float)

            # E_q[(p/q)^a] is 1 for a in {0, 1} --> Logarithm is 0.
            zero_mask = (alpha == 0) | (alpha == 1)

            res[~zero_mask] = (alpha[~zero_mask] - 1) * self.rdp(alpha[~zero_mask], *args, **kwargs)

            return res


class RandomizedResponseMechanism(BaseMechanism):
    def __init__(self, true_response_prob: float) -> None:
        super().__init__()
        self.true_response_prob = true_response_prob

    def epsilon_dp(self) -> float:
        """Return smallest epsilon s.t. mechanisms is epsilon-DP

        Returns:
            float: Epsilon parameter of tradtional DP.
        """
        if self.true_response_prob in [0, 1]:
            return np.inf
        else:
            return max(
                np.log(self.true_response_prob) - np.log(1 - self.true_response_prob),
                np.log(1 - self.true_response_prob) - np.log(self.true_response_prob),
            )

    def adp(self, alpha: NDArray) -> NDArray[np.float64]:
        """Theorem 2  from "Privacy Profiles and Amplification by Subsampling"

        Args:
            alpha (float): Array of arbitrary shape

        Returns:
            NDArray[np.float64]: Deltas of same shape as alpha
        """

        if np.any(alpha < 0):
            raise ValueError('ADP only considers alpha >= 0.')

        p = self.true_response_prob

        res = np.maximum(p - alpha * (1 - p), 0)
        res += np.maximum((1 - p) - alpha * p, 0)

        return res

    def rdp(self, alpha: NDArray) -> NDArray[np.float64]:
        """Proposition 5 from Mironov et al. 2017.

        Args:
            alpha (float): Array of arbitrary shape

        Returns:
            NDArray[np.float64]: Epsilons of same shape as alpha
        """

        if np.any(alpha < 1):
            raise ValueError('RDP only considers alpha >= 1.')

        # alpha > 1 case
        summands = [
            (np.log(self.true_response_prob) * alpha
             - np.log(1 - self.true_response_prob) * (alpha - 1)),
            (np.log(1 - self.true_response_prob) * alpha
             - np.log(self.true_response_prob) * (alpha - 1))
        ]

        summands = np.stack(summands, axis=0)
        assert summands.shape[1:] == alpha.shape

        res = logsumexp(summands, axis=0) / (alpha - 1)

        # alpha = 1 case
        res[alpha == 1] = (
            (2 * self.true_response_prob - 1)
            * (np.log(self.true_response_prob)
               - np.log(1 - self.true_response_prob))
            )

        return res


class GaussianMechanism(BaseMechanism):

    def __init__(self, standard_deviation: float) -> None:
        super().__init__()
        self.standard_deviation = standard_deviation

    def epsilon_dp(self) -> float:
        """Return smallest epsilon s.t. mechanisms is epsilon-DP

        The Gaussian mechanism is not epsilon-DP

        Returns:
            float: Epsilon parameter of tradtional DP.
        """
        return np.inf

    def adp(self, alpha: NDArray,
            sensitivity: None | NDArray = None) -> NDArray[np.float64]:
        """Theorem 4  from "Privacy Profiles and Amplification by Subsampling"

        Args:
            alpha (float): Array of arbitrary shape
            sensitivity (None | NDArray, optional): L2 sensitivity of underlying function.
                Same shape as alpha. Assumed to be 1 when not specified.

        Returns:
            NDArray[np.float64]: Deltas of same shape as alpha
        """

        if np.any(alpha < 0):
            raise ValueError('ADP only considers alpha >= 0.')

        if sensitivity is None:
            sensitivity = np.ones_like(alpha, dtype=float)

        eps = np.log(alpha)

        res = norm.cdf(
            sensitivity / (2 * self.standard_deviation)
            - eps * self.standard_deviation / sensitivity
        )

        res -= alpha * norm.cdf(
            - sensitivity / (2 * self.standard_deviation)
            - eps * self.standard_deviation / sensitivity
        )

        return res

    def rdp(self, alpha: NDArray,
            sensitivity: None | NDArray = None) -> NDArray[np.float64]:
        """Proposition 7 from Mironov et al. 2017.

        Args:
            alpha (NDArray): Array of arbitrary shape
            sensitivity (None | NDArray, optional): L2 sensitivity of underlying function.
                Same shape as alpha. Assumed to be 1 when not specified.

        Returns:
            NDArray[np.float64]: log(Phi_alpha)s of same shape as alpha
        """
        if np.any(alpha < 1):
            raise ValueError('RDP only considers alpha >= 1.')
        else:
            if sensitivity is None:
                sensitivity = np.ones_like(alpha, dtype=float)

            return alpha * (sensitivity ** 2) / (2 * (self.standard_deviation ** 2))


class LaplaceMechanism(BaseMechanism):

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def epsilon_dp(self) -> float:
        """Return smallest epsilon s.t. mechanisms is epsilon-DP

        Returns:
            float: Epsilon parameter of tradtional DP.
        """
        return 1 / self.scale

    def adp(self, alpha: NDArray,
            sensitivity: None | NDArray = None) -> NDArray[np.float64]:
        """Theorem 3  from "Privacy Profiles and Amplification by Subsampling"

        Args:
            alpha (float): Array of arbitrary shape
            sensitivity (None | NDArray, optional): L2 sensitivity of underlying function.
                Same shape as alpha. Assumed to be 1 when not specified.

        Returns:
            NDArray[np.float64]: Deltas of same shape as alpha
        """

        if np.any(alpha < 0):
            raise ValueError('ADP only considers alpha >= 0.')

        if sensitivity is None:
            sensitivity = np.ones_like(alpha, dtype=float)

        eps = np.log(alpha)

        res = np.empty_like(alpha)

        always_clipped_mask = (eps > sensitivity / self.scale)
        never_clipped_mask = (eps < -1 * sensitivity / self.scale)
        remaining_mask = (~always_clipped_mask) & (~never_clipped_mask)

        res[always_clipped_mask] = 0
        res[never_clipped_mask] = 1 - alpha[never_clipped_mask]

        threshold = (sensitivity - eps * self.scale) / 2

        res[remaining_mask] = laplace.cdf(threshold[remaining_mask], loc=0, scale=self.scale)

        res[remaining_mask] -= (
            alpha[remaining_mask] * laplace.cdf(
                threshold[remaining_mask],
                loc=sensitivity[remaining_mask], scale=self.scale))

        return res

    def rdp(self, alpha: NDArray,
            sensitivity: None | NDArray = None) -> NDArray[np.float64]:
        """Proposition 6 from Mironov et al. 2017.

        Args:
            alpha (NDArray): Array of arbitrary shape
            sensitivity (None | NDArray, optional): L1 sensitivity of underlying function.
                Same shape as alpha. Assumed to be 1 when not specified.

        Returns:
            NDArray[np.float64]: log(Phi_alpha)s of same shape as alpha
        """
        if np.any(alpha < 1):
            raise ValueError('RDP only considers alpha >= 1.')
        else:
            if sensitivity is None:
                sensitivity = np.ones_like(alpha, dtype=float)

            adjusted_scale = self.scale / sensitivity

            log_summands = np.array([
                np.log(alpha) - np.log(2 * alpha - 1) + (alpha - 1) / adjusted_scale,
                np.log(alpha - 1) - np.log(2 * alpha - 1) - alpha / adjusted_scale
            ])

            return logsumexp(log_summands, axis=0) / (alpha - 1)
