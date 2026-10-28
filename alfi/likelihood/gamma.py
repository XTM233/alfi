from gpytorch.likelihoods import Likelihood, _OneDimensionalLikelihood
# from gpytorch import base_distributions
import torch
import math
import gpytorch
from gpytorch.distributions import MultivariateNormal
from torch.distributions import constraints
from torch.nn import Parameter
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform


# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Gamma"]


def _standard_gamma(concentration):
    return torch._standard_gamma(concentration)

class Gammalikelihood(_OneDimensionalLikelihood):
    """
    Implements the Gamma likelihood used for GP regression.
    
    :param alpha: shape parameter, assume alpha=beta
    :param invlink: Inverse link function (default: torch.exp) to convert samples from the GP to the mean of the likelihood, ie positive
    :param scale: Scale factor to adjust the mean of the negative binomial
    """
    def __init__(self, alpha=1.0, invlink=torch.exp, mean=1.0, learnable_alpha=True, S=20):
        super().__init__()
        # TODO ensure that alpha is positive and is updated for each gene
        # TODO add alpha into parameter_trace
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=learnable_alpha) # dispersion
        self.mean = mean # by default 1.0 as gamma is multiplicative noise
        self.invlink = invlink
        # TODO test which takes control, lfm.config.num_samples or self.num_likelihood_samples
        self.num_likelihood_samples = S # the number of Monte Caro samples to approximate E_{q(f)}[logp p(y|f)]

    def forward(self, function_samples, **kwargs):
        """
        Computes the conditional distribution p(y | f).
        
        :param function_samples: Samples from the function (f), assuming the axis 0 is the sample dimension
        :return: Gamma distribution
        """
        if self.invlink is not None:
            m = self.invlink(function_samples)
        elif torch.any(function_samples <= 0):
            print("Warning: invlink is None, using function_samples directly, which may lead to negative values.")
            m = torch.clamp(function_samples, min=1e-5)  # ensure m is positive
        beta = self.alpha/m # rate parameter
        assert torch.all(beta > 0), "Beta parameter must be positive, got: {}".format(beta)
        assert torch.all(m > 0), "Mean parameter must be positive, got: {}".format(m)
        # TODO test if mask kwarg is needed
        return gpytorch.distributions.base_distributions.Gamma(
            concentration=self.alpha,
            rate=beta,
            validate_args=True)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        """
        Computes the log marginal likelihood of the approximate predictive distribution.
        
        :param observations: Values of y
        :param function_dist: Distribution for f(x)
        :return: Log marginal likelihood
        """
        marginal = self.marginal(function_dist, *args, **kwargs)
        return marginal.log_prob(observations)

    def marginal(self, function_dist, **kwargs):
        """
        Computes a predictive distribution p(y* | x*) given a posterior distribution p(f | D, x).
        
        :param function_dist: Distribution for f(x)
        :return: Negative Binomial distribution
        """
        # mean = function_dist.mean
        # var = function_dist.variance
        # r = 1 / self.alpha
        # if self.invlink == torch.exp:
        #     # TODO take in account scale parameter
        #     m = torch.exp(mean + var / 2)
        # else:
        #     raise NotImplementedError
        # p = r / (r + m)
        # return torch.distributions.NegativeBinomial(total_count=r, probs=p)
        raise NotImplementedError("Marginal distribution not implemented for Negative Binomial likelihood")

    def expected_log_prob(self, observations, function_dist, **kwargs):
        """
        Computes the expected log likelihood, where the expectation is over the GP variational distribution.
        
        :param observations: Values of y
        :param function_dist: Distribution for f(x) at (T, D), assume it's a MVN distribution
        :return: Expected log likelihood
        """
        # NOTE this is not the same as the Monte Carlo samples built from de_solve(h_0, f, t)
        # NOTE cannot parse **kwargs TypeError: NegativeBinomial.__init__() got an unexpected keyword argument 'mask'
        # TODO check shape of return
        function_samples = function_dist.rsample(sample_shape=torch.Size([self.num_likelihood_samples]))  # (S, T, D)
        print("function_samples shape", function_samples.shape)
        gamma_dist = self.forward(function_samples, **kwargs)
        try:
            assert torch.all(observations >= 0), "Observations must be positive, got: {}".format(observations)
        except AssertionError as e:
            print(observations[torch.where(observations < 0)])
            raise e
        log_probs = gamma_dist.log_prob(observations)  # (S, T, D)
        print("log_probs shape", log_probs.shape)
        # prevent overflow
        # log_probs[torch.where((observations == 0)*(gamma_dist.concentration <= 1e-5))] = 0
        log_probs = torch.clamp(log_probs, min=-1e5, max=1e5)
        return log_probs.mean(dim=0).sum(dim=-1)
    
    def double_sample(self, function_dist, total_num_samples):
        """
        Samples from the function distribution and returns the mean and variance.
        
        :param function_dist: Distribution for f(x)
        :param num_samples: Number of samples to draw
        :return: Mean and variance of the samples
        """
        # TODO test after tesing expected_log_prob
        sample_per_layer = math.ceil(math.sqrt(total_num_samples))
        function_samples = function_dist.rsample(sample_shape=torch.Size([int(sample_per_layer)]))  # (S, T, D)
        gamma_dist = self.forward(function_samples)
        samples = gamma_dist.rsample(sample_shape=torch.Size([sample_per_layer]))
        if total_num_samples != sample_per_layer **2:
            samples = torch.random.choice(samples, size=(total_num_samples,))
        return samples
