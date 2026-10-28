from gpytorch.likelihoods import Likelihood, _OneDimensionalLikelihood
# from gpytorch import base_distributions
import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from torch.distributions import constraints
from torch.nn import Parameter
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform


import torch
from torch.special import digamma
from torch.special import gammainc as pgamma
from torch.special import gammaincc as pgamma_upper
from torch import lgamma

class ContinuousNegativeBinomial(torch.distributions.Distribution):
    """
    Continuous Negative Binomial distribution.
    
    :param alpha: Positive real number (overdispersion parameter)
    :param m: Non negative real number (mean parameter)
    """
    arg_constraints = {'alpha': constraints.positive, 'm': constraints.nonnegative}
    support = constraints.nonnegative
    has_rsample = False

    def __init__(self, alpha, m, validate_args=None):
        self.alpha = alpha
        self.m = m
        super().__init__(batch_shape=torch.Size(), validate_args=validate_args)
        self.r = 1 / self.alpha  
        self.p = self.r / (self.m + self.r)
        self.p = torch.clamp(self.p, min=1e-5, max=1-1e-5)

    def sample(self, sample_shape=torch.Size()):
        # return torch._standard_gamma(self.r, sample_shape) / self.beta
        raise NotImplementedError("Sampling is not implemented for ContinuousNegativeBinomial")
    def log_prob(self, value):

        return (
            lgamma(self.r + value) - lgamma(value + 1) - lgamma(self.r)
            + value * torch.log(1 - self.p)
            + self.r * torch.log(self.p)
            )
    def entropy(self):
        raise NotImplementedError("Entropy is not implemented for ContinuousNegativeBinomial")
    def mean(self):
        return self.m
    def variance(self):
        self.r = 1 / self.alpha  # shape parameter
        self.beta = self.r / self.m  # rate parameter
        return self.r / (self.beta ** 2) + self.r / (self.beta)

class CNegativeBinomialLikelihood(_OneDimensionalLikelihood):
    """
    Implements the Negative Binomial likelihood used for GP regression.
    
    :param alpha: Dispersion parameter, ie var = mu + mu**2 * alpha, alpha := 1/phi
    :param invlink: Inverse link function (default: torch.exp) to convert samples from the GP to the mean of the likelihood, ie positive
    :param scale: Scale factor to adjust the mean of the negative binomial
    """
    def __init__(self, alpha=1.0, invlink=torch.exp, scale=None, learnable_alpha=True, S=20):
        super().__init__()
        # TODO ensure that alpha is positive and is updated for each gene
        # TODO add alpha into parameter_trace
        self.raw_alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float64).log(), requires_grad=learnable_alpha)  # raw dispersion
        self.scaler = scale # a scaler object
        self.invlink = invlink
        self.num_likelihood_samples = S # the number of Monte Caro samples to approximate E_{q(f)}[logp p(y|f)]

    def forward(self, function_samples, **kwargs):
        """
        Computes the conditional distribution p(y | f).
        
        :param function_samples: Samples from the function (f), assuming the axis 0 is the sample dimension
        :return: Negative Binomial distribution
        """
        function_samples = self.scale(function_samples)
        if self.invlink is not None:
            m = self.invlink(function_samples)
        else:
            m = function_samples
        m = torch.clamp(m, min=0.0)  # Ensure m is non-negative
        # TODO remove mask kwarg
        return ContinuousNegativeBinomial(self.alpha, m, validate_args=True) 

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

    def expected_log_prob(self, observations, function_dist, *params, **kwargs):
        """
        Computes the expected log likelihood, where the expectation is over the GP variational distribution.
        
        :param observations: Values of y
        :param function_dist: Distribution for f(x) at (T, D), assume it's a MVN distribution
        :return: Expected log likelihood
        """
        # NOTE this is not the same as the Monte Carlo samples built from de_solve(h_0, f, t)
        # NOTE cannot parse **kwargs TypeError: NegativeBinomial.__init__() got an unexpected keyword argument 'mask'
        function_samples = function_dist.rsample(sample_shape=torch.Size([self.num_likelihood_samples]))  # (S, T, D)
        observations = self.scale(observations)
        if not torch.all(observations >= 0):
            # by pass small numerical issues
            # assert torch.all(observations >= -1e-13), "Observations must be non-negative for Negative Binomial likelihood"
            observations = torch.clamp(observations, min=1e-5)
        function_samples = function_samples.squeeze()
        observations = observations.squeeze()
        log_probs = self.forward(function_samples, **kwargs).log_prob(observations)  # (S, T, D)
        assert log_probs.dtype == torch.float64, "Log probabilities must be in float64 for numerical stability"
        return log_probs.mean(dim=0)

    def scale(self, transformed_data):
        """
        Scales the input tensor using the scale factor.
        :input transformed_data: Tensor to be inverse scaled
        :return: Scaled tensor
        """
        if self.scaler is not None:
            shape = transformed_data.shape
            result = torch.tensor(self.scaler.inverse_transform(transformed_data.detach().numpy().reshape(-1, 1)), dtype=torch.float64).view(shape)

            return result
        else:
            return transformed_data

    @property
    def alpha(self):
        """
        Returns the dispersion parameter alpha.
        :return: Dispersion parameter alpha
        """
        return torch.exp(self.raw_alpha)
