from gpytorch.likelihoods import Likelihood, _OneDimensionalLikelihood
# from gpytorch import base_distributions
import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from torch.distributions import constraints
from torch.nn import Parameter
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform


# class NegativeBinomial(ScalarLikelihood):
#     """
#     alpha: dispersion parameter
#     scale: to adjust the mean of the negative binomial by a scale factor
#     """
#     def __init__(self, alpha=1.0, invlink: Callable[[tf.Tensor], tf.Tensor] =tf.exp, scale: float=1.0, **kwargs: Any,) -> None:
#         super().__init__(**kwargs)
#         self.alpha = Parameter(alpha, transform=positive())
#         self.scale:AnyNDArray= np.array(scale, dtype=default_float())
#         #self.scale = Parameter(scale)
#         self.invlink = invlink
   
#     @inherit_check_shapes
#     def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType)-> tf.Tensor:
#         return negative_binomial(self.invlink(F) * self.scale, Y, self.alpha)
   
#     @inherit_check_shapes
#     def _conditional_mean(self, F: TensorType) -> tf.Tensor:
#         return self.invlink(F) * self.scale
   
#     @inherit_check_shapes
#     def _conditional_variance(self, F : TensorType) -> tf.Tensor:
#         m = self.invlink(F) * self.scale
#         return m + m ** 2 * self.alpha


# def negative_binomial(m: TensorType, Y: TensorType, alpha: TensorType)-> tf.Tensor:
#     """
#     P(Y) = Gamma(k + Y) / (Y! Gamma(k)) * (m / (m+k))^Y * (1 + m/k)^(-k)
#     """
#     k = 1 / alpha
#     return (
#         tf.math.lgamma(k + Y)
#         - tf.math.lgamma(Y + 1)
#         - tf.math.lgamma(k)
#         + Y * tf.math.log(m / (m + k))
#         - k * tf.math.log(1 + m * alpha)
#     )

class NegativeBinomialLikelihood(_OneDimensionalLikelihood):
    """
    Implements the Negative Binomial likelihood used for GP regression.
    
    :param alpha: Dispersion parameter, ie var = mu + mu**2 * alpha, alpha := 1/phi
    :param invlink: Inverse link function (default: torch.exp) to convert samples from the GP to the mean of the likelihood, ie positive
    :param scale: Scale factor to adjust the mean of the negative binomial
    """
    def __init__(self, alpha=1.0, invlink=torch.exp, scale=1.0, learnable_alpha=True, S=20):
        super().__init__()
        # TODO ensure that alpha is positive and is updated for each gene
        # TODO add alpha into parameter_trace
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=learnable_alpha) # dispersion
        self.scale = scale
        self.invlink = invlink
        self.num_likelihood_samples = S # the number of Monte Caro samples to approximate E_{q(f)}[logp p(y|f)]

    def forward(self, function_samples, **kwargs):
        """
        Computes the conditional distribution p(y | f).
        
        :param function_samples: Samples from the function (f), assuming the axis 0 is the sample dimension
        :return: Negative Binomial distribution
        """
        if self.invlink is not None:
            m = self.invlink(function_samples) * self.scale
        else:
            m = function_samples * self.scale

        k = 1 / self.alpha # scale
        
        # if (m < 1e-5).any():
        #     print("m", m[torch.where(m < 1e-5)])
        p = 1 - (k / (k + m)) # shape (S, T)
        p = torch.clamp(p, min=1e-5, max=1 - 1e-5)  # Ensure p is in (0, 1)
        # TODO remove mask kwarg
        return gpytorch.distributions.base_distributions.NegativeBinomial(total_count=k, probs=p)

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
        log_probs = self.forward(function_samples, **kwargs).log_prob(observations)  # (S, T, D)
        return log_probs.mean(dim=0).sum(dim=-1)
