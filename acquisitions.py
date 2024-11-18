from typing import Optional, Tuple, Union

from botorch.acquisition.analytic import AnalyticAcquisitionFunction, ExpectedImprovement as EI_botorch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from torch import Tensor, ones


class ExpectedImprovement(EI_botorch):
    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            observation_noise: bool = False,
    ):
        """Initialize CustomExpectedImprovement with observation_noise option.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (to be improved upon).
            sampler: The sampler used to draw base samples.
            objective: The objective to be maximized.
            observation_noise: Indicates whether to include observation noise in the posterior (default: False).
        """
        super().__init__(model=model, best_f=best_f, maximize=maximize, posterior_transform=posterior_transform)
        self.observation_noise = observation_noise

    def _mean_and_sigma(
            self, X: Tensor, compute_sigma: bool = True, min_var: float = 1e-12
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the first and second moments of the model posterior, optionally with observation noise.

        Args:
            X: `batch_shape x q x d`-dim Tensor of model inputs.
            compute_sigma: Boolean indicating whether or not to compute the second
                moment (default: True).
            min_var: The minimum value the variance is clamped to. Should be positive.

        Returns:
            A tuple of tensors containing the first and second moments of the model
            posterior. Removes the last two dimensions if they have size one. Only
            returns a single tensor of means if compute_sigma is True.
        """
        self.to(device=X.device)  # ensures buffers / parameters are on the same device
        # Use self.observation_noise when calling model's posterior
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform, observation_noise=self.observation_noise
        )
        mean = posterior.mean.squeeze(-2).squeeze(-1)  # removing redundant dimensions
        if not compute_sigma:
            return mean, None
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)
        return mean, sigma


class ExpectedImprovementwithLookahead(AnalyticAcquisitionFunction):
    """
    This is the acquisition function EI(x), where alpha is a decay
    factor that reduces or increases the emphasis of the cost model c(x).
    """

    def __init__(self, model, best_f, xc_dims, d, xuc_samples):
        """

        :param model: GP model
        :param best_f: best objective value observed so far (to be improved upon)
        :param xc_dims: costly dimensions
        :param d: total number of dimensions
        :param xuc_samples: 2D tensor of nxuc x d'
        """
        super().__init__(model=model)
        self.model = model
        self.ei = ExpectedImprovement(model=model, best_f=best_f)
        self.xc_dims = xc_dims
        self.dim = d
        self.xuc_samples = xuc_samples

        # get xuc mask from xc_dims
        self.xuc_mask = ones(self.dim, dtype=bool)
        self.xuc_mask[self.xc_dims] = False

    def forward(self, X):
        # repeat X across the xuc_dimensions on the 0 axis
        Xuc = X.unsqueeze(0).repeat_interleave(self.xuc_samples.shape[0], dim=0)  # shape: nxuc x n x 1 x d

        # Correct the reshaping of self.xuc_samples for broadcasting
        # Assuming self.xuc_samples has shape [1000, 1], target is [1000, 10000, 1]
        xuc_samples_expanded = self.xuc_samples.unsqueeze(1)  # Prepares for broadcasting by adding the right dimension

        # Now, ensure we're targeting the correct shape without adding unnecessary dimensions
        Xuc[:, :, 0, self.xuc_mask] = xuc_samples_expanded

        # return the expected improvement of X and the expected improvement of Xuc
        ei_acq = self.ei(X)
        eiuc_acq = self.ei(Xuc).mean(dim=0)
        return ei_acq + eiuc_acq


