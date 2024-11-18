import math

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor


class GoldsteinPrice(SyntheticTestFunction):
    r"""Goldstein-Price test function.

    Two-dimensional function (usually evaluated on `[-2, 2]^2`):

        f(x, y) = [1 + (x + y + 1)^2 * (19 - 14x + 3x^2 - 14y + 6xy + 3y^2)] *
                  [30 + (2x - 3y)^2 * (18 - 32x + 12x^2 + 48y - 36xy + 27y^2)]
    """

    dim = 2
    _bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    _optimal_value = 3.0  # Minimum value of the function
    _optimizers = [(0.0, -1.0)]  # Location of the minimum
    _check_grad_at_opt: bool = False

    def evaluate_true(self, X: Tensor) -> Tensor:
        x, y = X[..., 0], X[..., 1]
        term1 = (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
        term2 = (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
        return (1 + term1) * (30 + term2)


class Salomon(SyntheticTestFunction):
    r"""Salomon synthetic test function.

    The function is defined as:
        f(\mathbf{x}) = 1 - \cos(2\pi\sqrt{\sum_{i=1}^{d} x_i^2}) + 0.1\sqrt{\sum_{i=1}^{d} x_i^2}

    where d is the dimensionality of the input vector \mathbf{x}. The function is usually
    evaluated on the domain `[-100, 100]^d`.

    The global minimum is at \mathbf{x^*} = \mathbf{0} with f(\mathbf{x^*}) = 0.
    """

    def __init__(
        self,
        dim=2,
        noise_std = None,
        negate = False,
        bounds = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        if bounds is None:
            bounds = [(-100.0, 100.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]  # Location of the global minimum
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        r_sum = torch.sqrt(torch.sum(X ** 2, dim=-1))
        return 1 - torch.cos(2 * math.pi * r_sum) + 0.1 * r_sum


class Schwefel(SyntheticTestFunction):
    r"""Schwefel synthetic test function.

    d-dimensional function (usually evaluated on `[-500, 500]^d`):

        f(x) = 418.9829 * d - sum(x_i * sin(sqrt(abs(x_i)))), for all i = 1, ..., d
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim=2,
        noise_std = None,
        negate = False,
        bounds = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        if bounds is None:
            bounds = [(-500.0, 500.0) for _ in range(self.dim)]
        # The global minimum location approximation for Schwefel's function
        # is around 420.9687 for each dimension, but setting exact optimizers
        # for high dimensions might not be straightforward.
        self._optimizers = [tuple(420.9687 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        term = X * torch.sin(torch.sqrt(torch.abs(X)))
        return 418.9829 * self.dim - torch.sum(term, dim=-1)


class Shubert(SyntheticTestFunction):
    r"""Shubert test function.

    Two-dimensional function (usually evaluated on `[-10, 10]^2`):

        f(x, y) = (sum_{i=1}^{5} i * cos[(i+1)*x + i]) *
                  (sum_{i=1}^{5} i * cos[(i+1)*y + i])
    """

    dim = 2
    _bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    # Note: The Shubert function has multiple global minima with the value approximately -186.7309
    _optimal_value = -186.7309
    # Note: There are 18 global minima, listing them all is beyond the scope of this template
    # _optimizers should ideally list all global minima locations, but we omit it for brevity
    _optimizers = [(-7.08349786, 4.85801859),
                   (4.85801859, -7.08349786),]
    _check_grad_at_opt: bool = False

    def evaluate_true(self, X: Tensor) -> Tensor:
        x, y = X[..., 0], X[..., 1]

        # Create a tensor for the range of i values
        i = torch.arange(1, 6, dtype=X.dtype, device=X.device).unsqueeze(-1)

        # Vectorized sum for x and y
        sum_x = torch.sum(i * torch.cos((i + 1) * x + i), dim=0)
        sum_y = torch.sum(i * torch.cos((i + 1) * y + i), dim=0)

        return sum_x * sum_y
