from __future__ import annotations

import warnings
from typing import Optional, Dict, Union, List, Tuple, Callable

import torch
from botorch import settings, manual_seed
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions import UnsupportedError, SamplingWarning, BotorchTensorDimensionError, \
    BadInitialCandidatesWarning
from botorch.optim import initialize_q_batch_nonneg, initialize_q_batch
from botorch.optim.initializers import is_nonnegative, sample_q_batches_from_polytope, sample_points_around_best
from botorch.utils import draw_sobol_samples
from torch import Tensor
from torch.quasirandom import SobolEngine

from multi_step_lookahead import CustomMultiStepLookahead, split_list_by_indices_pytorch


def gen_batch_initial_conditions(
        acq_function: AcquisitionFunction,
        bounds: Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int,
        fixed_features: Optional[Dict[int, float]] = None,
        options: Optional[Dict[str, Union[bool, float, int]]] = None,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        generator: Optional[Callable[[int, int, int], Tensor]] = None,
        fixed_X_fantasies: Optional[Tensor] = None,
        xc_dims: Optional[torch.Tensor] = None,
) -> Tensor:
    r"""Generate a batch of initial conditions for random-restart optimization.

    TODO: Error handle when xc_dims is None

    Args:
        acq_function: The acquisition function to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic. Note: if `sample_around_best` is True (the default is False),
            then `2 * raw_samples` samples are used.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for initial condition generation. For valid options see
            `initialize_q_batch` and `initialize_q_batch_nonneg`. If `options`
            contains a `nonnegative=True` entry, then `acq_function` is
            assumed to be non-negative (useful when using custom acquisition
            functions). In addition, an "init_batch_limit" option can be passed
            to specify the batch limit for the initialization. This is useful
            for avoiding memory limits when computing the batch posterior over
            raw samples.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
        generator: Callable for generating samples that are then further
            processed. It receives `n`, `q` and `seed` as arguments and
            returns a tensor of shape `n x q x d`.
        fixed_X_fantasies: A fixed set of fantasy points to concatenate to
            the `q` candidates being initialized along the `-2` dimension. The
            shape should be `num_pseudo_points x d`. E.g., this should be
            `num_fantasies x d` for KG and `num_fantasies*num_pareto x d`
            for HVKG.
        xc_dims: indices of fixed features for the batch initialisation

    Returns:
        A `num_restarts x q x d` tensor of initial conditions.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
    """
    if xc_dims is None:
        raise ValueError("xc_dims must be provided. Function misbehaves without it.")

    if bounds.isinf().any():
        raise NotImplementedError(
            "Currently only finite values in `bounds` are supported "
            "for generating initial conditions for optimization."
        )
    options = options or {}
    sample_around_best = options.get("sample_around_best", False)
    if sample_around_best and equality_constraints:
        raise UnsupportedError(
            "Option 'sample_around_best' is not supported when equality"
            "constraints are present."
        )
    if sample_around_best and generator:
        raise UnsupportedError(
            "Option 'sample_around_best' is not supported when custom "
            "generator is be used."
        )
    seed: Optional[int] = options.get("seed")
    batch_limit: Optional[int] = options.get(
        "init_batch_limit", options.get("batch_limit")
    )
    factor, max_factor = 1, 5
    init_kwargs = {}
    device = bounds.device
    bounds_cpu = bounds.cpu()
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = initialize_q_batch

    q = 1 if q is None else q
    # the dimension the samples are drawn from
    effective_dim = bounds.shape[-1] * q
    if effective_dim > SobolEngine.MAXDIM and settings.debug.on():
        warnings.warn(
            f"Sample dimension q*d={effective_dim} exceeding Sobol max dimension "
            f"({SobolEngine.MAXDIM}). Using iid samples instead.",
            SamplingWarning,
        )

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            n = raw_samples * factor
            if generator is not None:
                X_rnd = generator(n, q, seed)
            # check if no constraints are provided
            elif not (inequality_constraints or equality_constraints):
                if effective_dim <= SobolEngine.MAXDIM:
                    X_rnd = draw_sobol_samples(bounds=bounds_cpu, n=n, q=q, seed=seed)
                else:
                    with manual_seed(seed):
                        # load on cpu
                        X_rnd_nlzd = torch.rand(
                            n, q, bounds_cpu.shape[-1], dtype=bounds.dtype
                        )
                    X_rnd = bounds_cpu[0] + (bounds_cpu[1] - bounds_cpu[0]) * X_rnd_nlzd
            else:
                X_rnd = sample_q_batches_from_polytope(
                    n=n,
                    q=q,
                    bounds=bounds,
                    n_burnin=options.get("n_burnin", 10000),
                    thinning=options.get("thinning", 32),
                    seed=seed,
                    equality_constraints=equality_constraints,
                    inequality_constraints=inequality_constraints,
                )
            # sample points around best
            if sample_around_best:
                X_best_rnd = sample_points_around_best(
                    acq_function=acq_function,
                    n_discrete_points=n * q,
                    sigma=options.get("sample_around_best_sigma", 1e-3),
                    bounds=bounds,
                    subset_sigma=options.get("sample_around_best_subset_sigma", 1e-1),
                    prob_perturb=options.get("sample_around_best_prob_perturb"),
                )
                if X_best_rnd is not None:
                    X_rnd = torch.cat(
                        [
                            X_rnd,
                            X_best_rnd.view(n, q, bounds.shape[-1]).cpu(),
                        ],
                        dim=0,
                    )

            # set fixed features for each batch equal to the first value
            if isinstance(acq_function, CustomMultiStepLookahead):
                branches = split_list_by_indices_pytorch(
                    acq_function.get_binary_tree_input_representation(X_rnd),
                    acq_function.dfs_branch_group_indices
                )

                # Constrained only affects branches[0]
                if acq_function.constrained:
                    # Skip branch
                    start = 1
                    # the branch with root in step 0 has all expensive dimensions fixed to the last setup used
                    for item in branches[0]:
                        item[..., xc_dims] = acq_function.model.input_transform.untransform(
                            acq_function.model.train_inputs[0]
                        )[-1, xc_dims]
                else:
                    # Constrain branch like any other branch
                    start = 0

                for group in branches[start:]:
                    for item in group[1:]:
                        # FIXME: Can break for cases where q > 1
                        item[..., xc_dims] = group[0][..., xc_dims]
            else:
                X_rnd[..., xc_dims] = X_rnd[:, 0, xc_dims].unsqueeze(1)

            if fixed_X_fantasies is not None:
                if (d_f := fixed_X_fantasies.shape[-1]) != (d_r := X_rnd.shape[-1]):
                    raise BotorchTensorDimensionError(
                        "`fixed_X_fantasies` and `bounds` must both have the same "
                        f"trailing dimension `d`, but have {d_f} and {d_r}, "
                        "respectively."
                    )
                X_rnd = torch.cat(
                    [
                        X_rnd,
                        fixed_X_fantasies.cpu()
                        .unsqueeze(0)
                        .expand(X_rnd.shape[0], *fixed_X_fantasies.shape),
                    ],
                    dim=-2,
                        )
            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]
                Y_rnd_list = []
                start_idx = 0
                while start_idx < X_rnd.shape[0]:
                    end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                    Y_rnd_curr = acq_function(
                        X_rnd[start_idx:end_idx].to(device=device)
                    ).cpu()
                    Y_rnd_list.append(Y_rnd_curr)
                    start_idx += batch_limit
                Y_rnd = torch.cat(Y_rnd_list)
            batch_initial_conditions = init_func(
                X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs
            ).to(device=device)
            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws):
                return batch_initial_conditions
            if factor < max_factor:
                factor += 1
                if seed is not None:
                    seed += 1  # make sure to sample different X_rnd
    warnings.warn(
        "Unable to find non-zero acquisition function values - initial conditions "
        "are being selected randomly.",
        BadInitialCandidatesWarning,
    )
    return batch_initial_conditions
