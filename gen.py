from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.generation.gen import (_process_scipy_result,
                                    _convert_nonlinear_inequality_constraints,
                                    make_scipy_nonlinear_inequality_constraints,
                                    _arrayify,
                                    make_scipy_bounds,
                                    make_scipy_linear_constraints,
                                    minimize_with_timeout,
                                    nonlinear_constraint_is_feasible)
from botorch.generation.utils import (
    _remove_fixed_features_from_optimization,
)
from botorch.logging import _get_logger
from botorch.optim.utils import columnwise_clamp
from botorch.optim.utils import fix_features
from torch import Tensor

from multi_step_lookahead import CustomMultiStepLookahead, split_list_by_indices_pytorch

logger = _get_logger()


def gen_candidates_scipy(
        initial_conditions: Tensor,
        acquisition_function: AcquisitionFunction,
        lower_bounds: Optional[Union[float, Tensor]] = None,
        upper_bounds: Optional[Union[float, Tensor]] = None,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        nonlinear_inequality_constraints: Optional[List[Tuple[Callable, bool]]] = None,
        options: Optional[Dict[str, Any]] = None,
        fixed_features: Optional[Dict[int, Optional[float]]] = None,
        timeout_sec: Optional[float] = None,
        xc_dims: Optional[torch.Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates using `scipy.optimize.minimize`.

    Optimizes an acquisition function starting from a set of initial candidates
    using `scipy.optimize.minimize` via a numpy converter.

    TODO: Error handle when xc_dims is None

    Args:
        initial_conditions: Starting points for optimization, with shape
            (b) x q x d.
        acquisition_function: Acquisition function to be used.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
        nonlinear_inequality_constraints: A list of tuples representing the nonlinear
            inequality constraints. The first element in the tuple is a callable
            representing a constraint of the form `callable(x) >= 0`. In case of an
            intra-point constraint, `callable()`takes in an one-dimensional tensor of
            shape `d` and returns a scalar. In case of an inter-point constraint,
            `callable()` takes a two dimensional tensor of shape `q x d` and again
            returns a scalar. The second element is a boolean, indicating if it is an
            intra-point or inter-point constraint (`True` for intra-point. `False` for
            inter-point). For more information on intra-point vs inter-point
            constraints, see the docstring of the `inequality_constraints` argument to
            `optimize_acqf()`. The constraints will later be passed to the scipy
            solver.
        options: Options used to control the optimization including "method"
            and "maxiter". Select method for `scipy.minimize` using the
            "method" key. By default uses L-BFGS-B for box-constrained problems
            and SLSQP if inequality or equality constraints are present. If
            `with_grad=False`, then we use a two-point finite difference estimate
            of the gradient.
        fixed_features: This is a dictionary of feature indices to values, where
            all generated candidates will have features fixed to these values.
            If the dictionary value is None, then that feature will just be
            fixed to the clamped value and not optimized. Assumes values to be
            compatible with lower_bounds and upper_bounds!
        timeout_sec: Timeout (in seconds) for `scipy.optimize.minimize` routine -
            if provided, optimization will stop after this many seconds and return
            the best solution found so far.
        xc_dims: indices of fixed features for the batch initialisation

    Returns:
        2-element tuple containing

        - The set of generated candidates.
        - The acquisition value for each t-batch.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0., 0.], [1., 2.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
        >>> batch_candidates, batch_acq_values = gen_candidates_scipy(
                initial_conditions=Xinit,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
    """
    if xc_dims is None:
        raise ValueError("xc_dims must be provided. Function misbehaves without it.")

    options = options or {}
    options = {**options, "maxiter": options.get("maxiter", 2000)}

    # if there are fixed features we may optimize over a domain of lower dimension
    reduced_domain = False
    if fixed_features:
        # if there are no constraints, things are straightforward
        if not (
                inequality_constraints
                or equality_constraints
                or nonlinear_inequality_constraints
        ):
            reduced_domain = True
        # if there are we need to make sure features are fixed to specific values
        else:
            reduced_domain = None not in fixed_features.values()

    if nonlinear_inequality_constraints:
        if not isinstance(nonlinear_inequality_constraints, list):
            raise ValueError(
                "`nonlinear_inequality_constraints` must be a list of tuples, "
                f"got {type(nonlinear_inequality_constraints)}."
            )
        nonlinear_inequality_constraints = _convert_nonlinear_inequality_constraints(
            nonlinear_inequality_constraints
        )

    if reduced_domain:
        _no_fixed_features = _remove_fixed_features_from_optimization(
            fixed_features=fixed_features,
            acquisition_function=acquisition_function,
            initial_conditions=initial_conditions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        )
        # call the routine with no fixed_features
        clamped_candidates, batch_acquisition = gen_candidates_scipy(
            initial_conditions=_no_fixed_features.initial_conditions,
            acquisition_function=_no_fixed_features.acquisition_function,
            lower_bounds=_no_fixed_features.lower_bounds,
            upper_bounds=_no_fixed_features.upper_bounds,
            inequality_constraints=_no_fixed_features.inequality_constraints,
            equality_constraints=_no_fixed_features.equality_constraints,
            nonlinear_inequality_constraints=_no_fixed_features.nonlinear_inequality_constraints,  # noqa: E501
            options=options,
            fixed_features=None,
            timeout_sec=timeout_sec,
        )
        clamped_candidates = _no_fixed_features.acquisition_function._construct_X_full(
            clamped_candidates
        )
        return clamped_candidates, batch_acquisition
    clamped_candidates = columnwise_clamp(
        X=initial_conditions, lower=lower_bounds, upper=upper_bounds
    )

    shapeX = clamped_candidates.shape
    x0 = clamped_candidates.view(-1)
    bounds = make_scipy_bounds(
        X=initial_conditions, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    constraints = make_scipy_linear_constraints(
        shapeX=shapeX,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )

    with_grad = options.get("with_grad", True)
    if with_grad:

        def f_np_wrapper(x: np.ndarray, f: Callable):
            """Given a torch callable, compute value + grad given a numpy array."""
            if np.isnan(x).any():
                raise RuntimeError(
                    f"{np.isnan(x).sum()} elements of the {x.size} element array "
                    f"`x` are NaN."
                )
            X = (
                torch.from_numpy(x)
                .to(initial_conditions)
                .view(shapeX)
                .contiguous()
                .requires_grad_(True)
            )
            X_fix = fix_features(X, fixed_features=fixed_features)
            loss = f(X_fix).sum()
            # compute gradient w.r.t. the inputs (does not accumulate in leaves)
            grad = torch.autograd.grad(loss, X)[0].contiguous()

            if isinstance(acquisition_function, CustomMultiStepLookahead):
                # Split the gradient into groups and average within each group
                branches = split_list_by_indices_pytorch(
                    acquisition_function.get_binary_tree_input_representation(grad),
                    acquisition_function.dfs_branch_group_indices)
                # len(branches) will always be a power of 2 - therefore so will half
                half = len(branches) // 2

                # Constrained only affects branches[0]
                if acquisition_function.constrained:
                    # Skip branch
                    start = 1
                    # derivatives of this branch are zero
                    for item in branches[0]:
                        item[..., xc_dims] = 0
                else:
                    # Calculate branch gradients like any other branch
                    start = 0

                for group in branches[start:half]:
                    # Reshape each tensor to collapse all but the last dimension and concatenate them
                    # TODO: permute batch size and q dimensions to add over fantasies and q
                    reshaped_grads = [tensor.reshape(-1, *group[0].shape)[..., xc_dims] for tensor in group]
                    # Group sum has shape f_k x f_k-1 x ... x (b) x 1 x d', no q dimension
                    # which should be same as group[0].shape but with the q dimension now 1
                    group_sum = torch.sum(torch.cat(reshaped_grads, dim=0), dim=0)

                    # Change grad in place
                    for item in group:
                        item[..., xc_dims] = group_sum
            else:
                # grad[..., xc_dims] has shape (b) x q x d'. Average on -2 dim across q
                # Replace all gradients on the constrained dimensions with their respective sum across q
                grad[..., xc_dims] = grad[..., xc_dims].sum(dim=-2, keepdim=True)

            gradf = _arrayify(grad.view(-1))
            if np.isnan(gradf).any():
                msg = (
                    f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                    "gradient array `gradf` are NaN. "
                    "This often indicates numerical issues."
                )
                if initial_conditions.dtype != torch.double:
                    msg += " Consider using `dtype=torch.double`."
                raise RuntimeError(msg)
            fval = loss.item()
            return fval, gradf

    else:

        def f_np_wrapper(x: np.ndarray, f: Callable):
            X = torch.from_numpy(x).to(initial_conditions).view(shapeX).contiguous()
            with torch.no_grad():
                X_fix = fix_features(X=X, fixed_features=fixed_features)
                loss = f(X_fix).sum()
            fval = loss.item()
            return fval

    if nonlinear_inequality_constraints:
        # Make sure `batch_limit` is 1 for now.
        if not (len(shapeX) == 3 and shapeX[:2] == torch.Size([1, 1])):
            raise ValueError(
                "`batch_limit` must be 1 when non-linear inequality constraints "
                "are given."
            )
        constraints += make_scipy_nonlinear_inequality_constraints(
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            f_np_wrapper=f_np_wrapper,
            x0=x0,
            shapeX=shapeX,
        )
    x0 = _arrayify(x0)

    def f(x):
        return -acquisition_function(x)

    res = minimize_with_timeout(
        fun=f_np_wrapper,
        args=(f,),
        x0=x0,
        method=options.get("method", "SLSQP" if constraints else "L-BFGS-B"),
        jac=with_grad,
        bounds=bounds,
        constraints=constraints,
        callback=options.get("callback", None),
        options={
            k: v
            for k, v in options.items()
            if k not in ["method", "callback", "with_grad"]
        },
        timeout_sec=timeout_sec,
    )
    _process_scipy_result(res=res, options=options)

    candidates = fix_features(
        X=torch.from_numpy(res.x).to(initial_conditions).reshape(shapeX),
        fixed_features=fixed_features,
    )

    # SLSQP sometimes fails in the line search or may just fail to find a feasible
    # candidate in which case we just return the starting point. This happens rarely,
    # so it shouldn't be an issue given enough restarts.
    if nonlinear_inequality_constraints:
        for con, is_intrapoint in nonlinear_inequality_constraints:
            if not nonlinear_constraint_is_feasible(
                    con, is_intrapoint=is_intrapoint, x=candidates
            ):
                candidates = torch.from_numpy(x0).to(candidates).reshape(shapeX)
                warnings.warn(
                    "SLSQP failed to converge to a solution the satisfies the "
                    "non-linear constraints. Returning the feasible starting point."
                )
                break

    clamped_candidates = columnwise_clamp(
        X=candidates, lower=lower_bounds, upper=upper_bounds, raise_on_violation=True
    )
    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates)

    return clamped_candidates, batch_acquisition

