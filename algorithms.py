import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood

from acquisitions import ExpectedImprovement

NOISE_SE = 1e-03
dtype = torch.double    # use double precision for numerical stability


def random_sample_bounds(bounds, n):
    """
    Generate n random samples from the bounds
    :param bounds: Tensor of shape (d, 2) containing the lower and upper bounds for each dimension
    :param n: Number of samples to generate
    :return: Tensor of shape (n, d) containing the n samples
    """
    x_list = []
    for lb, ub in bounds:
        # Generate a tensor with values uniformly distributed between 0 and 1
        # Scale and shift to match the bounds [lb, ub]
        tensor = torch.rand(n, dtype=dtype) * (ub - lb) + lb
        tensor = tensor.unsqueeze(-1)
        x_list.append(tensor)

    return torch.cat(x_list, dim=-1) if x_list else torch.empty(0, dtype=dtype)


def generate_initial_data(obj, n, xc_dims=None, k=1):
    """
    Generate initial data with multiple evaluations per expensive xc

    :param obj: objective function
    :param n: Number of (initial) points to generate.
    :param k: the number of cheap evaluations per expensive xc. default is 1 for normal BO
    :param xc_dims: Indices of fixed features
    :return: Tensor of dimension (n, d) containing the initial points,
            Tensor of dimension (n, 1) containing the objective values at the initial points,
            Tensor of dimension (n, 1) containing the cost values at the initial points
    """
    # if k is not 1 then xc_dims must be provided and vice versa
    if k != 1 and xc_dims is None:
        raise ValueError("If k is not 1, then xc_dims must be provided.")
    elif k == 1 and xc_dims is not None:
        raise ValueError("If xc_dims is provided, then k must not be 1.")
    
    xc_dims = xc_dims if xc_dims is not None else torch.tensor([], dtype=torch.long)

    # Check divisibility of n by k
    if n % k != 0:
        raise ValueError(f"Number of initial points {n} must be divisible by k = {k}.")

    # Get empty tensor train_x of shape (n, d)
    train_x = torch.empty((n, obj.dim), dtype=dtype)

    # Create a boolean mask for the entire tensor
    mask = torch.zeros(obj.dim, dtype=torch.bool)

    # Update the mask to True for the subset of indices
    mask[xc_dims] = True

    # Get xc_bounds and xuc_bounds
    xc_bounds = obj.bounds.T[mask, :]
    xuc_bounds = obj.bounds.T[~mask, :]

    # Get how many xuc sampled from n
    nxc = n // k

    # Generate xc and repeat along the first dimension for k times
    xc = random_sample_bounds(xc_bounds, nxc)
    xc = xc.repeat_interleave(k, dim=0)

    # Update train_x
    train_x[:, mask] = xc

    # Generate xuc
    xuc = random_sample_bounds(xuc_bounds, n)

    # Update train_x
    train_x[:, ~mask] = xuc

    exact_obj = obj(train_x).unsqueeze(-1)  # add output dimension

    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

    return train_x, train_obj, exact_obj


def bo_unconstrained(obj, cost_model, seed, budget, noise_se=NOISE_SE, acq_analysis=False):
    """
    Bayesian optimization with Expected Improvement acquisition function

    :param obj: objective function
    :param cost_model: cost model
    :param seed: seed for random number generator
    :param budget: budget of the optimization
    :param noise_se: noise standard deviation
    :param acq_analysis: boolean flag to return acquisition values
    :return:
    """
    # Set random seed
    torch.manual_seed(seed)
    n_init = 2 * obj.dim

    # Generate initial data with initial obj.dim points
    train_X, train_Y, exact_Y = generate_initial_data(obj, n=n_init)

    X_cost = torch.tensor(0, dtype=dtype)   # Ignore cost of initial points

    X_acq = [None] * n_init

    cost_budget = budget * obj.dim * (cost_model.switching_cost + 1)
    while X_cost.sum() < cost_budget:
        # Train GP - Matérn 5/2 kernels with Automatic Relevance Discovery (ARD)
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                          train_Yvar=torch.full_like(train_Y, noise_se**2),
                          input_transform=Normalize(d=train_X.shape[-1]),
                          outcome_transform=Standardize(m=1))

        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll)

        new_x, acq_x = optimize_acqf(
            acq_function=ExpectedImprovement(gp, best_f=train_Y.max().item()),
            bounds=obj.bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        # Get objective value and cost
        exact_y = obj(new_x).unsqueeze(-1)  # add output dimension
        train_y = exact_y + noise_se * torch.randn_like(exact_y)

        # Update training points
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, torch.tensor([train_y], dtype=train_Y.dtype, device=train_Y.device).unsqueeze(1)])
        exact_Y = torch.cat([exact_Y, torch.tensor([exact_y], dtype=exact_Y.dtype, device=exact_Y.device).unsqueeze(1)])

        X_acq.append(acq_x.item())

        # Update cost. n_init-1 because we need the last point to calculate
        # the cost but calculate cost from n_init onwards
        X_cost = cost_model(train_X[n_init-1:])[1:]   # Ignore cost of initial points

    if acq_analysis:
        return train_X, train_Y, exact_Y, {'X_acq': X_acq}
    else:
        return train_X, train_Y, exact_Y


def bo_randomffc(obj, p, cost_model, seed, budget, noise_se=NOISE_SE):
    """
    Bayesian optimization with Expected Improvement acquisition function

    :param obj: objective function
    :param p: probability of using Fixed Feature acquisition function
    :param cost_model: cost model
    :param seed: seed for random number generator
    :param budget: budget of the optimization
    :param noise_se: noise standard deviation
    :return:
    """
    xc_dims = cost_model.xc_dims

    # Set random seed
    torch.manual_seed(seed)
    n_init = 2 * obj.dim

    # Generate initial data with initial obj.dim points
    train_X, train_Y, exact_Y = generate_initial_data(obj, n=n_init)

    X_cost = torch.tensor(0, dtype=dtype)   # Ignore cost of initial points

    cost_budget = budget * obj.dim * (cost_model.switching_cost + 1)
    while X_cost.sum() < cost_budget:
        # Train GP - Matérn 5/2 kernels with Automatic Relevance Discovery (ARD)
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                          train_Yvar=torch.full_like(train_Y, noise_se**2),
                          input_transform=Normalize(d=train_X.shape[-1]),
                          outcome_transform=Standardize(m=1))

        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll)

        acq_function = ExpectedImprovement(gp, best_f=train_Y.max().item())

        # get random boolean with probability pff to be True
        if torch.rand(1) < p:
            ff_acq = FixedFeatureAcquisitionFunction(
                acq_function=acq_function,
                d=obj.dim,
                columns=xc_dims,    # indices of fixed features
                values=train_X[-1, xc_dims],
            )

            new_xuc, _ = optimize_acqf(   # returns a 2D tensor of shape (q, d)
                acq_function=ff_acq,
                bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), xc_dims)],    # remove bounds of fixed features
                q=1,
                num_restarts=10,
                raw_samples=512,
            )

            # Construct full input tensor
            new_x = ff_acq._construct_X_full(new_xuc)
        else:
            new_x, _ = optimize_acqf(
                acq_function=acq_function,
                bounds=obj.bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )

        # Get objective value and cost
        exact_y = obj(new_x).unsqueeze(-1)  # add output dimension
        train_y = exact_y + noise_se * torch.randn_like(exact_y)

        # Update training points
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, torch.tensor([train_y], dtype=train_Y.dtype, device=train_Y.device).unsqueeze(1)])
        exact_Y = torch.cat([exact_Y, torch.tensor([exact_y], dtype=exact_Y.dtype, device=exact_Y.device).unsqueeze(1)])

        # Update cost. n_init-1 because we need the last point to calculate
        # the cost but calculate cost from n_init onwards
        X_cost = cost_model(train_X[n_init-1:])[1:]   # Ignore cost of initial points

    return train_X, train_Y, exact_Y


def eipu(obj, cost_model, seed, budget, exponent=None, noise_se=NOISE_SE, acq_analysis=False):
    """
    Expected Improvement with Cost (EIpu) algorithm.
    :param obj: Objective function
    :param cost_model: Cost model
    :param seed: Seed for random number generator
    :param budget: Budget of the optimization
    :param exponent: Exponent for the switching cost
    :param noise_se: Noise standard deviation
    :param acq_analysis: Boolean flag to return acquisition values
    :return:
    """
    # Set random seed
    torch.manual_seed(seed)
    n_init = 2 * obj.dim

    # Generate initial data with initial obj.dim points
    train_X, train_Y, exact_Y = generate_initial_data(obj, n=n_init)

    X_cost = torch.tensor(0, dtype=dtype)   # Ignore cost of initial points

    X_acq_cheap = [None] * n_init
    X_acq_expensive = [None] * n_init
    X_acq_final = [None] * n_init

    cost_budget = budget * obj.dim * (cost_model.switching_cost + 1)
    while X_cost.sum() < cost_budget:
        # Train GP - Matérn 5/2 kernels with Automatic Relevance Discovery (ARD)
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                          train_Yvar=torch.full_like(train_Y, noise_se**2),
                          input_transform=Normalize(d=train_X.shape[-1]),
                          outcome_transform=Standardize(m=1))

        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll)

        # Expected Improvement acquisition function
        ei = ExpectedImprovement(gp, best_f=train_Y.max().item())

        # Get the best point on the entire input space
        full_x, full_x_acq = optimize_acqf(   # returns a 2D tensor of shape (q, d)
            acq_function=ei,
            bounds=obj.bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        # Fixed Feature acquisition function
        ff_ei = FixedFeatureAcquisitionFunction(
            acq_function=ei,
            d=obj.dim,
            columns=cost_model.xc_dims,    # indices of fixed features
            values=train_X[-1, cost_model.xc_dims],
        )

        ff_x, ff_x_acq = optimize_acqf(   # returns a 2D tensor of shape (q, d)
            acq_function=ff_ei,
            bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), cost_model.xc_dims)],    # remove bounds of fixed features
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        X_acq_cheap.append(ff_x_acq)
        X_acq_expensive.append(full_x_acq)

        if exponent == 'heatup':
            alpha = 1 - (cost_budget - X_cost.sum()) / cost_budget
        elif exponent == 'cooldown':
            alpha = (cost_budget - X_cost.sum()) / cost_budget
        elif exponent is None:
            alpha = 1
        else:
            raise ValueError(f"Unknown exponent: {exponent}")

        if ff_x_acq > full_x_acq/(cost_model.switching_cost+1)**alpha:
            new_x = ff_ei._construct_X_full(ff_x)
            X_acq_final.append(ff_x_acq.item())
        else:
            new_x = full_x
            X_acq_final.append((full_x_acq/(cost_model.switching_cost+1)**alpha).item())

        # Get objective value and cost
        exact_y = obj(new_x).unsqueeze(-1)  # add output dimension
        train_y = exact_y + noise_se * torch.randn_like(exact_y)

        # Update training points
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, torch.tensor([train_y], dtype=train_Y.dtype, device=train_Y.device).unsqueeze(1)])
        exact_Y = torch.cat([exact_Y, torch.tensor([exact_y], dtype=exact_Y.dtype, device=exact_Y.device).unsqueeze(1)])

        # Update cost. n_init-1 because we need the last point to calculate
        # the cost but calculate cost from n_init onwards
        X_cost = cost_model(train_X[n_init-1:])[1:]   # Ignore cost of initial points

    if acq_analysis:
        return train_X, train_Y, exact_Y, {
            'X_acq_cheap': X_acq_cheap,
            'X_acq_expensive': X_acq_expensive,
            'X_acq_final': X_acq_final,
        }
    else:
        return train_X, train_Y, exact_Y


