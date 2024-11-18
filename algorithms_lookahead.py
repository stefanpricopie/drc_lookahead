from functools import partial

import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction, qExpectedImprovement, qMultiStepLookahead
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch import ExactMarginalLogLikelihood

from acquisitions import ExpectedImprovement, ExpectedImprovementwithLookahead
from algorithms import generate_initial_data
from gen import gen_candidates_scipy
from initializers import gen_batch_initial_conditions
from multi_step_lookahead import CustomMultiStepLookahead, make_best_f

NOISE_SE = 1e-03
dtype = torch.double    # use double precision for numerical stability


def bo_mean(obj, cost_model, seed, budget, exponent=None, noise_se=NOISE_SE, acq_analysis=False):
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
    x_acq_cheap = [None] * n_init
    x_acq_expensive = [None] * n_init
    X_acq_final = [None] * n_init
    xuc_samples_list = [None] * n_init

    cost_budget = budget * obj.dim * (cost_model.switching_cost + 1)
    while X_cost.sum() < cost_budget:
        # Train GP - Matérn 5/2 kernels with Automatic Relevance Discovery (ARD)
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                          train_Yvar=torch.full_like(train_Y, noise_se**2),
                          input_transform=Normalize(d=train_X.shape[-1]),
                          outcome_transform=Standardize(m=1))

        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll)

        # Get xuc samples with Sobol
        xuc_bounds = obj.bounds[:, ~torch.isin(torch.arange(obj.dim), cost_model.xc_dims)]
        xuc_samples = draw_sobol_samples(bounds=xuc_bounds, n=20, q=1).squeeze(-2)

        # Expected Improvement acquisition function
        eilu = ExpectedImprovementwithLookahead(gp, best_f=train_Y.max().item(),
                                                xc_dims=cost_model.xc_dims, d=obj.dim, xuc_samples=xuc_samples)


        # Get the best point on the entire input space
        full_x, full_x_acq = optimize_acqf(   # returns a 2D tensor of shape (q, d)
            acq_function=eilu,
            bounds=obj.bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        # Fixed Feature acquisition function
        ff_eilu = FixedFeatureAcquisitionFunction(
            acq_function=eilu,
            d=obj.dim,
            columns=cost_model.xc_dims,    # indices of fixed features
            values=train_X[-1, cost_model.xc_dims],
        )

        ff_x, ff_x_acq = optimize_acqf(   # returns a 2D tensor of shape (q, d)
            acq_function=ff_eilu,
            bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), cost_model.xc_dims)],    # remove bounds of fixed features
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        X_acq_cheap.append(ff_x_acq)
        X_acq_expensive.append(full_x_acq)
        xuc_samples_list.append(xuc_samples)

        ei = ExpectedImprovement(gp, best_f=train_Y.max().item())

        x_acq_cheap.append(ei(ff_eilu._construct_X_full(ff_x)))
        x_acq_expensive.append(ei(full_x))

        if exponent == 'heatup':
            alpha = 1 - (cost_budget - X_cost.sum()) / cost_budget
        elif exponent == 'cooldown':
            alpha = (cost_budget - X_cost.sum()) / cost_budget
        elif exponent is None:
            alpha = 1
        else:
            raise ValueError(f"Unknown exponent: {exponent}")

        if ff_x_acq > full_x_acq/(cost_model.switching_cost + 1)**alpha:
            new_x = ff_eilu._construct_X_full(ff_x)
            X_acq_final.append(ff_x_acq.item())
        else:
            new_x = full_x
            X_acq_final.append((full_x_acq/(cost_model.switching_cost + 1)**alpha).item())

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
        return train_X, train_Y, exact_Y,  {
            'X_acq_cheap': X_acq_cheap,
            'X_acq_expensive': X_acq_expensive,
            'x_acq_cheap': x_acq_cheap,
            'x_acq_expensive': x_acq_expensive,
            'X_acq_final': X_acq_final,
            'xuc_samples_list': xuc_samples_list,
        }
    else:
        return train_X, train_Y, exact_Y


def bo_batch(obj, cost_model, seed, budget, exponent=None, noise_se=NOISE_SE, acq_analysis=False):
    """
    Bayesian optimization with Expected Improvement acquisition function

    :param obj: objective function
    :param cost_model: cost model
    :param seed: seed for random number generator
    :param budget: Budget of the optimization
    :param exponent: Exponent for the switching cost
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

    X_acq_cheap = [None] * n_init
    X_acq_expensive = [None] * n_init
    x_acq_cheap = [None] * n_init
    x_acq_expensive = [None] * n_init
    X_acq_final = [None] * n_init
    Xbatch_list = [None] * n_init

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
        qei = qExpectedImprovement(gp, best_f=train_Y.max().item())

        # Get the best point on the entire input space
        full_x, full_x_acq = optimize_acqf(
            acq_function=qei,
            bounds=obj.bounds,
            q=2,
            num_restarts=10,
            raw_samples=512,
            ic_generator=partial(gen_batch_initial_conditions,
                                 xc_dims=cost_model.xc_dims),
            gen_candidates=partial(gen_candidates_scipy,
                                   xc_dims=cost_model.xc_dims),
        )

        # Fixed Feature acquisition function
        ff_qei = FixedFeatureAcquisitionFunction(
            acq_function=qei,
            d=obj.dim,
            columns=cost_model.xc_dims,    # indices of fixed features
            values=train_X[-1, cost_model.xc_dims],
        )

        ff_x, ff_x_acq = optimize_acqf(   # returns a 2D tensor of shape (q, d)
            acq_function=ff_qei,
            bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), cost_model.xc_dims)],    # remove bounds of fixed features
            q=2,
            num_restarts=10,
            raw_samples=512,
        )

        X_acq_cheap.append(ff_x_acq)
        X_acq_expensive.append(full_x_acq)

        ei = ExpectedImprovement(gp, best_f=train_Y.max().item())

        x_acq_cheap.append(ei(ff_qei._construct_X_full(ff_x).unsqueeze(-2)).max(0)[0])
        x_acq_expensive.append(ei(full_x.unsqueeze(-2)).max(0)[0])

        if exponent == 'heatup':
            alpha = 1 - (cost_budget - X_cost.sum()) / cost_budget
        elif exponent == 'cooldown':
            alpha = (cost_budget - X_cost.sum()) / cost_budget
        elif exponent is None:
            alpha = 1
        else:
            raise ValueError(f"Unknown exponent: {exponent}")

        if ff_x_acq > full_x_acq/(cost_model.switching_cost + 1)**alpha:
            new_x = ff_qei._construct_X_full(ff_x)
            X_acq_final.append(ff_x_acq.item())
        else:
            new_x = full_x
            X_acq_final.append((full_x_acq/(cost_model.switching_cost + 1)**alpha).item())

        Xbatch_list.append(new_x)

        # select from batch the best point
        ei = ExpectedImprovement(gp, best_f=train_Y.max().item())
        ei_value = ei(new_x.unsqueeze(-2))
        _, max_index = ei_value.max(0)
        new_x = new_x[max_index].unsqueeze(0)

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
        return train_X, train_Y, exact_Y,  {
            'X_acq_cheap': X_acq_cheap,
            'X_acq_expensive': X_acq_expensive,
            'x_acq_cheap': x_acq_cheap,
            'x_acq_expensive': x_acq_expensive,
            'X_acq_final': X_acq_final,
            'Xbatch_list': Xbatch_list,
        }
    else:
        return train_X, train_Y, exact_Y


def bo_lookahead(obj, cost_model, seed, budget,
                 num_fantasies_c, num_fantasies_uc, exponent=None,
                 noise_se=NOISE_SE, acq_analysis=False):
    """
    Bayesian optimization with Expected Improvement acquisition function

    :param obj: objective function
    :param cost_model: cost model
    :param seed: seed for random number generator
    :param budget: Budget of the optimization
    :param exponent: Exponent for the switching cost
    :param num_fantasies_c: number of fantasies for the context
    :param num_fantasies_uc: number of fantasies for the uncertain part
    :param noise_se: noise standard deviation
    :param acq_analysis: Boolean flag to return acquisition values
    :return:
    """
    assert len(num_fantasies_c) == len(num_fantasies_uc)
    k = len(num_fantasies_c)

    # Set random seed
    torch.manual_seed(seed)
    n_init = 2 * obj.dim

    # Generate initial data with initial obj.dim points
    train_X, train_Y, exact_Y = generate_initial_data(obj, n=n_init)

    X_cost = torch.tensor(0, dtype=dtype)   # Ignore cost of initial points

    X_acq_cheap = [None] * n_init
    X_acq_expensive = [None] * n_init
    x_acq_cheap = [None] * n_init
    x_acq_expensive = [None] * n_init
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

        batch_sizes = [1] * k
        valfunc_cls = [ExpectedImprovement] * (k + 1)
        valfunc_argfacs = [make_best_f] * (k + 1)
        if exponent == 'heatup':
            alpha = 1 - (cost_budget - X_cost.sum()) / cost_budget
        elif exponent == 'cooldown':
            alpha = (cost_budget - X_cost.sum()) / cost_budget
        elif exponent is None:
            alpha = 1
        else:
            raise ValueError(f"Unknown exponent: {exponent}")

        # Initialize the qMultiStepLookahead acquisition function
        qMS = CustomMultiStepLookahead(
            model=gp,
            batch_sizes=batch_sizes,  # TODO: fix code for q > 1
            num_fantasies_c=num_fantasies_c,  # Number of fantasies for each step
            num_fantasies_uc=num_fantasies_uc,  # Number of fantasies for each step
            valfunc_cls=valfunc_cls,  # Acquisition function class
            valfunc_argfacs=valfunc_argfacs,  # Argument factory
            switch_cost=(cost_model.switching_cost + 1),  # Switch cost
            alpha_exp=alpha,  # Switch cost exponent
            xc_dims=cost_model.xc_dims,  # Dimensions to consider for the context
        )

        # Get the best point on the entire input space
        full_x, full_x_acq = optimize_acqf(
            acq_function=qMS,
            bounds=obj.bounds,
            q=qMS.get_augmented_q_batch_size(1),
            num_restarts=10,
            raw_samples=512,
            ic_generator=partial(gen_batch_initial_conditions,
                                 xc_dims=cost_model.xc_dims),
            gen_candidates=partial(gen_candidates_scipy,
                                   xc_dims=cost_model.xc_dims),
            return_full_tree=True,
        )

        # Fixed Feature acquisition function
        qMS_FF = CustomMultiStepLookahead(
            model=gp,
            batch_sizes=batch_sizes,  # TODO: fix code for q > 1
            num_fantasies_c=num_fantasies_c,  # Number of fantasies for each step
            num_fantasies_uc=num_fantasies_uc,  # Number of fantasies for each step
            valfunc_cls=valfunc_cls,  # Acquisition function class
            valfunc_argfacs=valfunc_argfacs,  # Argument factory
            switch_cost=(cost_model.switching_cost + 1)**alpha,  # Switch cost with exponent
            alpha_exp=alpha,  # Switch cost exponent
            xc_dims=cost_model.xc_dims,  # Dimensions to consider for the context
            constrained=True,   # Constrained optimization
        )

        # Get the best point on the entire input space
        ff_x, ff_x_acq = optimize_acqf(
            acq_function=qMS_FF,
            bounds=obj.bounds,
            q=qMS_FF.get_augmented_q_batch_size(1),
            num_restarts=10,
            raw_samples=512,
            ic_generator=partial(gen_batch_initial_conditions,
                                 xc_dims=cost_model.xc_dims),
            gen_candidates=partial(gen_candidates_scipy,
                                   xc_dims=cost_model.xc_dims),
            return_full_tree=True,
        )

        X_acq_cheap.append(ff_x_acq)
        X_acq_expensive.append(full_x_acq)

        ei = ExpectedImprovement(gp, best_f=train_Y.max().item())

        # TODO: generalize this. [0] only works if batch at step 0 is 1
        x_acq_cheap.append(ei(ff_x[0].unsqueeze(-2)))
        x_acq_expensive.append(ei(full_x[0].unsqueeze(-2)))

        # No need to discount by cost as qMS already does that
        if ff_x_acq > full_x_acq:
            new_x = ff_x
            X_acq_final.append(ff_x_acq.item())
        else:
            new_x = full_x
            X_acq_final.append(full_x_acq.item())

        new_x = qMS.extract_candidates(new_x)   # qMS_FF works also

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
        return train_X, train_Y, exact_Y,  {
            'X_acq_cheap': X_acq_cheap,
            'X_acq_expensive': X_acq_expensive,
            'x_acq_cheap': x_acq_cheap,
            'x_acq_expensive': x_acq_expensive,
            'X_acq_final': X_acq_final,
        }
    else:
        return train_X, train_Y, exact_Y


def bo_lookahead_og(obj, cost_model, seed, budget, num_fantasies, noise_se=NOISE_SE, acq_analysis=False):
    """
    Bayesian optimization with Expected Improvement acquisition function

    :param obj: objective function
    :param cost_model: cost model
    :param seed: seed for random number generator
    :param budget: budget of the optimization
    :param num_fantasies: number of fantasies to use in the acquisition function
    :param noise_se: noise standard deviation
    :param acq_analysis: boolean flag to return acquisition values
    :return:
    """
    # Set random seed
    torch.manual_seed(seed)
    n_init = 2 * obj.dim
    k = len(num_fantasies)

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

        batch_sizes = [1] * k
        valfunc_cls = [ExpectedImprovement] * (k + 1)
        valfunc_argfacs = [make_best_f] * (k + 1)

        # Initialize the qMultiStepLookahead acquisition function
        qMS = qMultiStepLookahead(
            model=gp,
            num_fantasies=num_fantasies,
            batch_sizes=batch_sizes,
            valfunc_cls=valfunc_cls,  # Acquisition function class
            valfunc_argfacs=valfunc_argfacs,  # Argument factory
        )

        new_x, acq_x = optimize_acqf(
            acq_function=qMS,
            bounds=obj.bounds,
            q=qMS.get_augmented_q_batch_size(1),
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
