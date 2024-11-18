#!/usr/bin/env python3
import argparse
import os
import pickle
import warnings

import torch
from botorch.exceptions import BadInitialCandidatesWarning, OptimizationWarning
from botorch.test_functions import (Ackley, Bukin, DropWave, EggHolder, Michalewicz, Rastrigin, Shekel)
from gpytorch.utils.warnings import NumericalWarning

from algorithms import bo_unconstrained, eipu
from algorithms_lookahead import bo_mean, bo_batch, bo_lookahead, bo_lookahead_og
from cost import SetupCostModel
from test_functions import Shubert

# ignore initial candidates warnings - BadInitialCandidatesWarning
# ignore small noise warnings - NumericalWarning
# ignore Optimization Warnings from optimize_acqf - OptimizationWarning
# ignore Optimization failed in `gen_candidates_scipy` - RuntimeWarning
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ALGORITHMS = {
    "bo": bo_unconstrained,
    "eipu": eipu,
    "mean": bo_mean,
    "bobatch": bo_batch,
    "bo_lookahead": bo_lookahead,
    "bo_look_og": bo_lookahead_og,
}

OBJECTIVES = {
    'ackley': Ackley,
    'bukin': Bukin,
    'dropwave': DropWave,
    'eggholder': EggHolder,
    'michalewicz': Michalewicz,
    'rastrigin': Rastrigin,
    'shekel': Shekel,
    'shubert': Shubert,
    # Add other objectives here
}

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, choices=OBJECTIVES.keys(), required=True, help='Problem to solve')
parser.add_argument('--dim', type=int, default=2, help='Dimensionality of the problem')
parser.add_argument('--algo', type=str, choices=ALGORITHMS.keys(), required=True, help='Algorithm to use')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--switching_cost', type=int, default=5, help='Setup cost')
parser.add_argument('--exponent', type=str, choices=['none', 'cooldown', 'heatup'], default='none',
                    help='Exponent for the switching cost')
parser.add_argument('--xcs', type=int, default=1, help='Iterable of dimension indices Xc')
parser.add_argument('--budget', type=int, default=10, help='Budget of the optimization')
parser.add_argument('--outdir', type=str, help='Output directory')
parser.add_argument('--lookahead_fantasies', type=int, nargs='*', default=[5],
                    help='Lookahead fantasies for the lookahead algorithm')
args = parser.parse_args()


def main(problem, dim, algo, algo_kwargs, seed, switching_cost, xcs, budget, outdir):
    """
    :param problem: Name of the problem to solve
    :param dim: Dimensionality of the problem
    :param algo: Algorithm to use
    :param algo_kwargs: Additional algorithm arguments
    :param seed: Random seed
    :param switching_cost: Setup cost
    :param xcs: Number of expensive dimensions in the problem 1 <= xcs < dim
    :param budget: Budget of the optimization
    :param outdir: Output directory
    :return:
    """

    # Create the objective function instance
    obj_class = OBJECTIVES[problem]
    if 'dim' in obj_class.__init__.__code__.co_varnames:
        obj = OBJECTIVES[problem](dim=dim, negate=True)
    elif 'm' in obj_class.__init__.__code__.co_varnames:
        obj = OBJECTIVES[problem](m=dim, negate=True)
    else:
        obj = OBJECTIVES[problem](negate=True)
        # check if the objective function matches the dimension
        if obj.dim != dim:
            raise ValueError(f"Dimension mismatch: {obj.dim} != {dim}")

    # Define the cost model
    xc_rng = torch.Generator()
    xc_rng.manual_seed(seed)

    XC_DIMS = torch.randperm(obj.dim, generator=xc_rng)[:xcs].sort().values
    cost_model = SetupCostModel(switching_cost=switching_cost, xc_dims=XC_DIMS)

    if algo == "bo_lookahead":
        num_fantasies_c = algo_kwargs['num_fantasies_c']
        num_fantasies_uc = algo_kwargs['num_fantasies_uc']
        num_fantasies = [item_c + item_uc for item_c, item_uc in zip(num_fantasies_c, num_fantasies_uc)]
    elif algo == "bo_look_og":
        num_fantasies = algo_kwargs['num_fantasies']
    else:
        num_fantasies = []

    num_fantasies_str = ','.join(map(str, num_fantasies))
    exponent = algo_kwargs.get('exponent')

    # Construct the file name using the determined suffix
    file_name = (f"{outdir}/{problem}{dim}d_{algo}{num_fantasies_str}{exponent}_sc{switching_cost}_xc{xcs}_r{seed}_"
                 f"{','.join(XC_DIMS.numpy().astype(str))}_B{budget}.pkl").lower()
    print(file_name)

    # if file exists, skip the experiment
    if os.path.exists(file_name):
        print(f"File already exists... Skip experiment")
        return

    # Run the experiment
    X, Y_noisy, Y_exact, X_acq = ALGORITHMS[algo](obj, cost_model, seed=seed, budget=budget, **algo_kwargs)
    C = cost_model(X)

    results = { # Save results to a dictionary
        'X': X,
        'Y_noisy': Y_noisy,
        'Y_exact': Y_exact,
        'X_acq': X_acq,
        'C': C,
        'obj': obj,
        'problem': problem,
        'dim': obj.dim,
        'switching_cost': switching_cost,
        'exponent': exponent,
        'xcs': xcs,
        'xc_dims': XC_DIMS,
        'algo': algo,
        'algo_kwargs': f"{algo}{num_fantasies_str}",
        'seed': seed,
        'budget': budget,
    }

    with open(file_name, 'wb') as f:
        pickle.dump(results, f)


# Main execution
if __name__ == "__main__":
    # Check if p is greater than 1 for pBO and pBOnested algorithms
    if args.algo.startswith('ffc') and args.k <= 1:
        raise ValueError("k must be greater than 1 for pBO and pBOPE algorithms.")

    if not 1 <= args.xcs < args.dim:
        raise ValueError(f"Invalid xcs: {args.xcs}. Must be in [1, {args.dim})")

    # Prepare additional kwargs
    kwargs = {}
    if args.algo == "bo_lookahead":
        # fantasies are the same for both constrained and unconstrained
        kwargs['num_fantasies_c'] = args.lookahead_fantasies
        kwargs['num_fantasies_uc'] = args.lookahead_fantasies
        kwargs['exponent'] = args.exponent
    elif args.algo == "bo_look_og":
        kwargs['num_fantasies'] = args.lookahead_fantasies
    elif args.algo != "bo":
        kwargs['exponent'] = args.exponent

    main(problem=args.problem, dim=args.dim, algo=args.algo, algo_kwargs=kwargs, seed=args.seed,
         switching_cost=args.switching_cost,
         xcs=args.xcs, budget=args.budget, outdir=args.outdir)
