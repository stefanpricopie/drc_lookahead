from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Type

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition import qMultiStepLookahead
from botorch.acquisition.multi_step_lookahead import _compute_stage_value
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
from torch import Size, Tensor

TAcqfArgConstructor = Callable[[Model, Tensor], Dict[str, Any]]


def make_best_f(model: Model, X: Tensor) -> Dict[str, Any]:
    r"""Extract the best observed training input from the model.
    TODO: Handle noise properly.
    """
    return {"best_f": model.outcome_transform.untransform(model.train_targets)[0].max(dim=-1).values}


def _dfs_branch_group_indices(k):
    """
    Generate a tensor where each element's value is the index of the depth-first search (DFS)
    group it belongs to, based on a complete binary tree of depth k. This function helps in
    grouping elements that would be together in a DFS traversal.

    Args:
        k (int): The depth of the binary tree.

    Returns:
        torch.Tensor: A tensor of group indices where each element indicates the group index
                      corresponding to its position in a DFS traversal of the binary tree.

    Example:
        For k=3, the binary tree looks like:
            1
           / \
          2   3
         /\   /\
        4  5 6  7
        The expected DFS grouping indices are: [0, 0, 1, 0, 2, 1, 3]
        Representing the groups: [1, 2, 4], [3, 6], [5], [7]
    """
    indices = torch.zeros_like(torch.arange(1, 2**k), dtype=torch.int) - 1
    idx = torch.arange(0, len(indices), 2, dtype=torch.int)
    for _ in range(k):
        indices[idx] = torch.arange(len(idx), dtype=torch.int)  # Ensure dtype matches
        idx = idx[:len(idx)//2] * 2 + 1
    return indices


def split_list_by_indices_pytorch(items, indices):
    """
    Split an iterable into groups based on a tensor of indices. Each index in the 'indices' tensor
    determines the group number for the corresponding item in the 'items' iterable.

    Args:
        items (Iterable[Any]): An iterable where each element is any type.
        indices (torch.Tensor): A tensor of integers where each element specifies the group number
            for the corresponding item in 'items'.

    Returns:
        list of list of Any: A list where each sublist contains items grouped together based on the indices.

    Raises:
        ValueError: If the length of 'indices' does not match the number of items in 'items' or if 'items' is empty.

    Example:
        items = ['a', 'b', 'c']
        indices = torch.tensor([0, 0, 1])
        grouped_items = split_list_by_indices_pytorch(items, indices)
        # Output:
        # Group 0: ['a', 'b']
        # Group 1: ['c']
    """
    if not len(items):
        raise ValueError("The 'items' tuple is empty.")

    # Check if the number of items matches the length of 'indices'
    if len(items) != len(indices):
        raise ValueError("The length of 'items' must match the length of 'indices'.")

    # Determine the number of groups
    num_groups = indices.max().item() + 1

    # Prepare a list to hold the result groups
    groups = [[] for _ in range(num_groups)]

    # Assign each tensor to its corresponding group
    for item, index in zip(items, indices):
        groups[index].append(item)

    return groups


class CustomMultiStepLookahead(qMultiStepLookahead):
    r"""MC-based batch constrained Multi-Step Look-Ahead (one-shot optimization)."""

    def __init__(
            self,
            model: Model,
            batch_sizes: List[int],
            num_fantasies_c: List[int],
            num_fantasies_uc: List[int],
            switch_cost: float,
            xc_dims: Tensor[int],
            alpha_exp: float,
            constrained: Optional[bool] = False,
            samplers: Optional[List[MCSampler]] = None,
            valfunc_cls: Optional[List[Optional[Type[AcquisitionFunction]]]] = None,
            valfunc_argfacs: Optional[List[Optional[TAcqfArgConstructor]]] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            inner_mc_samples: Optional[List[int]] = None,
            X_pending: Optional[Tensor] = None,
            collapse_fantasy_base_samples: bool = True,
    ):
        r"""q-Multi-Step Look-Ahead (one-shot optimization).

        Performs a `k`-step lookahead by means of repeated fantasizing.

        Allows to specify the stage value functions by passing the respective class
        objects via the `valfunc_cls` list. Optionally, `valfunc_argfacs` takes a list
        of callables that generate additional kwargs for these constructors. By default,
        `valfunc_cls` will be chosen as `[None, ..., None, PosteriorMean]`, which
        corresponds to the (parallel) multi-step KnowledgeGradient. If, in addition,
        `k=1` and `q_1 = 1`, this reduces to the classic Knowledge Gradient.

        WARNING: The complexity of evaluating this function is exponential in the number
        of lookahead steps!

        Args:
            model: A fitted model.
            batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes for the
                `k` look-ahead steps.
            num_fantasies_c: A list `[c_1, ..., c_k]` containing the number of
                constrained fantasies to be used in each stage.
            num_fantasies_uc: A list `[uc_1, ..., uc_k]` containing the number of
                unconstrained fantasies to be used in each stage.
            switch_cost: The cost of unconstrained evaluation.
            xc_dims: The indices of the costly dimensions.
            alpha_exp: The exponent of the cost parameter in the acquisition function.
            constrained: Boolean if X at step 0 is constrained on the last evaluation.
            samplers: A list of MCSampler objects to be used for sampling fantasies in
                each stage.
            valfunc_cls: A list of `k + 1` acquisition function classes to be used as
                the (stage + terminal) value functions. Each element (except for the
                last one) can be `None`, in which case a zero stage value is assumed for
                the respective stage. If `None`, this defaults to
                `[None, ..., None, PosteriorMean]`
            valfunc_argfacs: A list of `k + 1` "argument factories", i.e. callables that
                map a `Model` and input tensor `X` to a dictionary of kwargs for the
                respective stage value function constructor (e.g. `best_f` for
                `ExpectedImprovement`). If None, only the standard (`model`, `sampler`
                and `objective`) kwargs will be used.
            objective: The objective under which the output is evaluated. If `None`, use
                the model output (requires a single-output model or a posterior
                transform). Otherwise the objective is MC-evaluated
                (using `inner_sampler`).
            posterior_transform: An optional PosteriorTransform. If given, this
                transforms the posterior before evaluation. If `objective is None`,
                then the output of the transformed posterior is used. If `objective` is
                given, the `inner_sampler` is used to draw samples from the transformed
                posterior, which are then evaluated under the `objective`.
            inner_mc_samples: A list `[n_0, ..., n_k]` containing the number of MC
                samples to be used for evaluating the stage value function. Ignored if
                the objective is `None`.
            X_pending: A `m x d`-dim Tensor of `m` design points that have points that
                have been submitted for function evaluation but have not yet been
                evaluated. Concatenated into `X` upon forward call. Copied and set to
                have no gradient.
            collapse_fantasy_base_samples: If True, collapse_batch_dims of the Samplers
                will be applied on fantasy batch dimensions as well, meaning that base
                samples are the same in all subtrees starting from the same level.
        """
        # Compute total number of fantasies by summing constrained and unconstrained fantasies
        num_fantasies = [c + uc for c, uc in zip(num_fantasies_c, num_fantasies_uc)]

        # Initialize the parent class with the computed total fantasies
        super().__init__(
            model=model,
            batch_sizes=batch_sizes,
            num_fantasies=num_fantasies,
            samplers=samplers,
            valfunc_cls=valfunc_cls,
            valfunc_argfacs=valfunc_argfacs,
            objective=objective,
            posterior_transform=posterior_transform,
            inner_mc_samples=inner_mc_samples,
            X_pending=X_pending,
            collapse_fantasy_base_samples=collapse_fantasy_base_samples,
        )
        # raise error if any item in batch_sizes is not 1
        if any(q != 1 for q in batch_sizes):
            raise ValueError("Batch sizes > 1 are not supported.")

        # Store k as the number of steps in the lookahead
        self.k = len(batch_sizes)

        # Store the constrained and unconstrained fantasies separately
        self.num_fantasies_c = num_fantasies_c
        self.num_fantasies_uc = num_fantasies_uc

        # Store the switch cost and the costly dimensions
        self.switch_cost = switch_cost
        self.alpha_exp = alpha_exp
        self.xc_dims = xc_dims
        self.constrained = constrained

        # Store the binary tree shapes for constrained and unconstrained fantasies at each step
        # Example values:
        # num_fantasies_c = [1, 3,]  # Number of constrained fantasies at each step
        # num_fantasies_uc = [2, 4,]  # Number of unconstrained fantasies at each step
        # Then, the binary tree shapes will be:
        # c_uc_binary_tree_shapes_levels = [
        #   [()],
        #   [(1,), (2,)],
        #   [(3, 1), (4, 1), (3, 2), (4, 2)]
        # ]
        self.c_uc_binary_tree_shapes_levels = [
            [combination[::-1] for combination in itertools.product(
                *zip(self.num_fantasies_c[:i], self.num_fantasies_uc[:i])
            )] for i in range(self.k + 1)
        ]

        # Flatten in a single list
        self.c_uc_binary_tree_shapes = list(itertools.chain(*self.c_uc_binary_tree_shapes_levels))

        # Store the DFS group indices for the binary tree shapes
        # For a complete binary tree of depth/steps k=2, the binary tree looks like:
        # dfs_branch_group_indices = [0, 0, 1, 0, 2, 1, 3]
        self.dfs_branch_group_indices = _dfs_branch_group_indices(self.k+1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qMultiStepLookahead on the candidate set X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        Xs = self.get_multi_step_tree_input_representation(X)

        # set batch_range on samplers if not collapsing on fantasy dims
        if not self._collapse_fantasy_base_samples:
            self._set_samplers_batch_range(batch_shape=X.shape[:-2])

        return _step(
            model=self.model,
            Xs=Xs,
            nf_c=self.num_fantasies_c,
            nf_uc=self.num_fantasies_uc,
            switch_cost=self.switch_cost,
            alpha_exp=self.alpha_exp,
            xc_dims=self.xc_dims,
            samplers=self.samplers,
            valfunc_cls=self._valfunc_cls,
            valfunc_argfacs=self._valfunc_argfacs,
            inner_samplers=self.inner_samplers,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            running_val=None,
        )

    def get_split_shapes_binary_tree(self, X: Tensor) -> tuple[Any, list[Size], list[Any]]:
        r"""
        Get the split shapes from X, considering binary tree combinations of constrained (f_c) and unconstrained
        (f_uc) fantasies. This is organized into a single list where each entry corresponds to a unique combination
        of fantasies determined by its binary tree level and position.

        Args:
            X: A `batch_shape x q_aug x d`-dim tensor including fantasy points.

        Returns:
            A 3-tuple `(batch_shape, shapes, sizes)`, where:
            - `shapes` is a list of tensor shapes, where each shape `shapes[i]` is determined by:
              `shapes[i] = f_uc(step) x ... x f_c1 x batch_shape x q(step+1) x d`
              Here, `i = 2^step + j` (1-indexed) for `0 <= j < 2^step`, where `step` is the depth in the binary tree (0-indexed,
              from 0 to k), and `j` represents the binary representation of the path taken to reach the node.
            - `sizes` contains the total number of elements for each shape in `shapes`, calculated as:
              `size[i] = (f_uc(step) * ... * f_c1) * q(step+1)`

        Each index `i` in `shapes` and `sizes` can be decomposed into `2^step + j` where `step` is the depth of the node in the
        binary tree and `j` represents the node's index at that depth, encoded in binary as specified by the unique path
        of choices between `f_c` and `f_uc`.

        Example:
            If there are `step=3` steps and `j = 1` (binary 001), it means:
                - Step 3: use `f_uc` (bit 2)
                - Step 2: use `f_c`  (bit 1)
                - Step 1: use `f_c`  (bit 0)
            This results in the tensor shape `f_uc3 x f_c2 x f_c1 x batch_shape x q4 x d` for `shapes[i]`,
            where `i = 2^3 + 1 = 9`.
        """
        batch_shape, (q_aug, d) = X.shape[:-2], X.shape[-2:]
        q = q_aug - self._num_auxiliary
        batch_sizes = [q] + self.batch_sizes
        # Calculating all possible shapes at each step in the lookahead by considering all combinations of
        # constrained and unconstrained fantasies. Each shape represents a possible configuration of the fantasy
        # samples arranged before the batch dimensions and spatial dimensions.
        shapes_tree_level = [
            [torch.Size(list(combination) + [*batch_shape, q_i, d]) for combination in combination_list]
            for i, (q_i, combination_list) in enumerate(zip(batch_sizes, self.c_uc_binary_tree_shapes_levels))
        ]
        # Flatten shapes_tree
        shapes_tree = list(itertools.chain(*shapes_tree_level))
        # Calculating the total number of elements for each shape configuration in `shapes`.
        # This calculation multiplies the number of elements in the fantasy dimensions by the batch size dimension at
        # that step.
        sizes_tree = [s[: (-2 - len(batch_shape))].numel() * s[-2] for s in shapes_tree]
        return batch_shape, shapes_tree, sizes_tree

    def get_binary_tree_input_representation(self, X: Tensor) -> List[Tensor]:
        r"""Get the multi-step tree representation of X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i`( = `f_c(i)` + `f_uc(i)`) is respective number of fantasies.

        Returns:
            A list of n = (2^(k+1) - 1) tensors `[X_0, ..., X_n]` of tensors, where `X_j` has some shape
            `f_uc(i) x ... x f_c1 x batch_shape x q_i x d` from the flatten shapes_tree. Note that i is
            the upper log base 2 of j.
        """
        batch_shape, shapes_tree, sizes_tree = self.get_split_shapes_binary_tree(X=X)
        # Each X_i in Xsplit has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        Xsplit = torch.split(X, sizes_tree, dim=-2)
        # now reshape (need to permute batch_shape and qtilde dimensions for i > 0)
        perm = [-2] + list(range(len(batch_shape))) + [-1]
        X0 = Xsplit[0].reshape(shapes_tree[0])
        # This takes all X_i and reshapes them from
        # batch_shape x qtilde x d to qtilde x batch_shape x d initially,
        # then reshapes to f_i x ... x f_1 x batch_shape x q_(i+1) x d
        # FIXME: This doesn't work well for q_i > 1
        Xother = [
            X.permute(*perm).reshape(shape) for X, shape in zip(Xsplit[1:], shapes_tree[1:])
        ]
        # concatenate in pending points
        if self.X_pending is not None:
            X0 = torch.cat([X0, match_batch_shape(self.X_pending, X0)], dim=-2)

        return [X0] + Xother

    def get_multi_step_tree_input_representation(self, X: Tensor) -> List[Tensor]:
        r"""Get the multi-step tree representation of X.

        TODO: Override this method and fix the implementation for q_i > 1.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            A list `[X_j, ..., X_k]` of tensors, where `X_i` has shape
            `f_i x .... x f_1 x batch_shape x q_i x d`.

        """
        batch_shape, shapes, sizes = self.get_split_shapes(X=X)
        # Each X_i in Xsplit has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        Xsplit = torch.split(X, sizes, dim=-2)
        # now reshape (need to permute batch_shape and qtilde dimensions for i > 0)
        perm = [-2] + list(range(len(batch_shape))) + [-1]
        X0 = Xsplit[0].reshape(shapes[0])
        Xother = [
            X.permute(*perm).reshape(shape) for X, shape in zip(Xsplit[1:], shapes[1:])
        ]
        # concatenate in pending points
        if self.X_pending is not None:
            X0 = torch.cat([X0, match_batch_shape(self.X_pending, X0)], dim=-2)

        return [X0] + Xother


def _step(
        model: Model,
        Xs: List[Tensor],
        nf_c: List[int],
        nf_uc: List[int],
        switch_cost: float,
        alpha_exp: float,
        xc_dims: Tensor[int],
        samplers: List[Optional[MCSampler]],
        valfunc_cls: List[Optional[Type[AcquisitionFunction]]],
        valfunc_argfacs: List[Optional[TAcqfArgConstructor]],
        inner_samplers: List[Optional[MCSampler]],
        objective: MCAcquisitionObjective,
        posterior_transform: Optional[PosteriorTransform],
        running_val: Optional[Tensor] = None,
        sample_weights: Optional[Tensor] = None,
        sample_costs: Optional[Tensor] = None,
        step_index: int = 0,
) -> Tensor:
    r"""Recursive multi-step look-ahead computation.

    Helper function computing the "value-to-go" of a multi-step lookahead scheme.

    Args:
        model: A Model of appropriate batch size. Specifically, it must be possible to
            evaluate the model's posterior at `Xs[0]`.
        Xs: A list `[X_j, ..., X_k]` of tensors, where `X_i` has shape
            `f_i x .... x f_1 x batch_shape x q_i x d`.
        nf_c: A list `[c_1, ..., c_k]` containing the number of constrained fantasies to
            be used in each stage.
        nf_uc: A list `[uc_1, ..., uc_k]` containing the number of unconstrained fantasies
            to be used in each stage.
        switch_cost: The cost of unconstrained evaluation.
        alpha_exp: The exponent of the cost parameter in the acquisition function.
        xc_dims: The indices of the costly dimensions.
        samplers: A list of `k - j` samplers, such that the number of samples of sampler
            `i` is `f_i`. The last element of this list is considered the
            "inner sampler", which is used for evaluating the objective in case it is an
            MCAcquisitionObjective.
        valfunc_cls: A list of acquisition function class to be used as the (stage +
            terminal) value functions. Each element (except for the last one) can be
            `None`, in which case a zero stage value is assumed for the respective
            stage.
        valfunc_argfacs: A list of callables that map a `Model` and input tensor `X` to
            a dictionary of kwargs for the respective stage value function constructor.
            If `None`, only the standard `model`, `sampler` and `objective` kwargs will
            be used.
        inner_samplers: A list of `MCSampler` objects, each to be used in the stage
            value function at the corresponding index.
        objective: The MCAcquisitionObjective under which the model output is evaluated.
        posterior_transform: A PosteriorTransform. Used to transform the posterior
            before sampling / evaluating the model output.
        running_val: As `batch_shape`-dim tensor containing the current running value.
        sample_weights: A tensor of shape `f_i x .... x f_1 x batch_shape` when called
            in the `i`-th step by which to weight the stage value samples. Used in
            conjunction with Gauss-Hermite integration or importance sampling. Assumed
            to be `None` in the initial step (when `step_index=0`).
        sample_costs: A tensor of shape `f_i x .... x f_1 x batch_shape` when called in
            the `i`-th step by which to weight the stage value samples. Used in
            conjunction with Gauss-Hermite integration or importance sampling. Assumed
            to be `None` in the initial step (when `step_index=0`).
        step_index: The index of the look-ahead step. `step_index=0` indicates the
            initial step.

    Returns:
        A `b`-dim tensor containing the multi-step value of the design `X`.
    """
    X = Xs[0]

    if sample_weights is None:  # only happens in the initial step
        sample_weights = torch.ones(*X.shape[:-2], device=X.device, dtype=X.dtype)

    if sample_costs is None:
        # Check if X has the same costly dimensions as the previous step. Returns size of batch_shape
        constrained = torch.eq(X[..., xc_dims],
                               model.input_transform.untransform(
                                   model.train_inputs[0]
                               )[-1, xc_dims]
        ).all(-1).all(-1)
        # 1 if constrained, switch_cost if unconstrained
        sample_costs = torch.where(constrained, torch.tensor(1.0, device=X.device, dtype=X.dtype), switch_cost)

    # compute stage value
    stage_val = _compute_stage_value(
        model=model,
        valfunc_cls=valfunc_cls[0],
        X=X,
        objective=objective,
        posterior_transform=posterior_transform,
        inner_sampler=inner_samplers[0],
        arg_fac=valfunc_argfacs[0],
    )
    if stage_val is not None:  # update running value
        # if not None, running_val has shape f_{i-1} x ... x f_1 x batch_shape
        # stage_val has shape f_i x ... x f_1 x batch_shape

        # this sum will add a dimension to running_val so that
        # updated running_val has shape f_i x ... x f_1 x batch_shape
        running_val = stage_val if running_val is None else running_val + stage_val

    # base case: no more fantasizing, return value
    if len(Xs) == 1:
        # compute weighted average over all leaf nodes of the tree
        batch_shape = running_val.shape[step_index:]
        # expand sample weights to make sure it is the same shape as running_val,
        # because we need to take a sum over sample weights for computing the
        # weighted average
        sample_weights = sample_weights.expand(running_val.shape)
        sample_costs = sample_costs.expand(running_val.shape)
        return (running_val * sample_weights / sample_costs ** alpha_exp).view(-1, *batch_shape).sum(dim=0)

    # construct fantasy model (with batch shape f_{j+1} x ... x f_1 x batch_shape)
    prop_grads = step_index > 0  # need to propagate gradients for steps > 0
    fantasy_model = model.fantasize(
        X=X, sampler=samplers[0], propagate_grads=prop_grads
    )

    # augment sample weights appropriately
    sample_weights = _construct_sample_weights(
        prev_weights=sample_weights, sampler=samplers[0]
    )

    # augment sample costs appropriately
    sample_costs = _construct_sample_costs(
        prev_costs=sample_costs,
        nf_c=nf_c[0],
        nf_uc=nf_uc[0],
        cost=switch_cost,
    )

    return _step(
        model=fantasy_model,
        Xs=Xs[1:],
        nf_c=nf_c[1:],
        nf_uc=nf_uc[1:],
        switch_cost=switch_cost,
        alpha_exp=alpha_exp,
        xc_dims=xc_dims,
        samplers=samplers[1:],
        valfunc_cls=valfunc_cls[1:],
        valfunc_argfacs=valfunc_argfacs[1:],
        inner_samplers=inner_samplers[1:],
        objective=objective,
        posterior_transform=posterior_transform,
        sample_weights=sample_weights,
        sample_costs=sample_costs,
        running_val=running_val,
        step_index=step_index + 1,
    )


def _construct_sample_weights(
        prev_weights: Tensor, sampler: MCSampler
) -> Optional[Tensor]:
    r"""Iteratively construct tensor of sample weights for multi-step look-ahead.

    Args:
        prev_weights: A `f_i x .... x f_1 x batch_shape` tensor of previous sample
            weights.
        sampler: A `MCSampler` that may have sample weights as the `base_weights`
            attribute. If the sampler does not have a `base_weights` attribute,
            samples are weighted uniformly.

    Returns:
        A `f_{i+1} x .... x f_1 x batch_shape` tensor of sample weights for the next
        step.
    """
    new_weights = getattr(sampler, "base_weights", None)  # TODO: generalize this
    if new_weights is None:
        # uniform weights
        nf = sampler.sample_shape[0]
        new_weights = torch.ones(
            nf, device=prev_weights.device, dtype=prev_weights.dtype
        )
    # reshape new_weights to be f_{i+1} x 1 x ... x 1
    new_weights = new_weights.view(-1, *(1 for _ in prev_weights.shape))
    # normalize new_weights to sum to 1.0
    new_weights = new_weights / new_weights.sum()
    return new_weights * prev_weights


def _construct_sample_costs(
        prev_costs: Tensor,
        nf_c: int,
        nf_uc: int,
        cost: float,
) -> Optional[Tensor]:
    r"""Iteratively construct tensor of sample costs for multi-step look-ahead.

    Args:
        prev_costs: A `f_i x .... x f_1 x batch_shape` tensor of previous sample
            costs.
        nf_c: Number of constrained fantasies for the next step.
        nf_uc: Number of unconstrained fantasies for the next step.
        cost: The cost of unconstrained evaluation.

    Returns:
        A `f_{i+1} x .... x f_1 x batch_shape` tensor of sample costs for the next
        step.
    """
    # TODO: generalize this by checking if the last setup repeats
    # Cost is 1 for constrained fantasies and `cost` for unconstrained fantasies
    new_costs = torch.Tensor([1] * nf_c + [cost] * nf_uc)
    # reshape new_costs to be f_{i+1} x 1 x ... x 1
    new_costs = new_costs.view(-1, *(1 for _ in prev_costs.shape))
    return new_costs + prev_costs

