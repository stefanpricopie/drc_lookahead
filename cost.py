import torch

from abc import ABC


class SetupCostModel(torch.nn.Module, ABC):
    """
    A basic cost model that assumes the cost is positive.
    It models the log cost to guarantee positive cost predictions.
    """

    def __init__(self, switching_cost, xc_dims, last_x_train=None):
        """
        :param switching_cost (float): Setup cost for each new evaluation in the xc_dim dimensions.
        :param xc_dims (Tensor): Indices of the dimensions that require a setup cost
        """
        super().__init__()
        self.switching_cost = switching_cost    # Note: switching_cost is relative to fixed_cost=1
        self.fixed_cost = 1
        self.xc_dims = xc_dims
        self.last_x_train = last_x_train

    def update_last_x_train(self, last_x_train):
        self.last_x_train = last_x_train

    def forward(self, X):
        # Check if last_x_train is provided, otherwise default to True
        if self.last_x_train is None:
            # Check if row i is different from row i-1
            shifted_X = torch.roll(X, shifts=1, dims=0)

            # Create a tensor of True values with the same batch shape as X
            switch_bool = torch.ne(X[:, self.xc_dims], shifted_X[:, self.xc_dims]).any(dim=-1)

            # First value is always True
            switch_bool[0] = True
        else:
            # torch.ne is the element-wise version of !=
            switch_bool = torch.ne(X[:, self.xc_dims], self.last_x_train[self.xc_dims]).any(dim=-1)

        # Calculate the total cost
        # switch_bool is a boolean tensor and needs to be cast to the same type as switching_cost for multiplication
        return switch_bool.float() * self.switching_cost + self.fixed_cost

    def __str__(self):
        return f"SC={self.switching_cost} XC={{{','.join(map(str, self.xc_dims.tolist()))}}})"
