import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_objective(obj, show_marginals=[]):
    assert obj.dim == 2, "Can only plot 2d functions"

    # Grid generation
    x = np.linspace(*obj.bounds.T[0], 100)
    y = np.linspace(*obj.bounds.T[1], 100)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1))

    # Function evaluation
    Z = obj(points).numpy().reshape(X.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=3)

    # Check if marginals should be shown
    if show_marginals:
        # Create subplots for marginals
        ax_x = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=3)
        ax_y = plt.subplot2grid((4, 4), (0, 3), rowspan=3, colspan=1)

        # Plot on appropriate axes based on the marginals list
        if 'max' in show_marginals:
            ax_x.plot(x, Z.max(axis=0), 'b-', label='Max value per x1')
            ax_y.plot(Z.max(axis=1), y, 'b-', label='Max value per x2')

            if 'mean' in show_marginals or 'median' in show_marginals:
                ax_x2 = ax_x.twinx()
                ax_y2 = ax_y.twiny()

                if 'mean' in show_marginals:
                    ax_x2.plot(x, Z.mean(axis=0), 'g--', label='Mean value per x1')
                    ax_y2.plot(Z.mean(axis=1), y, 'g--', label='Mean value per x2')

                if 'median' in show_marginals:
                    ax_x2.plot(x, np.median(Z, axis=0), 'r-.', label='Median value per x1')
                    ax_y2.plot(np.median(Z, axis=1), y, 'r-.', label='Median value per x2')
        else:
            if 'mean' in show_marginals:
                ax_x.plot(x, Z.mean(axis=0), 'g--', label='Mean value per x1')
                ax_y.plot(Z.mean(axis=1), y, 'g--', label='Mean value per x2')

            if 'median' in show_marginals:
                ax_x.plot(x, np.median(Z, axis=0), 'r-.', label='Median value per x1')
                ax_y.plot(np.median(Z, axis=1), y, 'r-.', label='Median value per x2')


        # Optima lines
        for optima in obj.optimizers[:, 0]:
            ax_x.axvline(optima, c='purple', linestyle='--')
        for optima in obj.optimizers[:, 1]:
            ax_y.axhline(optima, c='purple', linestyle='--')
    else:
        ax_x = ax_y = None

    # Main contour plot
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    ax.scatter(*torch.Tensor(obj.optimizers).T, c='purple', marker='*', s=50, label='optima')
    ax.set_title(f'2D {obj._get_name()} Function')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # Create an empty plot with dummy entries to create the legend
    ax_legend = plt.subplot2grid((4, 4), (3, 3))
    if 'max' in show_marginals:
        ax_legend.plot([], [], 'b-', label='Max')
    if 'mean' in show_marginals:
        ax_legend.plot([], [], 'g--', label='Mean')
    if 'median' in show_marginals:
        ax_legend.plot([], [], 'r-.', label='Median')
    ax_legend.plot([], [], 'purple', linestyle='--', label='Optima')
    ax_legend.legend(loc='center')
    ax_legend.axis('off')

    if show_marginals:
        return fig, (ax, ax_x, ax_y)
    else:
        return fig, ax