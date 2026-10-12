"""Figures written during the search: the current Pareto front and the hypervolume history."""

import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_front(step, F, baseline=None, title='Pareto Front', ylabel='Calibration accuracy (%)',
                      fig_path='./pareto_front.png', final=False, wandb_run=None):
    """Plot a front of (1 - accuracy, 1/CR) objective vectors as accuracy vs. CR."""
    if F is None or len(F) == 0:
        return
    cr = 1 / F[:, 1]
    acc = 100 * (1 - F[:, 0])
    order = np.argsort(cr)

    plt.figure(figsize=(5, 5))
    if baseline is not None:
        plt.axhline(y=baseline * 100, color='r', linestyle='--', linewidth=2, label='Uncompressed', alpha=0.7)
    plt.plot(cr[order], acc[order], marker='o', linestyle='-', color='tab:blue', markersize=6,
             markeredgecolor='k', alpha=0.8, linewidth=1.2, label='Pareto front')
    plt.xlim(1, max(32, 1.05 * cr[np.isfinite(cr)].max(initial=1)))
    plt.ylim(0, 100)
    plt.xlabel('Compression ratio')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    if wandb_run is not None:
        import wandb
        key = f"figures/{'final_' if final else ''}pareto_front"
        wandb_run.log({key: wandb.Image(fig_path, caption=title)}, step=step)


def plot_hypervolume_history(steps, hv_values, fig_path='./hypervolume_history.png',
                             title='Hypervolume during evolution', final=False, wandb_run=None):
    """Plot the hypervolume of the front after each generation."""
    if len(steps) == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(steps, hv_values, 'b-o', markersize=4, linewidth=1.5)
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    if wandb_run is not None:
        import wandb
        key = f"figures/{'final_' if final else ''}hypervolume_history"
        wandb_run.log({key: wandb.Image(fig_path, caption=title)}, step=steps[-1])
