"""Figures of a sweep: the polarization curve and through-plane profiles."""

import matplotlib.pyplot as plt

from pemfc_model_1d import layers as ly


def plot_polarization_curve(sweep):
    fig, ax = plt.subplots()
    ax.plot(sweep.current_densities / 1e4, sweep.voltages, "o-", markersize=3)
    ax.set_xlabel("Current density [A/cm$^2$]")
    ax.set_ylabel("Cell voltage [V]")
    ax.grid(True, alpha=0.3)
    return fig


def plot_profiles(sweep, names=None):
    """One panel per unknown (or flux) in `names`, one line per operating point."""
    names = names or list(ly.VARIABLES)
    fig, axes = plt.subplots(2, -(-len(names) // 2), figsize=(12, 6), squeeze=False)
    boundaries = [
        profile.x[-1] for profile in list(sweep.points[0].layers.values())[:-1]
    ]
    for ax, name in zip(axes.flat, names, strict=False):
        for k, point in enumerate(sweep.points):
            color = plt.cm.viridis(k / max(len(sweep.points) - 1, 1))
            for profile in point.layers.values():
                if name in profile.values:
                    ax.plot(profile.x * 1e6, profile.values[name], color=color)
        for x in boundaries:
            ax.axvline(x * 1e6, color="0.8", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("x [µm]")
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
