"""Six-panel relative difference maps and regime maps.

The notebook owns the solutions, differences, domains, regimes, and common
interpolation grid. This module contains plotting functions only.

Inputs shared by both plotters:
    XI, YI: physical parameter coordinate meshes, shape (n_y, n_x).
    polygon: an (n_vertices, 2) array of (rho, Sigma_0') coordinates.
    x_cuts: the ordered six-cut dictionary used to compute sol_dif.

plot_relative_differences additionally takes points, whose columns are
log10(rho) and log10(Sigma_0') at the original solution nodes. Metrics are
"combined", "delta_w", or "delta_p". The common finite input samples across
sol_dif's models match the domain-calculation cell in the example notebook.

plot_regime_maps takes regime_maps[metric][cut_key]["R0" ... "R3"].
Both functions return (fig, axes); call plt.show() in the notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import NullLocator
from scipy.interpolate import griddata


DEFAULT_X_CUTS = {
    "1e-2": 1e-2,
    "1e-1": 1e-1,
    "1e0": 1.0,
    "1e1": 10.0,
    "1e2": 100.0,
    "inf": np.inf,
}

REGIME_COLORS = ("#8da0cb", "#fc8d62", "#66c2a5", "#e78ac3")


def _title(distance):
    if np.isinf(distance):
        return r"$\xi_0=\infty$"
    labels = {1e-2: r"10^{-2}", 1e-1: r"10^{-1}", 1.0: "1", 10.0: "10", 100.0: r"10^2"}
    return rf"$\ell={labels.get(distance, f'{distance:g}')}\,\mathrm{{m}}$"


def _error_samples(sol_dif, cut_key, metric):
    samples = {}
    for model, by_cut in sol_dif.items():
        fields = by_cut[cut_key]
        error = (np.maximum(fields["delta_w"], fields["delta_p"])
                 if metric == "combined" else fields[metric])
        samples[model] = np.asarray(error).ravel()
    valid = np.logical_and.reduce([np.isfinite(z) for z in samples.values()])
    return samples, valid


def _make_axes(x_cuts, polygon, figsize, xlim, ylim):
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True,
                             sharey=True, layout="constrained")
    outline = np.vstack((polygon, polygon[0]))
    for ax, distance in zip(axes.ravel(), x_cuts.values()):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(_title(distance), fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=11,
                       labelbottom=True, labelleft=True)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.plot(outline[:, 0], outline[:, 1], color="black", lw=1.0, zorder=2)
    for ax in axes.ravel():
        ax.set_xlabel(r"$\varrho$", fontsize=16)
    for ax in axes.ravel():
        ax.set_ylabel(r"$\Sigma_0'$", fontsize=16)
    return fig, axes


def plot_relative_differences(
    sol_dif,
    model_name,
    points,
    XI,
    YI,
    polygon,
    *,
    metric="combined",
    x_cuts=None,
    threshold=5.0,
    method="cubic",
    figsize=(12, 14),
    xlim=(1e-8, 2e6),
    ylim=(3e-3, 2e5),
    cmap="rainbow",
    colorbar_format="%.0f",
    filename=None,
    dpi=300,
):
    x_cuts = DEFAULT_X_CUTS if x_cuts is None else x_cuts
    polygon = np.asarray(polygon)
    XIlog, YIlog = np.log10(XI), np.log10(YI)
    label = {"combined": r"$\Delta$, %", "delta_w": r"$\Delta_{\Omega}$, %",
             "delta_p": r"$\Delta_{\Pi}$, %"}[metric]

    samples_inf, valid_inf = _error_samples(sol_dif, "inf", metric)
    vmax = max(float(np.max(samples_inf[model_name][valid_inf])), 1e-12)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    levels = np.linspace(0.0, vmax, 501)
    fig, axes = _make_axes(x_cuts, polygon, figsize, xlim, ylim)

    for ax, cut_key in zip(axes.ravel(), x_cuts):
        samples, valid = _error_samples(sol_dif, cut_key, metric)
        z = samples[model_name][valid]
        panel_max = float(np.max(z))
        ZI = griddata(points[valid], z, (XIlog, YIlog), method=method)
        ZI = np.clip(ZI, np.min(z), panel_max)

        filled = ax.contourf(XI, YI, ZI, levels=levels, cmap=cmap, norm=norm, zorder=0)
        hatched = ax.contourf(XI, YI, ZI, levels=[0.0, threshold],
                              colors="none", hatches=["///"], zorder=1)
        clip = Polygon(polygon, closed=True, transform=ax.transData)
        filled.set_clip_path(clip)
        hatched.set_clip_path(clip)
        ax.set_rasterization_zorder(1)

        bar_max = panel_max if panel_max > 0 else vmax / 500.0
        ticks = np.linspace(0.0, panel_max, 6) if panel_max > 0 else [0.0]
        cbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=ax,
            boundaries=np.linspace(0.0, bar_max, 257), ticks=ticks,
            spacing="proportional", fraction=0.05, pad=0.03,
            format=colorbar_format,
        )
        cbar.set_label(label, fontsize=14)
        cbar.ax.tick_params(labelsize=11)

    if filename is not None:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    return fig, axes


def plot_regime_maps(
    regime_maps,
    XI,
    YI,
    polygon,
    *,
    metric="combined",
    x_cuts=None,
    figsize=(12, 14),
    xlim=(1e-8, 2e6),
    ylim=(3e-3, 2e5),
    colors=REGIME_COLORS,
    alpha=0.6,
    legend=True,
    filename=None,
    dpi=300,
):
    x_cuts = DEFAULT_X_CUTS if x_cuts is None else x_cuts
    polygon = np.asarray(polygon)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)
    fig, axes = _make_axes(x_cuts, polygon, figsize, xlim, ylim)

    for ax, cut_key in zip(axes.ravel(), x_cuts):
        masks = regime_maps[metric][cut_key]
        labels = np.full(XI.shape, np.nan)
        for i in range(4):
            labels[masks[f"R{i}"]] = i
        mesh = ax.pcolormesh(
            XI, YI, np.ma.masked_invalid(labels), shading="nearest",
            cmap=cmap, norm=norm, alpha=alpha, rasterized=True,
        )
        mesh.set_clip_path(Polygon(polygon, closed=True, transform=ax.transData))

    if legend:
        handles = [Patch(facecolor=color, edgecolor="none", alpha=alpha,
                         label=rf"$\mathcal{{R}}_{{{i}}}$")
                   for i, color in enumerate(colors)]
        axes[0, 0].legend(handles=handles, loc="upper right", fontsize=12, framealpha=1.0)

    if filename is not None:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    return fig, axes
