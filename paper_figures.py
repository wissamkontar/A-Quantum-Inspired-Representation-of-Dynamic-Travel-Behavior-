#!/usr/bin/env python3
"""
Figure-generation utilities for
"A Quantum-Inspired Representation of Dynamic Travel Behavior".

This module consolidates the plotting code used to generate several
figures in the paper. It is intended to be used after the main model,
comparison models, and/or counterfactual analysis have produced the
necessary arrays and tables.

This file does not re-estimate the model. It only produces figures from
previously estimated model outputs.

Typical use
-----------
Import the functions into an analysis script or notebook:

    from paper_figures import (
        plot_individual_temperature_shock,
        plot_population_temperature_shock,
        plot_context_activation,
        plot_profile_fingerprints,
        plot_state_evolution,
    )

All plotting functions save both PDF and PNG versions of the figure.

Dependencies
------------
numpy
pandas
matplotlib
scipy
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


# ============================================================
# Global plotting defaults
# ============================================================

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 14,
})

DEFAULT_PROFILE_NAMES = [
    "Shopper/Leisure",
    "Commuter",
    "Home Anchor",
]

DEFAULT_PROFILE_COLORS = [
    "#90CAF9",
    "#FFAB91",
    "#A5D6A7",
]


def _ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================
# 1. Individual temperature-shock figure
# ============================================================

def plot_individual_temperature_shock(
    pi_baseline,
    pi_shocked,
    actual_temps,
    shocked_temps,
    shock_start,
    shock_end,
    temp_increase,
    output_dir=".",
    profile_names=None,
    profile_colors=None,
    filename="counterfactual_temperature_shock",
):
    """
    Plot the episode-level counterfactual response for one traveler.

    Parameters
    ----------
    pi_baseline : ndarray, shape (T, K)
        Baseline profile-activation probabilities.

    pi_shocked : ndarray, shape (T, K)
        Profile activations under the temperature shock.

    actual_temps : array-like, length T
        Observed temperatures.

    shocked_temps : array-like, length T
        Counterfactual temperatures.

    shock_start, shock_end : int
        Beginning and end of the shocked episode window.

    temp_increase : float
        Temperature increase applied in the counterfactual.

    Notes
    -----
    The upper panel reports the difference

        pi_shocked - pi_baseline

    for each behavioral profile. The lower panel shows the observed and
    counterfactual temperature series.
    """
    output_dir = _ensure_output_dir(output_dir)

    profile_names = profile_names or DEFAULT_PROFILE_NAMES
    profile_colors = profile_colors or DEFAULT_PROFILE_COLORS

    pi_baseline = np.asarray(pi_baseline)
    pi_shocked = np.asarray(pi_shocked)
    actual_temps = np.asarray(actual_temps)
    shocked_temps = np.asarray(shocked_temps)

    T, K = pi_baseline.shape
    episodes = np.arange(T)

    if pi_shocked.shape != pi_baseline.shape:
        raise ValueError("pi_baseline and pi_shocked must have the same shape.")

    pi_delta_individual = pi_shocked - pi_baseline

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    for ax in axes:
        ax.axvspan(
            shock_start,
            shock_end,
            alpha=0.08,
            color="red",
        )
        ax.axvline(
            shock_start,
            color="red",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
        )
        ax.axvline(
            shock_end,
            color="red",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
        )

    # Panel 1: change in profile activation
    ax1 = axes[0]
    ax1.axhline(
        0,
        color="black",
        linewidth=0.8,
        linestyle="--",
    )

    for k in range(K):
        ax1.plot(
            episodes,
            pi_delta_individual[:, k],
            color=profile_colors[k],
            linewidth=2.0,
            label=profile_names[k],
        )

    ax1.set_ylabel(
        r"$\Delta$ Profile Activation $\pi_k$",
        fontsize=13,
    )
    ax1.legend(
        loc="upper right",
        fontsize=11,
    )
    ax1.set_xlim(0, T - 1)
    ax1.tick_params(
        axis="both",
        labelsize=11,
    )
    ax1.grid(
        True,
        alpha=0.3,
    )
    ax1.tick_params(
        axis="x",
        labelbottom=False,
    )

    # Panel 2: temperature
    ax2 = axes[1]
    ax2.plot(
        episodes,
        actual_temps,
        color="black",
        linewidth=1.2,
        linestyle="-",
        label="Actual temperature",
    )
    ax2.plot(
        episodes,
        shocked_temps,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=f"Simulated (+{temp_increase:.0f}°C)",
    )
    ax2.set_ylabel(
        "Temp (°C)",
        fontsize=13,
    )
    ax2.set_xlabel(
        "Episode",
        fontsize=13,
    )
    ax2.legend(
        loc="upper right",
        fontsize=10,
    )
    ax2.set_xlim(0, T - 1)
    ax2.tick_params(
        axis="both",
        labelsize=11,
    )
    ax2.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{filename}.{ext}",
            bbox_inches="tight",
            dpi=150,
        )

    plt.close(fig)

    print(f"Saved: {filename}.pdf / .png")


# ============================================================
# 2. Population temperature-shock figure
# ============================================================

def plot_population_temperature_shock(
    pi_delta,
    mode_delta,
    mode_names,
    n_valid,
    output_dir=".",
    profile_names=None,
    profile_colors=None,
    mode_colors=None,
    filename="population_temperature_shock",
):
    """
    Plot the population-average effects of the temperature shock.

    Parameters
    ----------
    pi_delta : ndarray, shape (N, K)
        Participant-level changes in profile activation.

    mode_delta : ndarray, shape (N, M)
        Participant-level changes in modeled transport-mode share.

    mode_names : sequence of str
        Mode names corresponding to the M columns in mode_delta.

    n_valid : int
        Number of valid participants included in the counterfactual.

    Notes
    -----
    Bars report sample means. Error bars are 95% normal-approximation
    intervals using 1.96 times the standard error, matching the original
    paper plotting code.
    """
    output_dir = _ensure_output_dir(output_dir)

    profile_names = profile_names or DEFAULT_PROFILE_NAMES
    profile_colors = profile_colors or DEFAULT_PROFILE_COLORS

    if mode_colors is None:
        mode_colors = [
            "#F48FB1",
            "#CE93D8",
            "#FFCC80",
            "#B0BEC5",
            "#80DEEA",
            "#C5E1A5",
        ]

    pi_delta = np.asarray(pi_delta)
    mode_delta = np.asarray(mode_delta)

    K = pi_delta.shape[1]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    # Profile activation deltas
    ax1 = axes[0]

    means_p = [
        pi_delta[:, k].mean()
        for k in range(K)
    ]
    sems_p = [
        pi_delta[:, k].std() / np.sqrt(n_valid)
        for k in range(K)
    ]

    ax1.bar(
        profile_names,
        means_p,
        color=profile_colors,
        yerr=[1.96 * s for s in sems_p],
        capsize=4,
        error_kw={"linewidth": 1.2},
    )
    ax1.axhline(
        0,
        color="black",
        linewidth=0.8,
        linestyle="--",
    )
    ax1.set_ylabel(
        r"$\Delta$ Profile Activation",
        fontsize=13,
    )
    ax1.tick_params(
        axis="x",
        labelrotation=15,
        labelsize=12,
    )
    for label in ax1.get_xticklabels():
        label.set_ha("right")

    ax1.tick_params(
        axis="y",
        labelsize=11,
    )
    ax1.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    # Mode-share deltas
    ax2 = axes[1]

    means_m = [
        mode_delta[:, m].mean()
        for m in range(len(mode_names))
    ]
    sems_m = [
        mode_delta[:, m].std() / np.sqrt(n_valid)
        for m in range(len(mode_names))
    ]

    ax2.bar(
        mode_names,
        means_m,
        color=mode_colors[:len(mode_names)],
        yerr=[1.96 * s for s in sems_m],
        capsize=4,
        error_kw={"linewidth": 1.2},
    )
    ax2.axhline(
        0,
        color="black",
        linewidth=0.8,
        linestyle="--",
    )
    ax2.set_ylabel(
        r"$\Delta$ Mode Share",
        fontsize=13,
    )
    ax2.tick_params(
        axis="x",
        labelrotation=15,
        labelsize=12,
    )
    for label in ax2.get_xticklabels():
        label.set_ha("right")

    ax2.tick_params(
        axis="y",
        labelsize=11,
    )
    ax2.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    plt.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{filename}.{ext}",
            bbox_inches="tight",
            dpi=150,
        )

    plt.close(fig)

    print(f"Saved: {filename}.pdf / .png")


# ============================================================
# 3. Context activation coefficient figure
# ============================================================

def plot_context_activation(
    beta,
    context_labels,
    output_dir=".",
    profile_names=None,
    filename="C_context_activation",
):
    """
    Plot estimated context-activation coefficients for each profile.

    Parameters
    ----------
    beta : ndarray, shape (K, Q)
        Context coefficient matrix.

    context_labels : sequence of str
        Column names corresponding to beta.

    Notes
    -----
    The function reorders variables to the same presentation order used
    in the paper. Positive coefficients are shown in red and negative
    coefficients in blue.
    """
    output_dir = _ensure_output_dir(output_dir)
    profile_names = profile_names or DEFAULT_PROFILE_NAMES

    beta = np.asarray(beta).copy()
    context_labels = list(context_labels)

    desired_order = [
        "Time_Morning",
        "Time_Midday",
        "Time_Afternoon",
        "Time_Evening",
        "Time_Night",
        "Day_Weekday",
        "Day_Saturday",
        "Day_Sunday",
        "Trip_distance",
        "Temp_celsius",
        "Precip_depth",
    ]

    missing = [
        c for c in desired_order
        if c not in context_labels
    ]
    if missing:
        raise ValueError(
            "Missing required context labels: "
            + ", ".join(missing)
        )

    idx = [
        context_labels.index(c)
        for c in desired_order
    ]
    beta = beta[:, idx]
    context_labels = desired_order

    pretty_labels_map = {
        "Time_Morning": "Morning (6–10 AM) –",
        "Time_Midday": "Midday (10–14 PM) –",
        "Time_Afternoon": "Afternoon (14–18 PM) –",
        "Time_Evening": "Evening (18–22 PM) –",
        "Time_Night": "Night (22–6 AM) –",
        "Day_Weekday": "Weekday –",
        "Day_Saturday": "Saturday –",
        "Day_Sunday": "Sunday –",
        "Trip_distance": "Trip Distance –",
        "Temp_celsius": "Temperature –",
        "Precip_depth": "Precipitation –",
    }

    POS_COLOR = "#A83520"
    NEG_COLOR = "#4472C4"

    FS_TITLE = 16
    FS_TICK = 15
    FS_VALUE = 15

    K = beta.shape[0]

    abs_max = (
        np.ceil(np.max(np.abs(beta)) * 10) / 10
        + 0.05
    )

    fig, axes = plt.subplots(
        1,
        K,
        figsize=(5.5 * K, 5.8),
        sharey=False,
    )

    if K == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        row = beta[k]
        y_pos = np.arange(
            len(context_labels)
        )

        ax.axvline(
            0,
            color="#AAAAAA",
            lw=0.9,
            zorder=2,
        )

        for yp, val in zip(
            y_pos,
            row,
        ):
            bar_col = (
                POS_COLOR
                if val > 0
                else NEG_COLOR
            )

            ax.barh(
                yp,
                val,
                height=0.40,
                color=bar_col,
                alpha=0.90,
                zorder=3,
            )

            if abs(val) > 0.02:
                ha = (
                    "left"
                    if val >= 0
                    else "right"
                )
                pad = (
                    abs_max * 0.04
                    if val >= 0
                    else -abs_max * 0.04
                )

                ax.text(
                    val + pad,
                    yp,
                    f"{val:+.2f}",
                    va="center",
                    ha=ha,
                    fontsize=FS_VALUE,
                    fontweight="bold",
                    color=bar_col,
                )

        ax.set_yticks(y_pos)

        if k == 0:
            pretty_labels = [
                pretty_labels_map[c]
                for c in context_labels
            ]
            ax.set_yticklabels(
                pretty_labels,
                fontsize=FS_TICK,
                fontweight="normal",
            )
            ax.set_ylabel(
                "Context Variable",
                fontsize=FS_TICK,
                fontweight="normal",
                rotation=90,
                labelpad=28,
            )
        else:
            ax.set_yticklabels([])

        ax.invert_yaxis()
        ax.tick_params(
            left=False,
            pad=1,
        )

        ax.set_xticks([])
        ax.set_xlim(
            -abs_max * 1.35,
            abs_max * 1.35,
        )

        ax.set_title(
            f"Profile {k + 1}: {profile_names[k]}",
            fontsize=FS_TITLE,
            fontweight="bold",
            color="black",
            pad=28,
        )

        ax.text(
            0.5,
            -0.045,
            "Context Activation Weight (β)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FS_TICK,
            fontweight="normal",
            color="black",
        )

        for spine in (
            "top",
            "right",
            "left",
        ):
            ax.spines[spine].set_visible(False)

        ax.spines["bottom"].set_visible(False)
        ax.grid(
            axis="x",
            alpha=0.15,
            lw=0.5,
        )

    plt.tight_layout(
        w_pad=0.4
    )

    for ext in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{filename}.{ext}",
            dpi=600,
            bbox_inches="tight",
        )

    plt.close(fig)

    print(f"Saved: {filename}.pdf / .png")


# ============================================================
# 4. Profile fingerprint figures
# ============================================================

POS_COLOR = "#A83520"
NEG_COLOR = "#4472C4"
POLY_COLOR = "#4472C4"
BASE_COLOR = "#BBBBBB"

RING_DPP = [-30, 0, 30, 60]
OFFSET = 90
RING_R = [
    d + OFFSET
    for d in RING_DPP
]
R_MAX = 158

_CIRC = np.linspace(
    0,
    2 * np.pi,
    361,
)

FS_SUPTITLE = 15
FS_TITLE = 12
FS_ACT = 12.5
FS_TICK = 10
FS_VALUE = 11
FS_LEGEND = 11
COL_W = 5.5

_ABBR = {
    "Caretaking": "Caretake",
    "Drop off / pick up": "Drop-off",
    "Eating / cooking": "Eat",
    "Errands": "Errands",
    "Exercising": "Exercise",
    "Hobby / Leisure": "Hobby",
    "Home": "Home",
    "Medical visit": "Medical",
    "Nothing else": "Nothing",
    "Other": "Other",
    "Package pick up / drop off": "Package",
    "Restaurant": "Restaurant",
    "Resting": "Rest",
    "Self-care": "Self-care",
    "Shopping": "Shop",
    "Sleeping": "Sleep",
    "Socializing": "Social",
    "Studying": "Study",
    "Waiting": "Wait",
    "Walking the dog": "Walk dog",
    "Working": "Work",
}


def _shorten(name):
    return _ABBR.get(
        name,
        name[:6],
    )


def _radar(
    ax,
    angles,
    angles_closed,
    vals,
    act_labels,
):
    vals_r = np.clip(
        vals + OFFSET,
        0.5,
        R_MAX,
    )
    vals_r_closed = np.concatenate(
        [vals_r, [vals_r[0]]]
    )

    for dpp, r_ring in zip(
        RING_DPP,
        RING_R,
    ):
        if dpp == 0:
            ax.plot(
                _CIRC,
                [r_ring] * 361,
                color=BASE_COLOR,
                lw=1.5,
                ls="--",
                zorder=2,
            )
        else:
            ax.plot(
                _CIRC,
                [r_ring] * 361,
                color="#AAAAAA",
                lw=0.8,
                ls="-",
                zorder=2,
            )

    for dpp, r_ring in zip(
        RING_DPP,
        RING_R,
    ):
        ax.text(
            angles[0] + 0.13,
            r_ring,
            str(dpp),
            ha="left",
            va="center",
            fontsize=9,
            color="#555555",
            fontweight="bold",
            zorder=6,
        )

    ax.fill(
        angles_closed,
        vals_r_closed,
        alpha=0.20,
        color=POLY_COLOR,
        zorder=3,
    )
    ax.plot(
        angles_closed,
        vals_r_closed,
        color=POLY_COLOR,
        lw=2.2,
        zorder=4,
    )

    for ang, val, r_val in zip(
        angles,
        vals,
        vals_r,
    ):
        dot_col = (
            POS_COLOR
            if val > 0
            else NEG_COLOR
        )

        ax.plot(
            ang,
            r_val,
            "o",
            color=dot_col,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=5,
        )

    short_labels = [
        _shorten(a)
        for a in act_labels
    ]

    ax.set_thetagrids(
        np.degrees(angles),
        short_labels,
        fontsize=FS_ACT,
    )

    for label in ax.get_xticklabels():
        label.set_color("navy")

    ax.tick_params(pad=7)
    ax.set_ylim(0, R_MAX)
    ax.yaxis.set_visible(False)
    ax.spines["polar"].set_visible(False)
    ax.xaxis.grid(False)
    ax.set_facecolor("white")


def _transport(
    ax,
    mode_axes,
    mode_vals,
):
    y_pos = np.arange(
        len(mode_axes)
    )

    ax.axvline(
        0,
        color="#AAAAAA",
        lw=0.9,
        zorder=2,
    )

    for yp, val in zip(
        y_pos,
        mode_vals,
    ):
        bar_col = (
            POS_COLOR
            if val > 0
            else NEG_COLOR
        )

        ax.barh(
            yp,
            val,
            height=0.40,
            color=bar_col,
            alpha=0.90,
            zorder=3,
        )

        if abs(val) > 0.5:
            ha = (
                "left"
                if val >= 0
                else "right"
            )
            pad = (
                0.9
                if val >= 0
                else -0.9
            )

            ax.text(
                val + pad,
                yp,
                f"{val:+.0f}",
                va="center",
                ha=ha,
                fontsize=FS_VALUE,
                fontweight="bold",
                color=bar_col,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [
            f"{m} –"
            for m in mode_axes
        ],
        fontsize=FS_TICK,
        fontweight="normal",
    )

    ax.tick_params(
        left=False,
        pad=1,
    )
    ax.set_xticks([0])
    ax.set_xticklabels(
        ["0"],
        fontsize=FS_TICK - 1,
    )
    ax.set_xlim(-38, 42)
    ax.set_title(
        "Transport Mode (Δ%)",
        fontsize=FS_TITLE,
        fontweight="bold",
        color="#37474F",
        pad=8,
    )

    for spine in (
        "top",
        "right",
        "left",
    ):
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_color(
        "#AAAAAA"
    )
    ax.spines["bottom"].set_linewidth(
        0.8
    )
    ax.grid(
        axis="x",
        alpha=0.15,
        lw=0.5,
    )


def _place_mode(
    fig,
    gs_r_spec,
    gs_t_spec,
    angles,
    angles_closed,
    act_delta,
    mode_delta,
    act_axes_f,
    mode_axes,
    eigval,
    mode_num,
):
    ax_r = fig.add_subplot(
        gs_r_spec,
        projection="polar",
    )

    _radar(
        ax_r,
        angles,
        angles_closed,
        act_delta,
        act_axes_f,
    )

    ax_r.set_title(
        f"M{mode_num},  λ = {eigval:.2f}\nActivity (Δ%)",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=18,
        color="#37474F",
    )

    ax_m = fig.add_subplot(
        gs_t_spec
    )

    _transport(
        ax_m,
        mode_axes,
        mode_delta,
    )


def plot_profile_fingerprints(
    rhos,
    Phi,
    df,
    output_dir=".",
    profile_names=None,
    act_base=None,
    mode_base=None,
):
    """
    Generate the profile-eigenmode fingerprint figures used in the paper.

    Parameters
    ----------
    rhos : sequence of ndarray
        Estimated profile density matrices.

    Phi : ndarray, shape (N, D)
        RFF representation of each observed activity episode.

    df : DataFrame
        Episode-level data containing activity_new and mode_choice_new.

    act_base, mode_base : pandas Series, optional
        Dataset-wide baseline percentages. If omitted, they are computed
        directly from df.

    Notes
    -----
    For each profile, eigenmodes with eigenvalue > 0.01 are retained,
    up to five modes. Activity and mode percentages are expressed as
    deviations from their dataset-wide baselines.
    """
    output_dir = _ensure_output_dir(output_dir)
    profile_names = profile_names or [
        "Shopper / Leisure",
        "Commuter",
        "Home Anchor",
    ]

    if act_base is None:
        act_base = (
            df["activity_new"]
            .value_counts(normalize=True)
            * 100
        )

    if mode_base is None:
        mode_base = (
            df["mode_choice_new"]
            .value_counts(normalize=True)
            * 100
        )

    act_axes = sorted(
        df["activity_new"]
        .dropna()
        .unique()
        .tolist()
    )

    mode_axes = [
        "mpt",
        "walk",
        "train",
        "local_pt",
        "bike",
        "other",
    ]

    for k, rho in enumerate(rhos):
        eigvals, eigvecs = np.linalg.eigh(
            rho
        )
        order = np.argsort(
            eigvals
        )[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        n_modes = min(
            5,
            int(
                np.sum(
                    eigvals > 0.01
                )
            ),
        )

        if n_modes == 0:
            n_modes = 1

        act_delta_all = []
        mode_delta_all = []

        for m in range(n_modes):
            v = eigvecs[:, m]
            activations = (
                Phi @ v
            ) ** 2
            activations /= (
                activations.sum()
            )

            act_pct = (
                pd.Series(activations)
                .groupby(
                    df["activity_new"].values
                )
                .sum()
                * 100
            )

            mode_pct = (
                pd.Series(activations)
                .groupby(
                    df["mode_choice_new"].values
                )
                .sum()
                * 100
            )

            act_delta_all.append(
                [
                    act_pct.get(a, 0)
                    - act_base.get(a, 0)
                    for a in act_axes
                ]
            )

            mode_delta_all.append(
                [
                    mode_pct.get(mo, 0)
                    - mode_base.get(mo, 0)
                    for mo in mode_axes
                ]
            )

        act_delta_all = np.array(
            act_delta_all
        )
        mode_delta_all = np.array(
            mode_delta_all
        )

        act_filter = np.any(
            np.abs(
                act_delta_all
            ) > 2,
            axis=0,
        )

        act_axes_f = [
            a
            for a, flag
            in zip(
                act_axes,
                act_filter,
            )
            if flag
        ]

        act_delta_f = (
            act_delta_all[
                :,
                act_filter,
            ]
        )

        N = len(act_axes_f)

        if N == 0:
            print(
                f"Profile {k + 1}: no activity "
                f"deviation exceeded the 2% threshold; skipped."
            )
            continue

        angles = np.linspace(
            0,
            2 * np.pi,
            N,
            endpoint=False,
        )
        angles_closed = np.concatenate(
            [angles, [angles[0]]]
        )

        fig = plt.figure(
            figsize=(
                COL_W * n_modes,
                9,
            )
        )

        gs = GridSpec(
            2,
            n_modes,
            figure=fig,
            height_ratios=[
                3,
                1,
            ],
            hspace=0.05,
            wspace=0.35,
        )

        for m in range(n_modes):
            _place_mode(
                fig,
                gs[0, m],
                gs[1, m],
                angles,
                angles_closed,
                act_delta_f[m],
                mode_delta_all[m],
                act_axes_f,
                mode_axes,
                eigvals[m],
                m + 1,
            )

        fig.suptitle(
            f"Profile {k + 1}: {profile_names[k]}",
            fontsize=FS_SUPTITLE,
            fontweight="bold",
            y=0.95,
            color="#37474F",
        )

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=BASE_COLOR,
                lw=1.8,
                ls="--",
                label="Baseline",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=POS_COLOR,
                markersize=10,
                label="Positive Values",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=NEG_COLOR,
                markersize=10,
                label="Negative Values",
            ),
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            fontsize=FS_LEGEND,
            frameon=True,
            edgecolor="lightgray",
            bbox_to_anchor=(0.5, 0.02),
        )

        for ext in (
            "pdf",
            "png",
        ):
            fig.savefig(
                output_dir
                / f"fingerprint_profile_{k + 1}.{ext}",
                dpi=400,
                bbox_inches="tight",
            )

        plt.close(fig)

        print(
            f"Saved: fingerprint_profile_{k + 1}.pdf / .png"
        )


# ============================================================
# 5. Traveler state evolution figure
# ============================================================

STATE_PROFILE_COLORS = [
    "#2E86AB",
    "#E84855",
    "#3BB273",
]

DEFAULT_CONTEXT_LABELS = [
    "time_Afternoon",
    "time_Evening",
    "time_Midday",
    "time_Morning",
    "time_Night",
    "dow_Saturday",
    "dow_Sunday",
    "dow_Weekday",
    "temp_celsius",
    "precip_depth",
    "trip_distance",
]


def plot_state_evolution(
    rho_profiles,
    beta_np,
    alpha_f,
    eta_val,
    Phi,
    C,
    ids,
    df,
    D,
    output_dir=".",
    profile_names=None,
    hmm_states_df=None,
    selected_pid=None,
    min_trips=40,
    context_labels=None,
):
    """
    Plot traveler-level dynamic profile activation and contextual inputs.

    If HMM state assignments are supplied, an HMM state strip is shown
    above the quantum-inspired profile-activation panel.

    Parameters
    ----------
    hmm_states_df : DataFrame, optional
        Must contain participant_id and hmm_state.

    selected_pid : optional
        Participant ID to display. If omitted, the function selects one
        eligible participant using seed 42, reproducing the original
        workflow.

    min_trips : int
        Minimum requested episode count. The original plotting workflow
        prioritized participants with >200 observations, then >100.

    Notes
    -----
    The HMM state relabeling used in the paper figure is preserved:
    states 1 and 2 are swapped before display to visually align the HMM
    state ordering with the profile ordering.
    """
    output_dir = _ensure_output_dir(output_dir)

    profile_names = profile_names or [
        "Shopper / Leisure",
        "Commuter",
        "Home Anchor",
    ]
    context_labels = (
        context_labels
        or DEFAULT_CONTEXT_LABELS
    )

    K = len(rho_profiles)
    identity = np.eye(D) / D

    if selected_pid is None:
        trip_counts = (
            df.groupby(
                "participant_id"
            )
            .size()
        )

        eligible = trip_counts[
            trip_counts > 200
        ].index.tolist()

        if not eligible:
            eligible = trip_counts[
                trip_counts > 100
            ].index.tolist()

            print(
                "No participants with >200 trips; "
                "falling back to >100."
            )

        if not eligible:
            eligible = trip_counts[
                trip_counts >= min_trips
            ].index.tolist()

        if not eligible:
            raise ValueError(
                "No participant satisfies the minimum episode requirement."
            )

        rng = np.random.default_rng(
            seed=42
        )
        selected_pid = rng.choice(
            eligible
        )

        print(
            f"Selected participant: {selected_pid} "
            f"({trip_counts[selected_pid]} trips)"
        )

    mask = ids == selected_pid
    Phi_p = Phi[mask]
    C_p = C[mask]
    df_p = (
        df.loc[mask]
        .reset_index(drop=True)
    )

    n_trips = len(Phi_p)

    print(
        f"Running evolution for participant "
        f"{selected_pid} ({n_trips} trips)..."
    )

    rho_i = identity.copy()
    pi_series = []

    for i in range(n_trips):
        phi = Phi_p[i]
        c = C_p[i]

        logits = beta_np @ c
        sig = 1 / (
            1 + np.exp(-logits)
        )
        pi = sig / sig.sum()

        pi_series.append(
            pi.copy()
        )

        mixture = sum(
            pi[k]
            * rho_profiles[k]
            for k in range(K)
        )

        rho_t = (
            (1 - alpha_f)
            * rho_i
            + alpha_f
            * mixture
        )

        outer = np.outer(
            phi,
            phi,
        )

        rho_i = (
            (1 - eta_val)
            * rho_t
            + eta_val
            * outer
        )

    pi_series = np.array(
        pi_series
    )
    trip_idx = np.arange(
        n_trips
    )

    pi_smooth = (
        pi_series.copy()
    )

    # Optional HMM state sequence
    hmm_state_seq = None

    if hmm_states_df is not None:
        mask_hmm = (
            hmm_states_df[
                "participant_id"
            ]
            == selected_pid
        )

        hmm_p = (
            hmm_states_df.loc[
                mask_hmm
            ]
            .reset_index(
                drop=True
            )
        )

        hmm_state_seq = (
            hmm_p[
                "hmm_state"
            ].values
            - 1
        )

        # Preserve state relabeling used in the paper figure.
        hmm_swapped = (
            hmm_state_seq.copy()
        )
        hmm_swapped[
            hmm_state_seq == 0
        ] = 1
        hmm_swapped[
            hmm_state_seq == 1
        ] = 0
        hmm_state_seq = (
            hmm_swapped
        )

    def get_col(candidates):
        for c in candidates:
            if c in df_p.columns:
                return df_p[c].values
        return None

    temp_vals = get_col(
        [
            "temp_celsius_new",
            "temp_celsius",
        ]
    )
    prec_vals = get_col(
        [
            "precip_depth_mm_new",
            "precip_depth_mm",
            "precip_depth",
        ]
    )
    dist_vals = get_col(
        [
            "trip_distance_new",
            "trip_distance",
        ]
    )

    # This division is preserved from the original plotting code.
    if dist_vals is not None:
        dist_vals = (
            dist_vals / 1000
        )

    time_cols = [
        "time_Afternoon",
        "time_Evening",
        "time_Midday",
        "time_Morning",
        "time_Night",
    ]

    dow_cols = [
        "dow_Saturday",
        "dow_Sunday",
        "dow_Weekday",
    ]

    def decode_categorical(
        C_arr,
        labels,
        prefix,
    ):
        idxs = [
            context_labels.index(l)
            for l in labels
            if l in context_labels
        ]

        if not idxs:
            return None

        vals = C_arr[:, idxs]
        best = np.argmax(
            vals,
            axis=1,
        )

        names = [
            l.replace(
                prefix,
                "",
            )
            for l in labels
            if l in context_labels
        ]

        return [
            names[b]
            for b in best
        ]

    tod_decoded = decode_categorical(
        C_p,
        time_cols,
        "time_",
    )
    dow_decoded = decode_categorical(
        C_p,
        dow_cols,
        "dow_",
    )

    strip_specs = []

    if tod_decoded:
        strip_specs.append(
            (
                "tod",
                tod_decoded,
                "Time",
            )
        )

    if dow_decoded:
        strip_specs.append(
            (
                "dow",
                dow_decoded,
                "Day",
            )
        )

    if temp_vals is not None:
        strip_specs.append(
            (
                "cont",
                temp_vals,
                "Temp (°C)",
            )
        )

    if prec_vals is not None:
        strip_specs.append(
            (
                "cont",
                prec_vals,
                "Prec (mm)",
            )
        )

    if dist_vals is not None:
        strip_specs.append(
            (
                "cont",
                dist_vals,
                "Dist (km)",
            )
        )

    n_strips = len(
        strip_specs
    )

    if hmm_state_seq is not None:
        h_ratios = (
            [0.65, 0.7]
            + [0.65]
            * n_strips
        )
    else:
        h_ratios = (
            [0.7]
            + [0.65]
            * n_strips
        )

    fig_h = (
        3.0
        + sum(
            h_ratios[1:]
        )
        * 0.9
    )

    if hmm_state_seq is not None:
        fig_h += (
            h_ratios[0]
            * 0.9
        )

    n_rows = (
        (
            1
            if hmm_state_seq
            is not None
            else 0
        )
        + 1
        + n_strips
    )

    fig = plt.figure(
        figsize=(
            14,
            fig_h,
        )
    )

    gs = gridspec.GridSpec(
        n_rows,
        1,
        height_ratios=h_ratios,
        hspace=0.33,
        figure=fig,
    )

    row_offset = 0

    if hmm_state_seq is not None:
        ax_hmm = fig.add_subplot(
            gs[0]
        )

        state_img = np.array(
            [hmm_state_seq]
        ).astype(float)

        cmap = (
            matplotlib.colors
            .ListedColormap(
                STATE_PROFILE_COLORS
            )
        )

        ax_hmm.imshow(
            state_img,
            aspect="auto",
            cmap=cmap,
            extent=[
                0,
                n_trips - 1,
                0,
                1,
            ],
            vmin=0,
            vmax=K - 1,
            interpolation="nearest",
            alpha=0.85,
        )

        ax_hmm.set_xlim(
            0,
            n_trips - 1,
        )
        ax_hmm.set_ylim(
            0,
            1,
        )
        ax_hmm.set_ylabel(
            "HMM",
            fontsize=9.5,
            fontweight="normal",
            color="black",
            rotation=90,
            ha="center",
            va="center",
            labelpad=36,
        )
        ax_hmm.set_yticks([])
        ax_hmm.tick_params(
            axis="x",
            labelbottom=False,
            length=0,
        )
        ax_hmm.tick_params(
            axis="y",
            length=0,
        )

        ax_hmm.spines[
            "top"
        ].set_visible(False)
        ax_hmm.spines[
            "right"
        ].set_visible(False)
        ax_hmm.spines[
            "bottom"
        ].set_visible(False)
        ax_hmm.spines[
            "left"
        ].set_visible(True)
        ax_hmm.spines[
            "left"
        ].set_linewidth(0.5)
        ax_hmm.spines[
            "left"
        ].set_color("#AAAAAA")

        row_offset = 1

    ax0 = fig.add_subplot(
        gs[row_offset]
    )

    baseline = np.zeros(
        n_trips
    )

    for k in range(K):
        top = (
            baseline
            + pi_smooth[:, k]
        )

        ax0.fill_between(
            trip_idx,
            baseline,
            top,
            alpha=0.72,
            color=STATE_PROFILE_COLORS[k],
            linewidth=0,
        )

        ax0.plot(
            trip_idx,
            top,
            color="white",
            linewidth=0.8,
            alpha=0.6,
            zorder=4,
        )

        baseline = top

    ax0.plot(
        trip_idx,
        np.ones(
            n_trips
        ),
        color="#CCCCCC",
        linewidth=0.5,
        zorder=5,
    )

    ax0.set_xlim(
        0,
        n_trips - 1,
    )
    ax0.set_ylim(
        0,
        1,
    )
    ax0.set_ylabel(
        r"$\pi_k(t)$",
        fontsize=11,
        rotation=90,
        ha="center",
        va="center",
        labelpad=20,
    )

    ax0.tick_params(
        axis="x",
        labelbottom=False,
        length=0,
    )
    ax0.tick_params(
        axis="y",
        labelsize=8,
    )

    ax0.set_yticks(
        [
            0,
            0.5,
            1.0,
        ]
    )
    ax0.set_yticklabels(
        [
            "0",
            "0.5",
            "1",
        ],
        fontsize=8,
    )

    ax0.yaxis.tick_left()
    ax0.spines[
        "top"
    ].set_visible(False)
    ax0.spines[
        "right"
    ].set_visible(False)
    ax0.spines[
        "bottom"
    ].set_visible(False)

    ax0.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.5,
    )

    # State-profile correspondence used in the paper.
    state_map = {
        0: 2,
        1: 1,
        2: 3,
    }

    legend_handles = [
        mpatches.Patch(
            color=STATE_PROFILE_COLORS[k],
            alpha=0.72,
            label=(
                f"Profile {k + 1}: "
                f"{profile_names[k]} "
                f"(State {state_map[k]})"
            ),
        )
        for k in range(K)
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.94,
        ),
        fontsize=11,
        frameon=True,
        edgecolor="lightgray",
        ncol=K,
        handlelength=1.2,
    )

    for s_idx, spec in enumerate(
        strip_specs
    ):
        ax = fig.add_subplot(
            gs[
                s_idx
                + 1
                + row_offset
            ]
        )

        ax.set_xlim(
            0,
            n_trips - 1,
        )

        strip_type = spec[0]
        label = spec[2]

        if strip_type == "cont":
            vals = (
                spec[1]
                .astype(float)
            )

            finite = vals[
                np.isfinite(vals)
            ]

            vmin = (
                np.percentile(
                    finite,
                    2,
                )
                if len(finite)
                else 0
            )

            vmax = (
                np.percentile(
                    finite,
                    98,
                )
                if len(finite)
                else 1
            )

            y_low = min(
                0,
                vmin,
            )
            y_high = max(
                0,
                vmax,
            )

            ax.axhline(
                0,
                color="#AAAAAA",
                linewidth=0.4,
                zorder=2,
            )
            ax.plot(
                trip_idx,
                np.clip(
                    vals,
                    vmin,
                    vmax,
                ),
                color="black",
                linewidth=0.4,
                alpha=0.95,
                zorder=3,
            )

            ax.set_ylim(
                y_low,
                y_high,
            )
            ax.yaxis.tick_left()

            y_mid = (
                y_high / 2
            )

            ax.set_yticks(
                [0, y_mid]
            )
            ax.set_yticklabels(
                [
                    "0",
                    f"{y_mid:.0f}",
                ],
                fontsize=7,
                color="black",
            )
            ax.tick_params(
                axis="y",
                length=0,
            )

            ax.spines[
                "left"
            ].set_visible(True)
            ax.spines[
                "left"
            ].set_linewidth(0.5)
            ax.spines[
                "left"
            ].set_color("#AAAAAA")

            ax.spines[
                "bottom"
            ].set_position(
                (
                    "data",
                    0,
                )
            )

            if label in [
                "Dist (km)",
                "Prec (mm)",
            ]:
                cont_labelpad = 22
            else:
                cont_labelpad = 26

            ax.set_ylabel(
                label,
                fontsize=9.5,
                fontweight="normal",
                color="black",
                rotation=90,
                ha="center",
                va="bottom",
                labelpad=cont_labelpad,
            )

        elif strip_type == "tod":
            tod_map = {
                "Morning": 1,
                "Midday": 2,
                "Afternoon": 3,
                "Evening": 4,
                "Night": 5,
            }

            cats = spec[1]
            num_vals = np.array(
                [
                    tod_map.get(
                        c,
                        np.nan,
                    )
                    for c in cats
                ],
                dtype=float,
            )

            ax.step(
                trip_idx,
                num_vals,
                color="black",
                linewidth=0.5,
                alpha=0.95,
                where="mid",
            )

            ax.set_ylim(
                0.5,
                5.5,
            )
            ax.yaxis.tick_left()
            ax.set_yticks(
                [1, 2, 3, 4, 5]
            )
            ax.set_yticklabels(
                [
                    "Morn.",
                    "Midd.",
                    "After.",
                    "Even.",
                    "Night",
                ],
                fontsize=7,
                color="black",
            )
            ax.tick_params(
                axis="y",
                length=0,
            )

            ax.spines[
                "left"
            ].set_visible(True)
            ax.spines[
                "left"
            ].set_linewidth(0.5)
            ax.spines[
                "left"
            ].set_color("#AAAAAA")

            ax.set_ylabel(
                label,
                fontsize=9.5,
                fontweight="normal",
                color="black",
                rotation=90,
                ha="center",
                va="center",
                labelpad=18,
            )

        elif strip_type == "dow":
            dow_map = {
                "Weekday": 1,
                "Saturday": 2,
                "Sunday": 3,
            }

            cats = spec[1]
            num_vals = np.array(
                [
                    dow_map.get(
                        c,
                        np.nan,
                    )
                    for c in cats
                ],
                dtype=float,
            )

            ax.step(
                trip_idx,
                num_vals,
                color="black",
                linewidth=0.5,
                alpha=0.95,
                where="mid",
            )

            ax.set_ylim(
                0.5,
                3.5,
            )
            ax.yaxis.tick_left()
            ax.set_yticks(
                [1, 2, 3]
            )
            ax.set_yticklabels(
                [
                    "Weekday",
                    "Sat.",
                    "Sun.",
                ],
                fontsize=7,
                color="black",
            )
            ax.tick_params(
                axis="y",
                length=0,
            )

            ax.spines[
                "left"
            ].set_visible(True)
            ax.spines[
                "left"
            ].set_linewidth(0.5)
            ax.spines[
                "left"
            ].set_color("#AAAAAA")

            ax.set_ylabel(
                label,
                fontsize=9.5,
                fontweight="normal",
                color="black",
                rotation=90,
                ha="center",
                va="center",
                labelpad=8,
            )

        ax.tick_params(
            axis="x",
            labelbottom=False,
            length=0,
        )

        ax.spines[
            "top"
        ].set_visible(False)
        ax.spines[
            "right"
        ].set_visible(False)

        if strip_type == "cont":
            ax.spines[
                "bottom"
            ].set_visible(True)
            ax.spines[
                "bottom"
            ].set_color("#AAAAAA")
            ax.spines[
                "bottom"
            ].set_linewidth(0.5)

        elif (
            s_idx
            < n_strips - 1
        ):
            ax.spines[
                "bottom"
            ].set_visible(False)

        else:
            ax.spines[
                "bottom"
            ].set_color("#AAAAAA")

        if (
            s_idx
            == n_strips - 1
        ):
            ax.set_xlabel(
                "Trip Sequence",
                fontsize=11,
                labelpad=18,
            )

    fig.subplots_adjust(
        left=0.12,
        right=0.86,
    )

    filename = (
        f"evolution_{selected_pid}"
    )

    for ext in (
        "pdf",
        "png",
    ):
        fig.savefig(
            output_dir
            / f"{filename}.{ext}",
            dpi=600,
            bbox_inches="tight",
        )

    plt.close(fig)

    print(
        f"Saved: {filename}.pdf / .png"
    )

    return selected_pid


# ============================================================
# Example usage notes
# ============================================================

if __name__ == "__main__":
    print(
        "paper_figures.py contains reusable figure-generation functions.\n"
        "It is not intended to estimate the model by itself.\n\n"
        "Import the desired function after running the model or loading\n"
        "saved model outputs. See the function docstrings for required\n"
        "inputs."
    )
