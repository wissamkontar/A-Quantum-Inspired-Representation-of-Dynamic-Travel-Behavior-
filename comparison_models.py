#!/usr/bin/env python3
"""
Comparison models used in
"A Quantum-Inspired Representation of Dynamic Travel Behavior".

This script consolidates the three benchmark models used in the paper:

1. Multinomial Logit (MNL)
2. Joint Latent Class Model (panel EM)
3. Hidden Markov Model (HMM)

All three models use the same cleaned TimeUse+ input file produced by the
preprocessing pipeline:

    activities_merged4.csv

The implementation below preserves the estimation logic and specifications
used in the original analysis notebook. The code has only been reorganized
into functions and supplemented with comments and structured output files
for reproducibility.

Run
---
python comparison_models.py

Optional:
python comparison_models.py --data path/to/activities_merged4.csv
python comparison_models.py --output comparison_results

Outputs
-------
comparison_results/
    comparison_summary.csv
    mnl/
        coefficients.csv
        fit_statistics.csv
    latent_class/
        class_assignments.csv
        class_summary.csv
        fit_statistics.csv
    hmm/
        transition_matrix.csv
        state_assignments.csv
        state_summary.csv
        fit_statistics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from scipy.special import logsumexp
from scipy.stats import norm as scipy_norm
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# Shared preprocessing
# ============================================================

REQUIRED_COLUMNS = [
    "event_duration_new",
    "distance_from_home",
    "time_of_day_new",
    "day_of_week_new",
    "temp_celsius_new",
    "precip_depth_mm_new",
    "trip_distance_new",
    "activity_new",
    "mode_choice_new",
    "participant_id",
    "started_at_utc_global",
]


def time_to_period(hour):
    """Convert hour of day to the five time periods used in the paper."""
    if 6 <= hour < 10:
        return "Morning"
    elif 10 <= hour < 14:
        return "Midday"
    elif 14 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 22:
        return "Evening"
    else:
        return "Night"


def dow_to_type(dow):
    """Collapse day of week into Weekday, Saturday, and Sunday."""
    if dow == 6:
        return "Saturday"
    elif dow == 7:
        return "Sunday"
    else:
        return "Weekday"


def load_common_data(path):
    """
    Load the final preprocessed TimeUse+ file and apply the common
    filtering and temporal recoding used by all comparison models.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input file is missing required columns: "
            + ", ".join(missing)
        )

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df = (
        df.sort_values(["participant_id", "started_at_utc_global"])
        .reset_index(drop=True)
    )

    df["time_period"] = df["time_of_day_new"].apply(time_to_period)
    df["day_type"] = df["day_of_week_new"].apply(dow_to_type)

    print(f"Loaded observations : {len(df):,}")
    print(f"Participants        : {df['participant_id'].nunique():,}")

    return df


def build_context_matrix(df):
    """
    Build the common contextual covariate matrix used by the latent
    class model. Morning and Weekday are the reference categories.
    """
    time_dummies = pd.get_dummies(df["time_period"], prefix="time")
    dow_dummies = pd.get_dummies(df["day_type"], prefix="dow")

    time_dummies = time_dummies.drop(columns=["time_Morning"])
    dow_dummies = dow_dummies.drop(columns=["dow_Weekday"])

    X = pd.concat(
        [
            time_dummies,
            dow_dummies,
            df[
                [
                    "temp_celsius_new",
                    "precip_depth_mm_new",
                    "trip_distance_new",
                ]
            ],
        ],
        axis=1,
    )

    cont_cols = [
        "temp_celsius_new",
        "precip_depth_mm_new",
        "trip_distance_new",
    ]

    scaler = StandardScaler()
    X[cont_cols] = scaler.fit_transform(X[cont_cols])
    X = sm.add_constant(X)

    return X, scaler


# ============================================================
# 1. Multinomial Logit Model
# ============================================================

def clustered_se_mnlogit(result, groups):
    """
    Cluster-robust covariance matrix by participant.

    This reproduces the participant-clustered standard errors used
    in the original MNL comparison.
    """
    score = result.model.score_obs(
        np.array(result.params).flatten()
    )

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_obs = len(groups)
    n_params = score.shape[1]

    B = np.zeros((n_params, n_params))

    for group in unique_groups:
        mask = groups == group
        sg = score[mask].sum(axis=0)
        B += np.outer(sg, sg)

    H_inv = np.array(result.cov_params())

    correction = (
        (n_groups / (n_groups - 1))
        * ((n_obs - 1) / (n_obs - n_params))
    )

    V_cluster = correction * H_inv @ B @ H_inv

    return V_cluster, n_groups


def fit_mnl(df, output_dir):
    """
    Fit the three-outcome MNL used in the manuscript comparison.

    Outcomes:
        Home Anchor = reference
        Commuter
        Shopper

    Only episodes classified as Home, Working, or Shopping enter
    this benchmark model.
    """
    print("\n" + "=" * 72)
    print("MULTINOMIAL LOGIT MODEL")
    print("=" * 72)

    def assign_group(activity):
        if activity == "Working":
            return "Commuter"
        elif activity == "Shopping":
            return "Shopper"
        elif activity == "Home":
            return "Home Anchor"
        return None

    df_mnl = df.copy()
    df_mnl["group"] = df_mnl["activity_new"].apply(assign_group)
    df_mnl = (
        df_mnl[df_mnl["group"].notna()]
        .copy()
        .reset_index(drop=True)
    )

    print(f"Observations : {len(df_mnl):,}")
    print(f"Participants : {df_mnl['participant_id'].nunique():,}")
    print(df_mnl["group"].value_counts())

    # Context and mode predictors
    time_dummies = pd.get_dummies(
        df_mnl["time_period"],
        prefix="time",
    )
    dow_dummies = pd.get_dummies(
        df_mnl["day_type"],
        prefix="dow",
    )
    mode_dummies = pd.get_dummies(
        df_mnl["mode_choice_new"].fillna("other"),
        prefix="mode",
    )

    ref_time = "time_Morning"
    ref_dow = "dow_Weekday"
    ref_mode = mode_dummies.columns[0]

    time_dummies = time_dummies.drop(columns=[ref_time])
    dow_dummies = dow_dummies.drop(columns=[ref_dow])
    mode_dummies = mode_dummies.drop(columns=[ref_mode])

    X = pd.concat(
        [
            time_dummies,
            dow_dummies,
            mode_dummies,
            df_mnl[
                [
                    "temp_celsius_new",
                    "precip_depth_mm_new",
                    "trip_distance_new",
                ]
            ],
        ],
        axis=1,
    )

    cont_cols = [
        "temp_celsius_new",
        "precip_depth_mm_new",
        "trip_distance_new",
    ]

    scaler = StandardScaler()
    X[cont_cols] = scaler.fit_transform(X[cont_cols])
    X = sm.add_constant(X)

    # Explicit response ordering used in the paper.
    df_mnl["group_code"] = df_mnl["group"].map(
        {
            "Home Anchor": 0,
            "Commuter": 1,
            "Shopper": 2,
        }
    )
    y = df_mnl["group_code"].values

    model = sm.MNLogit(y, X.astype(float))
    result = model.fit(
        method="bfgs",
        maxiter=2000,
        disp=True,
    )

    groups = df_mnl["participant_id"].values
    V_cluster, n_groups = clustered_se_mnlogit(
        result,
        groups,
    )
    clustered_se = np.sqrt(np.diag(V_cluster))

    n_vars = len(X.columns)
    equation_names = ["Commuter", "Shopper"]
    coefficient_tables = []

    for eq_idx, eq_name in enumerate(equation_names):
        start = eq_idx * n_vars
        end = start + n_vars

        coefs = result.params.iloc[:, eq_idx].values
        se_cl = clustered_se[start:end]
        z_cl = coefs / se_cl
        p_cl = 2 * (
            1 - stats.norm.cdf(np.abs(z_cl))
        )
        ci_low = coefs - 1.96 * se_cl
        ci_high = coefs + 1.96 * se_cl

        table = pd.DataFrame(
            {
                "equation": eq_name,
                "variable": X.columns,
                "coef": coefs,
                "cluster_se": se_cl,
                "z": z_cl,
                "p_value": p_cl,
                "ci_2.5": ci_low,
                "ci_97.5": ci_high,
            }
        )

        coefficient_tables.append(table)

        print(f"\nEquation: {eq_name} vs. Home Anchor")
        print(
            table.set_index("variable")[
                ["coef", "cluster_se", "z", "p_value"]
            ].round(4).to_string()
        )

    coefficients = pd.concat(
        coefficient_tables,
        ignore_index=True,
    )

    fit = pd.DataFrame(
        [
            {
                "model": "MNL",
                "log_likelihood": result.llf,
                "null_log_likelihood": result.llnull,
                "mcfadden_r2": result.prsquared,
                "aic": result.aic,
                "bic": result.bic,
                "observations": int(result.nobs),
                "participants": int(n_groups),
                "reference_outcome": "Home Anchor",
                "reference_time": "Morning",
                "reference_day": "Weekday",
                "reference_mode": ref_mode,
            }
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(
        output_dir / "coefficients.csv",
        index=False,
    )
    fit.to_csv(
        output_dir / "fit_statistics.csv",
        index=False,
    )

    print("\nGoodness of Fit")
    print(f"Log-Likelihood : {result.llf:.4f}")
    print(f"Null LL        : {result.llnull:.4f}")
    print(f"McFadden R2    : {result.prsquared:.4f}")
    print(f"AIC            : {result.aic:.4f}")
    print(f"BIC            : {result.bic:.4f}")

    return fit.iloc[0].to_dict()


# ============================================================
# 2. Joint Latent Class Model
# ============================================================

def mnl_probs(X, beta, n_alts):
    """
    Multinomial-logit probabilities with alternative 0 as reference.
    """
    n_obs = X.shape[0]
    V = np.zeros((n_obs, n_alts))

    for j in range(1, n_alts):
        V[:, j] = X @ beta[j - 1]

    V_max = V.max(axis=1, keepdims=True)
    eV = np.exp(V - V_max)

    return eV / eV.sum(axis=1, keepdims=True)


def mnl_loglik_weighted(
    beta_flat,
    X,
    y,
    weights,
    n_alts,
):
    beta = beta_flat.reshape(
        n_alts - 1,
        X.shape[1],
    )
    probs = mnl_probs(
        X,
        beta,
        n_alts,
    )
    ll = np.log(
        probs[np.arange(len(y)), y]
        + 1e-300
    )

    return -(weights * ll).sum()


def mnl_loglik_weighted_grad(
    beta_flat,
    X,
    y,
    weights,
    n_alts,
):
    beta = beta_flat.reshape(
        n_alts - 1,
        X.shape[1],
    )
    probs = mnl_probs(
        X,
        beta,
        n_alts,
    )

    grad = np.zeros_like(beta)

    for j in range(1, n_alts):
        indicator = (y == j).astype(float)
        grad[j - 1] = -(
            (
                weights
                * (indicator - probs[:, j])
            )
            @ X
        )

    return grad.flatten()


def fit_latent_class(
    df,
    output_dir,
    K=3,
    seed=42,
):
    """
    Fit the joint panel latent class model.

    Each participant belongs to one latent class for the full
    observation period. Conditional on class, the joint episode
    likelihood combines:

        activity choice  : multinomial logit
        transport mode   : multinomial logit
        activity duration: normal density

    Class posteriors are computed at the participant level.
    """
    print("\n" + "=" * 72)
    print("JOINT LATENT CLASS MODEL")
    print("=" * 72)

    df_lc = df.copy()

    le_act = LabelEncoder()
    le_mode = LabelEncoder()

    df_lc["act_code"] = le_act.fit_transform(
        df_lc["activity_new"].fillna("Other")
    )
    df_lc["mode_code"] = le_mode.fit_transform(
        df_lc["mode_choice_new"].fillna("other")
    )

    n_acts = len(le_act.classes_)
    n_modes = len(le_mode.classes_)

    X_lc, _ = build_context_matrix(df_lc)
    X_np = X_lc.values.astype(float)
    n_vars = X_np.shape[1]

    y_act = df_lc["act_code"].values.astype(int)
    y_mode = df_lc["mode_code"].values.astype(int)
    y_dur = df_lc["event_duration_new"].values.astype(float)

    n_obs = len(y_act)

    persons = df_lc["participant_id"].values
    unique_persons = np.unique(persons)
    n_persons = len(unique_persons)

    # Precompute observation indices for each participant.
    person_indices = [
        np.where(persons == pid)[0]
        for pid in unique_persons
    ]

    np.random.seed(seed)

    pi = np.ones(K) / K

    betas_act = [
        np.random.randn(
            (n_acts - 1) * n_vars
        )
        * 0.01
        for _ in range(K)
    ]

    betas_mode = [
        np.random.randn(
            (n_modes - 1) * n_vars
        )
        * 0.01
        for _ in range(K)
    ]

    dur_mu = np.array(
        [y_dur.mean()] * K
    )
    dur_logstd = np.array(
        [np.log(y_dur.std())] * K
    )

    print(
        f"Fitting panel latent class model | K={K}"
    )
    print(f"Activity alternatives : {n_acts}")
    print(f"Mode alternatives     : {n_modes}")
    print(f"Observations          : {n_obs:,}")
    print(f"Participants          : {n_persons:,}")

    max_iter = 100
    tol = 1e-5
    prev_ll = -np.inf

    for iteration in range(max_iter):
        # -------------------------
        # E-step
        # -------------------------
        log_post_person = np.zeros(
            (n_persons, K)
        )

        for k in range(K):
            beta_act_k = betas_act[k].reshape(
                n_acts - 1,
                n_vars,
            )
            probs_act_k = mnl_probs(
                X_np,
                beta_act_k,
                n_acts,
            )
            ll_act_k = np.log(
                probs_act_k[
                    np.arange(n_obs),
                    y_act,
                ]
                + 1e-300
            )

            beta_mode_k = betas_mode[k].reshape(
                n_modes - 1,
                n_vars,
            )
            probs_mode_k = mnl_probs(
                X_np,
                beta_mode_k,
                n_modes,
            )
            ll_mode_k = np.log(
                probs_mode_k[
                    np.arange(n_obs),
                    y_mode,
                ]
                + 1e-300
            )

            dur_std_k = np.exp(
                dur_logstd[k]
            )
            ll_dur_k = scipy_norm.logpdf(
                y_dur,
                loc=dur_mu[k],
                scale=dur_std_k,
            )

            ll_total_k = (
                ll_act_k
                + ll_mode_k
                + ll_dur_k
            )

            for p_idx, indices in enumerate(
                person_indices
            ):
                log_post_person[p_idx, k] = (
                    np.log(pi[k] + 1e-300)
                    + ll_total_k[indices].sum()
                )

        log_post_norm = (
            log_post_person
            - logsumexp(
                log_post_person,
                axis=1,
                keepdims=True,
            )
        )

        post_person = np.exp(
            log_post_norm
        )

        post_obs = np.zeros(
            (n_obs, K)
        )

        for p_idx, indices in enumerate(
            person_indices
        ):
            post_obs[indices] = (
                post_person[p_idx]
            )

        ll = logsumexp(
            log_post_person,
            axis=1,
        ).sum()

        if iteration % 5 == 0:
            print(
                f"  Iter {iteration:3d} | "
                f"LL = {ll:.2f} | "
                f"pi = {pi.round(3)}"
            )

        if abs(ll - prev_ll) < tol:
            print(
                f"  Converged at iteration "
                f"{iteration}"
            )
            break

        prev_ll = ll

        # -------------------------
        # M-step
        # -------------------------
        pi = post_person.mean(axis=0)
        pi = np.clip(pi, 1e-6, 1)
        pi /= pi.sum()

        for k in range(K):
            wk = post_obs[:, k]

            res_act = optimize.minimize(
                fun=mnl_loglik_weighted,
                x0=betas_act[k],
                args=(
                    X_np,
                    y_act,
                    wk,
                    n_acts,
                ),
                jac=mnl_loglik_weighted_grad,
                method="L-BFGS-B",
                options={
                    "maxiter": 100,
                    "ftol": 1e-8,
                },
            )
            betas_act[k] = res_act.x

            res_mode = optimize.minimize(
                fun=mnl_loglik_weighted,
                x0=betas_mode[k],
                args=(
                    X_np,
                    y_mode,
                    wk,
                    n_modes,
                ),
                jac=mnl_loglik_weighted_grad,
                method="L-BFGS-B",
                options={
                    "maxiter": 100,
                    "ftol": 1e-8,
                },
            )
            betas_mode[k] = res_mode.x

            wk_sum = wk.sum()

            dur_mu[k] = (
                (wk * y_dur).sum()
                / wk_sum
            )

            dur_var_k = (
                (
                    wk
                    * (
                        y_dur
                        - dur_mu[k]
                    ) ** 2
                ).sum()
                / wk_sum
            )

            dur_logstd[k] = np.log(
                np.sqrt(dur_var_k)
                + 1e-6
            )

    # -------------------------
    # Final likelihood
    # -------------------------
    log_post_final = np.zeros(
        (n_persons, K)
    )

    for k in range(K):
        beta_act_k = betas_act[k].reshape(
            n_acts - 1,
            n_vars,
        )
        probs_act_k = mnl_probs(
            X_np,
            beta_act_k,
            n_acts,
        )
        ll_act_k = np.log(
            probs_act_k[
                np.arange(n_obs),
                y_act,
            ]
            + 1e-300
        )

        beta_mode_k = betas_mode[k].reshape(
            n_modes - 1,
            n_vars,
        )
        probs_mode_k = mnl_probs(
            X_np,
            beta_mode_k,
            n_modes,
        )
        ll_mode_k = np.log(
            probs_mode_k[
                np.arange(n_obs),
                y_mode,
            ]
            + 1e-300
        )

        dur_std_k = np.exp(
            dur_logstd[k]
        )
        ll_dur_k = scipy_norm.logpdf(
            y_dur,
            loc=dur_mu[k],
            scale=dur_std_k,
        )

        ll_total_k = (
            ll_act_k
            + ll_mode_k
            + ll_dur_k
        )

        for p_idx, indices in enumerate(
            person_indices
        ):
            log_post_final[p_idx, k] = (
                np.log(pi[k] + 1e-300)
                + ll_total_k[indices].sum()
            )

    ll_final = logsumexp(
        log_post_final,
        axis=1,
    ).sum()

    n_params = (
        K
        * (
            (n_acts - 1) * n_vars
            + (n_modes - 1) * n_vars
            + 2
        )
        + (K - 1)
    )

    aic = (
        -2 * ll_final
        + 2 * n_params
    )
    bic = (
        -2 * ll_final
        + n_params * np.log(n_persons)
    )

    post_final = np.exp(
        log_post_final
        - logsumexp(
            log_post_final,
            axis=1,
            keepdims=True,
        )
    )

    assigned_class = np.argmax(
        post_final,
        axis=1,
    )

    assignment_data = {
        "participant_id": unique_persons,
        "assigned_class": assigned_class + 1,
    }

    for k in range(K):
        assignment_data[
            f"post_class{k + 1}"
        ] = post_final[:, k]

    assignments = pd.DataFrame(
        assignment_data
    )

    class_rows = []

    for k in range(K):
        n_k = (
            assigned_class == k
        ).sum()

        beta_act_k = betas_act[k].reshape(
            n_acts - 1,
            n_vars,
        )
        probs_act_k = mnl_probs(
            X_np,
            beta_act_k,
            n_acts,
        )
        mean_act = probs_act_k.mean(axis=0)

        beta_mode_k = betas_mode[k].reshape(
            n_modes - 1,
            n_vars,
        )
        probs_mode_k = mnl_probs(
            X_np,
            beta_mode_k,
            n_modes,
        )
        mean_mode = probs_mode_k.mean(axis=0)

        top_acts = np.argsort(
            mean_act
        )[::-1][:5]
        top_modes = np.argsort(
            mean_mode
        )[::-1][:3]

        class_rows.append(
            {
                "class": k + 1,
                "class_probability": pi[k],
                "assigned_persons": n_k,
                "assigned_percent": (
                    100 * n_k / n_persons
                ),
                "duration_mean": dur_mu[k],
                "duration_std": np.exp(
                    dur_logstd[k]
                ),
                "top_activities": "; ".join(
                    f"{le_act.classes_[i]} "
                    f"({mean_act[i] * 100:.1f}%)"
                    for i in top_acts
                ),
                "top_modes": "; ".join(
                    f"{le_mode.classes_[i]} "
                    f"({mean_mode[i] * 100:.1f}%)"
                    for i in top_modes
                ),
            }
        )

    class_summary = pd.DataFrame(
        class_rows
    )

    fit = pd.DataFrame(
        [
            {
                "model": "Latent Class",
                "log_likelihood": ll_final,
                "aic": aic,
                "bic": bic,
                "observations": n_obs,
                "participants": n_persons,
                "parameters": n_params,
                "classes": K,
            }
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    assignments.to_csv(
        output_dir / "class_assignments.csv",
        index=False,
    )
    class_summary.to_csv(
        output_dir / "class_summary.csv",
        index=False,
    )
    fit.to_csv(
        output_dir / "fit_statistics.csv",
        index=False,
    )

    print("\nGoodness of Fit")
    print(f"Log-Likelihood : {ll_final:.4f}")
    print(f"AIC            : {aic:.4f}")
    print(f"BIC            : {bic:.4f}")

    print("\nClass Summary")
    print(
        class_summary.round(4).to_string(
            index=False
        )
    )

    return fit.iloc[0].to_dict()


# ============================================================
# 3. Hidden Markov Model
# ============================================================

def fit_hmm(
    df,
    output_dir,
    K=3,
    seed=42,
):
    """
    Fit the three-state HMM used in the manuscript comparison.

    Hidden states evolve through a Markov transition matrix. Conditional
    on the current hidden state, activity, transport mode, and duration
    are modeled jointly as independent emission components:

        activity choice   : categorical
        transport mode    : categorical
        activity duration : normal

    Estimation uses the Baum-Welch EM algorithm, matching the original
    comparison notebook.
    """
    print("\n" + "=" * 72)
    print("HIDDEN MARKOV MODEL")
    print("=" * 72)

    df_hmm = df.copy()

    le_act = LabelEncoder()
    le_mode = LabelEncoder()

    df_hmm["act_code"] = le_act.fit_transform(
        df_hmm["activity_new"].fillna("Other")
    )
    df_hmm["mode_code"] = le_mode.fit_transform(
        df_hmm["mode_choice_new"].fillna("other")
    )

    n_acts = len(le_act.classes_)
    n_modes = len(le_mode.classes_)

    y_act = df_hmm["act_code"].values.astype(int)
    y_mode = df_hmm["mode_code"].values.astype(int)
    y_dur = df_hmm["event_duration_new"].values.astype(float)

    persons = df_hmm["participant_id"].values
    unique_persons = np.unique(persons)
    n_persons = len(unique_persons)

    person_indices = [
        np.where(persons == pid)[0]
        for pid in unique_persons
    ]

    np.random.seed(seed)

    log_pi = np.log(
        np.ones(K) / K
    )

    # High diagonal values initialize the model with behavioral persistence.
    A = np.ones((K, K)) * 0.1
    np.fill_diagonal(A, 0.8)
    A = A / A.sum(
        axis=1,
        keepdims=True,
    )
    log_A = np.log(
        A + 1e-300
    )

    emit_act = [
        np.ones(n_acts) / n_acts
        for _ in range(K)
    ]
    emit_mode = [
        np.ones(n_modes) / n_modes
        for _ in range(K)
    ]

    emit_dur_mu = (
        np.array(
            [y_dur.mean()] * K
        )
        + np.random.randn(K) * 10
    )
    emit_dur_logstd = np.array(
        [np.log(y_dur.std())] * K
    )

    def emission_logprob(
        k,
        obs_indices,
    ):
        """Joint log emission probability for one hidden state."""
        ll_act = np.log(
            emit_act[k][
                y_act[obs_indices]
            ]
            + 1e-300
        )
        ll_mode = np.log(
            emit_mode[k][
                y_mode[obs_indices]
            ]
            + 1e-300
        )
        ll_dur = scipy_norm.logpdf(
            y_dur[obs_indices],
            loc=emit_dur_mu[k],
            scale=np.exp(
                emit_dur_logstd[k]
            ),
        )

        return (
            ll_act
            + ll_mode
            + ll_dur
        )

    print(f"Fitting HMM | K={K}")
    print(f"Observations : {len(df_hmm):,}")
    print(f"Participants : {n_persons:,}")

    max_iter = 50
    tol = 1e-4
    prev_ll = -np.inf
    total_ll = -np.inf

    for iteration in range(max_iter):
        gamma_sum = np.zeros(K)
        xi_sum = np.zeros((K, K))

        emit_act_num = [
            np.zeros(n_acts)
            for _ in range(K)
        ]
        emit_mode_num = [
            np.zeros(n_modes)
            for _ in range(K)
        ]

        emit_dur_num_mu = np.zeros(K)
        emit_dur_num_s2 = np.zeros(K)

        total_ll = 0.0

        for indices in person_indices:
            T = len(indices)

            if T < 2:
                continue

            # -------------------------
            # Forward pass
            # -------------------------
            log_alpha = np.zeros(
                (T, K)
            )

            for k in range(K):
                log_alpha[0, k] = (
                    log_pi[k]
                    + emission_logprob(
                        k,
                        indices[[0]],
                    )
                )

            for t in range(1, T):
                for k in range(K):
                    log_alpha[t, k] = (
                        logsumexp(
                            log_alpha[t - 1]
                            + log_A[:, k]
                        )
                        + emission_logprob(
                            k,
                            indices[[t]],
                        )
                    )

            ll_seq = logsumexp(
                log_alpha[-1]
            )
            total_ll += ll_seq

            # -------------------------
            # Backward pass
            # -------------------------
            log_beta = np.zeros(
                (T, K)
            )

            for t in range(
                T - 2,
                -1,
                -1,
            ):
                for k in range(K):
                    log_beta[t, k] = (
                        logsumexp(
                            log_A[k]
                            + np.array(
                                [
                                    emission_logprob(
                                        j,
                                        indices[[t + 1]],
                                    )
                                    for j in range(K)
                                ]
                            )
                            + log_beta[t + 1]
                        )
                    )

            # -------------------------
            # State posteriors
            # -------------------------
            log_gamma = (
                log_alpha
                + log_beta
            )
            log_gamma -= logsumexp(
                log_gamma,
                axis=1,
                keepdims=True,
            )
            gamma = np.exp(
                log_gamma
            )

            # -------------------------
            # Transition posteriors
            # -------------------------
            for t in range(T - 1):
                log_xi = np.zeros(
                    (K, K)
                )

                for i in range(K):
                    for j in range(K):
                        log_xi[i, j] = (
                            log_alpha[t, i]
                            + log_A[i, j]
                            + emission_logprob(
                                j,
                                indices[[t + 1]],
                            )
                            + log_beta[t + 1, j]
                        )

                log_xi -= logsumexp(
                    log_xi.flatten()
                )
                xi_sum += np.exp(
                    log_xi
                )

            # -------------------------
            # Sufficient statistics
            # -------------------------
            gamma_sum += gamma.sum(
                axis=0
            )

            for k in range(K):
                for t in range(T):
                    emit_act_num[k][
                        y_act[indices[t]]
                    ] += gamma[t, k]

                    emit_mode_num[k][
                        y_mode[indices[t]]
                    ] += gamma[t, k]

                emit_dur_num_mu[k] += (
                    gamma[:, k]
                    * y_dur[indices]
                ).sum()

                emit_dur_num_s2[k] += (
                    gamma[:, k]
                    * (
                        y_dur[indices]
                        - emit_dur_mu[k]
                    ) ** 2
                ).sum()

        if iteration % 5 == 0:
            print(
                f"  Iter {iteration:3d} | "
                f"LL = {total_ll:.2f}"
            )

        if abs(
            total_ll - prev_ll
        ) < tol:
            print(
                f"  Converged at iteration "
                f"{iteration}"
            )
            break

        prev_ll = total_ll

        # -------------------------
        # M-step
        # -------------------------
        # This update follows the original analysis notebook.
        log_pi = np.log(
            gamma_sum / gamma_sum.sum()
            + 1e-300
        )

        A = xi_sum / xi_sum.sum(
            axis=1,
            keepdims=True,
        )
        A = np.clip(
            A,
            1e-6,
            1,
        )
        A /= A.sum(
            axis=1,
            keepdims=True,
        )
        log_A = np.log(A)

        for k in range(K):
            emit_act[k] = (
                emit_act_num[k]
                / (
                    emit_act_num[k].sum()
                    + 1e-300
                )
            )

            emit_mode[k] = (
                emit_mode_num[k]
                / (
                    emit_mode_num[k].sum()
                    + 1e-300
                )
            )

            emit_dur_mu[k] = (
                emit_dur_num_mu[k]
                / gamma_sum[k]
            )

            emit_dur_logstd[k] = np.log(
                np.sqrt(
                    emit_dur_num_s2[k]
                    / gamma_sum[k]
                )
                + 1e-6
            )

    n_params = (
        (K - 1)
        + K * (K - 1)
        + K * (n_acts - 1)
        + K * (n_modes - 1)
        + K * 2
    )

    aic = (
        -2 * total_ll
        + 2 * n_params
    )
    bic = (
        -2 * total_ll
        + n_params * np.log(n_persons)
    )

    pi_final = np.exp(log_pi)
    pi_final /= pi_final.sum()

    transition_df = pd.DataFrame(
        A,
        index=[
            f"State {k + 1}"
            for k in range(K)
        ],
        columns=[
            f"State {k + 1}"
            for k in range(K)
        ],
    )

    state_rows = []

    for k in range(K):
        top_acts = np.argsort(
            emit_act[k]
        )[::-1][:5]
        top_modes = np.argsort(
            emit_mode[k]
        )[::-1][:3]

        state_rows.append(
            {
                "state": k + 1,
                "initial_probability": pi_final[k],
                "duration_mean": emit_dur_mu[k],
                "duration_std": np.exp(
                    emit_dur_logstd[k]
                ),
                "self_transition": A[k, k],
                "top_activities": "; ".join(
                    f"{le_act.classes_[i]} "
                    f"({emit_act[k][i] * 100:.1f}%)"
                    for i in top_acts
                ),
                "top_modes": "; ".join(
                    f"{le_mode.classes_[i]} "
                    f"({emit_mode[k][i] * 100:.1f}%)"
                    for i in top_modes
                ),
            }
        )

    state_summary = pd.DataFrame(
        state_rows
    )

    # -------------------------
    # Viterbi decoding
    # -------------------------
    state_records = []

    for pid, indices in zip(
        unique_persons,
        person_indices,
    ):
        T = len(indices)

        if T < 2:
            continue

        log_delta = np.zeros(
            (T, K)
        )
        psi = np.zeros(
            (T, K),
            dtype=int,
        )

        for k in range(K):
            log_delta[0, k] = (
                log_pi[k]
                + emission_logprob(
                    k,
                    indices[[0]],
                )
            )

        for t in range(1, T):
            for k in range(K):
                trans_scores = (
                    log_delta[t - 1]
                    + log_A[:, k]
                )

                psi[t, k] = np.argmax(
                    trans_scores
                )

                log_delta[t, k] = (
                    trans_scores[
                        psi[t, k]
                    ]
                    + emission_logprob(
                        k,
                        indices[[t]],
                    )
                )

        state_seq = np.zeros(
            T,
            dtype=int,
        )
        state_seq[-1] = np.argmax(
            log_delta[-1]
        )

        for t in range(
            T - 2,
            -1,
            -1,
        ):
            state_seq[t] = psi[
                t + 1,
                state_seq[t + 1],
            ]

        for t, idx in enumerate(indices):
            state_records.append(
                {
                    "participant_id": pid,
                    "episode_idx": idx,
                    "hmm_state": (
                        state_seq[t] + 1
                    ),
                    "activity": (
                        le_act.classes_[
                            y_act[idx]
                        ]
                    ),
                    "mode": (
                        le_mode.classes_[
                            y_mode[idx]
                        ]
                    ),
                }
            )

    state_assignments = pd.DataFrame(
        state_records
    )

    fit = pd.DataFrame(
        [
            {
                "model": "HMM",
                "log_likelihood": total_ll,
                "aic": aic,
                "bic": bic,
                "observations": len(df_hmm),
                "participants": n_persons,
                "parameters": n_params,
                "states": K,
            }
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    transition_df.to_csv(
        output_dir / "transition_matrix.csv"
    )
    state_assignments.to_csv(
        output_dir / "state_assignments.csv",
        index=False,
    )
    state_summary.to_csv(
        output_dir / "state_summary.csv",
        index=False,
    )
    fit.to_csv(
        output_dir / "fit_statistics.csv",
        index=False,
    )

    print("\nGoodness of Fit")
    print(f"Log-Likelihood : {total_ll:.4f}")
    print(f"AIC            : {aic:.4f}")
    print(f"BIC            : {bic:.4f}")

    print("\nTransition Matrix")
    print(
        transition_df.round(4).to_string()
    )

    print("\nState Summary")
    print(
        state_summary.round(4).to_string(
            index=False
        )
    )

    return fit.iloc[0].to_dict()


# ============================================================
# Main
# ============================================================

def run_all_models(
    data_path,
    output_root,
):
    """
    Estimate all three comparison models and save a compact summary.
    """
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_common_data(
        data_path
    )

    results = []

    results.append(
        fit_mnl(
            df,
            output_root / "mnl",
        )
    )

    results.append(
        fit_latent_class(
            df,
            output_root / "latent_class",
            K=3,
            seed=42,
        )
    )

    results.append(
        fit_hmm(
            df,
            output_root / "hmm",
            K=3,
            seed=42,
        )
    )

    summary = pd.DataFrame(
        results
    )

    # Columns differ slightly because the three models have different
    # structures. Pandas leaves non-applicable fields blank.
    summary.to_csv(
        output_root / "comparison_summary.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("COMPARISON MODELS COMPLETE")
    print("=" * 72)

    display_cols = [
        c
        for c in [
            "model",
            "log_likelihood",
            "aic",
            "bic",
            "observations",
            "participants",
        ]
        if c in summary.columns
    ]

    print(
        summary[display_cols]
        .round(4)
        .to_string(index=False)
    )

    print(
        f"\nResults saved to: "
        f"{output_root}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the MNL, joint latent class, and HMM "
            "comparison models used in the paper."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "activities_merged4.csv"
        ),
        help=(
            "Path to the final preprocessed TimeUse+ CSV."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "comparison_results"
        ),
        help=(
            "Directory in which model outputs are saved."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_all_models(
        data_path=args.data,
        output_root=args.output,
    )


if __name__ == "__main__":
    main()
