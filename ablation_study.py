#!/usr/bin/env python3
"""
Ablation study for the quantum-inspired traveler behavioral profiling model.

This script consolidates the three ablation experiments used in the paper:

1. No context activation:
       alpha = 0
2. No behavioral adaptation / state update:
       eta = 0
3. No sub-pattern structure:
       profile rank r = 1

Optionally, the full model can also be estimated in the same run so that
Delta NLL values relative to the full specification can be computed.

Input
-----
activities_merged4.csv

Main outputs
------------
ablation_results/
    full/                     (if --include-full is used)
    no_context_activation/
    no_state_evolution/
    rank1/
    ablation_summary.csv

Run
---
Run the three ablations only:

    python ablation_study.py

Run the full model plus all three ablations:

    python ablation_study.py --include-full

Specify a different input file:

    python ablation_study.py --data path/to/activities_merged4.csv --include-full
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler


# ============================================================
# Utility
# ============================================================

def log(msg):
    print(msg, flush=True)


# ============================================================
# 1. Data Loading and Preprocessing
# ============================================================

def load_data(path):
    df = pd.read_csv(path)

    def time_to_period(hour):
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
        if dow == 6:
            return "Saturday"
        elif dow == 7:
            return "Sunday"
        else:
            return "Weekday"

    df["time_period"] = df["time_of_day_new"].apply(time_to_period)
    df["day_type"] = df["day_of_week_new"].apply(dow_to_type)

    required = [
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

    df = df.dropna(subset=required).copy()
    log(f"Rows after dropping NaN: {len(df):,}")

    df = (
        df.sort_values(["participant_id", "started_at_utc_global"])
        .reset_index(drop=True)
    )

    act_dummies = pd.get_dummies(
        df["activity_new"].fillna("Other").astype(str),
        prefix="act",
    )
    mode_dummies = pd.get_dummies(
        df["mode_choice_new"].fillna("other").astype(str),
        prefix="mode",
    )

    log(f"Activity categories: {list(act_dummies.columns)}")
    log(f"Mode categories:     {list(mode_dummies.columns)}")

    behavior_df = pd.concat(
        [
            act_dummies,
            mode_dummies,
            df[["event_duration_new"]],
        ],
        axis=1,
    )
    behavior_cols = list(behavior_df.columns)

    time_dummies = pd.get_dummies(
        df["time_period"],
        prefix="time",
    )
    dow_dummies = pd.get_dummies(
        df["day_type"],
        prefix="dow",
    )

    context_df = pd.concat(
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
    context_cols = list(context_df.columns)

    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(behavior_df.values)

    scaler_c = StandardScaler()
    C = scaler_c.fit_transform(context_df.values)

    ids = df["participant_id"].values
    times = df["started_at_utc_global"].values

    log(f"Loaded {len(df):,} activity episodes")
    log(
        f"  d={X.shape[1]} state variables | "
        f"q={C.shape[1]} context variables"
    )
    log(
        f"  {df['participant_id'].nunique():,} unique participants"
    )

    return (
        X,
        C,
        ids,
        times,
        scaler_x,
        scaler_c,
        behavior_cols,
        context_cols,
        df,
    )


# ============================================================
# 2. Random Fourier Feature Map
# ============================================================

def compute_rff(X, D=200, gamma=1.0):
    rff = RBFSampler(
        gamma=gamma,
        n_components=D,
        random_state=42,
    )
    Phi = rff.fit_transform(X)

    norms = np.linalg.norm(Phi, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    Phi /= norms

    log(f"RFF dimension: D={Phi.shape[1]}")
    return Phi, rff


# ============================================================
# 3. Model Definition
# ============================================================

class QuantumTravelerModel(nn.Module):
    """
    Quantum-inspired traveler behavioral profiling model.

    Ablations are controlled explicitly through:
      - alpha_fixed
      - eta_fixed
      - profile rank

    Setting alpha_fixed=0 removes context activation.
    Setting eta_fixed=0 removes behavioral adaptation.
    Setting rank=1 removes within-profile sub-pattern structure.
    """

    def __init__(
        self,
        K,
        D,
        q,
        rank=10,
        alpha_fixed=0.3,
        eta_fixed=0.1,
    ):
        super().__init__()

        self.K = K
        self.D = D
        self.rank = rank
        self.alpha_fixed = float(alpha_fixed)
        self.eta_fixed = float(eta_fixed)

        self.Vs = nn.ParameterList(
            [
                nn.Parameter(torch.randn(D, rank) * 0.1)
                for _ in range(K)
            ]
        )

        self.beta = nn.Parameter(
            torch.randn(K, q) * 0.05
        )

    def get_alpha(self):
        return self.alpha_fixed

    def get_eta(self):
        return torch.tensor(
            self.eta_fixed,
            dtype=torch.float32,
        )

    def build_profiles(self):
        profiles = []

        for V in self.Vs:
            M = V @ V.T
            tr = torch.trace(M)

            if tr < 1e-12:
                profiles.append(
                    torch.eye(self.D) / self.D
                )
            else:
                profiles.append(M / tr)

        return profiles

    def context_activation(self, c):
        """
        Normalized sigmoid profile activation used in the main model.
        """
        logits = self.beta @ c
        sig = torch.sigmoid(logits)
        return sig / sig.sum().clamp(min=1e-12)

    def forward_chunk(
        self,
        Phi_chunk,
        C_chunk,
        ids_chunk,
        traveler_states,
    ):
        alpha = self.get_alpha()
        eta = self.get_eta()
        profiles = self.build_profiles()

        chunk_size = Phi_chunk.shape[0]
        total_negloglik = torch.tensor(0.0)
        identity = torch.eye(self.D) / self.D

        for i in range(chunk_size):
            traveler = ids_chunk[i]
            phi = Phi_chunk[i]
            c = C_chunk[i]

            if traveler not in traveler_states:
                traveler_states[traveler] = identity.clone()

            rho_prev = traveler_states[traveler]

            # Step 1: Context-dependent profile activation
            pi = self.context_activation(c)

            # Step 2: Predicted traveler state
            mixture = torch.zeros(self.D, self.D)
            for k in range(self.K):
                mixture = mixture + pi[k] * profiles[k]

            rho_t = (
                (1.0 - alpha) * rho_prev
                + alpha * mixture
            )

            # Step 3: Born-rule likelihood
            p = phi @ rho_t @ phi
            p = torch.clamp(p, min=1e-12)
            total_negloglik = (
                total_negloglik - torch.log(p)
            )

            # Step 4: Observation-driven behavioral adaptation
            outer = torch.outer(phi, phi)
            rho_t = (
                (1.0 - eta) * rho_t
                + eta * outer
            )

            # Preserve the original truncated-recursion behavior.
            traveler_states[traveler] = rho_t.detach()

        return total_negloglik, traveler_states


# ============================================================
# 4. Density Matrix Enforcement
# ============================================================

def enforce_density_matrix_np(rho):
    rho = (rho + rho.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)

    total = np.sum(eigvals)

    if total < 1e-12:
        return np.eye(rho.shape[0]) / rho.shape[0]

    eigvals /= total

    return (eigvecs * eigvals) @ eigvecs.T


# ============================================================
# 5. Training
# ============================================================

def train_model(
    Phi,
    C,
    ids,
    *,
    K,
    D,
    rank,
    epochs,
    lr,
    alpha,
    eta,
    chunk_size,
    seed,
    lambda_entropy,
    lambda_beta,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = Phi.shape[0]
    q = C.shape[1]

    Phi_t = torch.tensor(
        Phi,
        dtype=torch.float32,
    )
    C_t = torch.tensor(
        C,
        dtype=torch.float32,
    )

    model = QuantumTravelerModel(
        K=K,
        D=D,
        q=q,
        rank=rank,
        alpha_fixed=alpha,
        eta_fixed=eta,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    best_loss = float("inf")
    best_state = None
    epoch_history = []

    log(
        f"Training | N={N:,} | K={K} | D={D} | "
        f"rank={rank} | alpha={alpha} | eta={eta} | "
        f"epochs={epochs} | chunk={chunk_size} | "
        f"lr={lr} | seed={seed}"
    )

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        traveler_states = {}

        n_chunks = (
            N + chunk_size - 1
        ) // chunk_size

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(
                start + chunk_size,
                N,
            )

            Phi_chunk = Phi_t[start:end]
            C_chunk = C_t[start:end]
            ids_chunk = ids[start:end]

            optimizer.zero_grad()

            chunk_loss, traveler_states = (
                model.forward_chunk(
                    Phi_chunk,
                    C_chunk,
                    ids_chunk,
                    traveler_states,
                )
            )

            # Entropy regularization
            entropy_penalty = torch.tensor(0.0)

            for V in model.Vs:
                rho = V @ V.T
                tr = torch.trace(rho)
                rho = rho / tr.clamp(min=1e-12)

                eigvals = torch.linalg.eigvalsh(rho)
                eigvals = torch.clamp(
                    eigvals,
                    min=1e-12,
                )

                entropy = -(
                    eigvals * torch.log(eigvals)
                ).sum()

                entropy_penalty += entropy

            chunk_loss = (
                chunk_loss
                - lambda_entropy * entropy_penalty
            )

            # L2 regularization on context coefficients
            beta_reg = (
                lambda_beta
                * (model.beta ** 2).sum()
            )
            chunk_loss = chunk_loss + beta_reg

            chunk_loss.backward()
            optimizer.step()

            epoch_loss += chunk_loss.item()

            if (
                (chunk_idx + 1) % 50 == 0
                or (chunk_idx + 1) == n_chunks
            ):
                avg_so_far = epoch_loss / end

                log(
                    f"  Chunk {chunk_idx + 1}/{n_chunks} | "
                    f"Processed: {end:,} | "
                    f"Running avg NLL: {avg_so_far:.6f}"
                )

        avg_nll = epoch_loss / N
        elapsed = time.time() - epoch_start

        epoch_history.append(
            {
                "epoch": epoch + 1,
                "total_loss": epoch_loss,
                "avg_loss": avg_nll,
                "elapsed_minutes": elapsed / 60,
            }
        )

        log(
            f"  Epoch {epoch + 1} complete | "
            f"Total loss: {epoch_loss:,.2f} | "
            f"Avg NLL: {avg_nll:.6f} | "
            f"Time: {elapsed / 60:.1f} min"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss

            best_state = {
                key: value.detach().clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    Vs_final = [
        V.detach().numpy()
        for V in model.Vs
    ]

    beta_final = model.beta.detach().numpy()

    rho_profiles = []

    for V_np in Vs_final:
        M = V_np @ V_np.T
        tr = np.trace(M)

        rho = (
            M / tr
            if tr > 1e-12
            else np.eye(D) / D
        )

        rho = enforce_density_matrix_np(rho)
        rho_profiles.append(rho)

    return {
        "model": model,
        "rho_profiles": rho_profiles,
        "Vs": Vs_final,
        "beta": beta_final,
        "alpha": float(alpha),
        "eta": float(eta),
        "best_loss": float(best_loss),
        "epoch_history": pd.DataFrame(epoch_history),
    }


# ============================================================
# 6. Profile Diagnostics
# ============================================================

def summarize_profiles(rho_profiles):
    rows = []

    for k, rho in enumerate(
        rho_profiles,
        start=1,
    ):
        eigvals = np.linalg.eigvalsh(rho)[::-1]

        row = {
            "profile": k,
            "trace": eigvals.sum(),
        }

        for j, value in enumerate(
            eigvals[:10],
            start=1,
        ):
            row[f"lambda_{j}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 7. Save Variant Results
# ============================================================

def save_variant_results(
    result,
    output_dir,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for k, rho in enumerate(
        result["rho_profiles"],
        start=1,
    ):
        np.save(
            output_dir
            / f"rho_profile_{k}.npy",
            rho,
        )

    np.savez(
        output_dir / "model_params.npz",
        beta=result["beta"],
        alpha=np.array(result["alpha"]),
        eta=np.array(result["eta"]),
        **{
            f"V_{k}": V
            for k, V
            in enumerate(result["Vs"])
        },
    )

    result["epoch_history"].to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    profile_summary = summarize_profiles(
        result["rho_profiles"]
    )
    profile_summary.to_csv(
        output_dir / "profile_eigenvalues.csv",
        index=False,
    )


# ============================================================
# 8. Ablation Study
# ============================================================

def run_ablation_study(
    data_path,
    output_root,
    include_full=False,
):
    # Configuration used in the paper
    D = 200
    K = 3
    FULL_RANK = 10
    EPOCHS = 8
    CHUNK_SIZE = 1000
    LR = 0.005
    SEED = 42
    LAMBDA_ENTROPY = 1e-4
    LAMBDA_BETA = 0.3

    X, C, ids, _, _, _, _, context_cols, _ = (
        load_data(data_path)
    )

    Phi, _ = compute_rff(
        X,
        D=D,
        gamma=1.0,
    )

    # Each dictionary changes only one component relative
    # to the full specification.
    variants = []

    if include_full:
        variants.append(
            {
                "name": "full",
                "label": "Full model",
                "alpha": 0.3,
                "eta": 0.1,
                "rank": FULL_RANK,
            }
        )

    variants.extend(
        [
            {
                "name": "no_context_activation",
                "label": "No context activation (alpha=0)",
                "alpha": 0.0,
                "eta": 0.1,
                "rank": FULL_RANK,
            },
            {
                "name": "no_state_evolution",
                "label": "No state evolution (eta=0)",
                "alpha": 0.3,
                "eta": 0.0,
                "rank": FULL_RANK,
            },
            {
                "name": "rank1",
                "label": "No sub-pattern structure (rank=1)",
                "alpha": 0.3,
                "eta": 0.1,
                "rank": 1,
            },
        ]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for idx, variant in enumerate(
        variants,
        start=1,
    ):
        log("")
        log("=" * 72)
        log(
            f"Ablation {idx}/{len(variants)}: "
            f"{variant['label']}"
        )
        log("=" * 72)

        result = train_model(
            Phi,
            C,
            ids,
            K=K,
            D=D,
            rank=variant["rank"],
            epochs=EPOCHS,
            lr=LR,
            alpha=variant["alpha"],
            eta=variant["eta"],
            chunk_size=CHUNK_SIZE,
            seed=SEED,
            lambda_entropy=LAMBDA_ENTROPY,
            lambda_beta=LAMBDA_BETA,
        )

        variant_dir = (
            output_root / variant["name"]
        )

        save_variant_results(
            result,
            variant_dir,
        )

        summary_rows.append(
            {
                "variant": variant["name"],
                "description": variant["label"],
                "alpha": variant["alpha"],
                "eta": variant["eta"],
                "rank": variant["rank"],
                "best_nll": result["best_loss"],
            }
        )

    summary = pd.DataFrame(summary_rows)

    if "full" in summary["variant"].values:
        full_nll = summary.loc[
            summary["variant"] == "full",
            "best_nll",
        ].iloc[0]

        summary["delta_nll_vs_full"] = (
            summary["best_nll"] - full_nll
        )
    else:
        summary["delta_nll_vs_full"] = np.nan

    summary.to_csv(
        output_root / "ablation_summary.csv",
        index=False,
    )

    # Save context-column order for reproducibility.
    pd.Series(
        context_cols,
        name="context_variable",
    ).to_csv(
        output_root / "context_columns.csv",
        index=False,
    )

    log("")
    log("=" * 72)
    log("ABLATION STUDY COMPLETE")
    log("=" * 72)
    log(summary.to_string(index=False))
    log(
        f"\nSummary saved to: "
        f"{output_root / 'ablation_summary.csv'}"
    )

    return summary


# ============================================================
# 9. Command-Line Interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the three ablation experiments for the "
            "quantum-inspired traveler behavioral profiling model."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("activities_merged4.csv"),
        help="Path to activities_merged4.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ablation_results"),
        help="Directory for ablation-study outputs",
    )

    parser.add_argument(
        "--include-full",
        action="store_true",
        help=(
            "Also estimate the full model and compute "
            "Delta NLL relative to it."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log(f"Python:  {sys.version}")
    log(f"NumPy:   {np.__version__}")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA:    {torch.cuda.is_available()}")
    log(f"Start:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    run_ablation_study(
        data_path=args.data,
        output_root=args.output,
        include_full=args.include_full,
    )

    log(
        f"End:     "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
