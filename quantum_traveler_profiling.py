
"""
Quantum-Inspired Traveler Behavioral Profiling (PyTorch) for the TimeUse+ dataset.

"""

import numpy as np
import pandas as pd
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.kernel_approximation import RBFSampler


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

    # ── Categorical encoding for time and day ─────────────────
    def time_to_period(hour):
        if 6 <= hour < 10:
            return 'Morning'
        elif 10 <= hour < 14:
            return 'Midday'
        elif 14 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'

    def dow_to_type(dow):
        if dow == 6:
            return 'Saturday'
        elif dow == 7:
            return 'Sunday'
        else:
            return 'Weekday'

    df['time_period'] = df['time_of_day_new'].apply(time_to_period)
    df['day_type']    = df['day_of_week_new'].apply(dow_to_type)

    # ── Drop rows with missing values ──────────────────────────
    required = ['event_duration_new',
                'time_of_day_new', 'day_of_week_new',
                'temp_celsius_new', 'precip_depth_mm_new',
                'trip_distance_new', 'activity_new', 'mode_choice_new',
                'participant_id', 'started_at_utc_global']
    df = df.dropna(subset=required).copy()
    log(f"Rows after dropping NaN: {len(df):,}")

    # ── Sort by person then global time ───────────────────────
    df = df.sort_values(['participant_id', 'started_at_utc_global']).reset_index(drop=True)

    # ── One-hot encode AFTER dropna and sort ──────────────────
    act_dummies  = pd.get_dummies(
        df['activity_new'].fillna('Other').astype(str), prefix='act'
    )
    mode_dummies = pd.get_dummies(
        df['mode_choice_new'].fillna('other').astype(str), prefix='mode'
    )

    log(f"Activity categories: {list(act_dummies.columns)}")
    log(f"Mode categories:     {list(mode_dummies.columns)}")

    # ── Build behavior matrix ──────────────────────────────────
    behavior_df = pd.concat([
        act_dummies,
        mode_dummies,
        df[['event_duration_new']]
    ], axis=1)

    behavior_cols = list(behavior_df.columns)

    # ── Build context matrix ───────────────────────────────────
    time_dummies = pd.get_dummies(df['time_period'], prefix='time')
    dow_dummies  = pd.get_dummies(df['day_type'],    prefix='dow')

    context_df = pd.concat([
        time_dummies,
        dow_dummies,
        df[['temp_celsius_new', 'precip_depth_mm_new', 'trip_distance_new']]
    ], axis=1)

    context_cols = list(context_df.columns)

    # ── Standardize ────────────────────────────────────────────
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(behavior_df.values)

    scaler_c = StandardScaler()
    C = scaler_c.fit_transform(context_df.values)

    ids   = df['participant_id'].values
    times = df['started_at_utc_global'].values

    log(f"Loaded {len(df):,} activity episodes")
    log(f"  d={X.shape[1]} state variables | q={C.shape[1]} context variables")
    log(f"  {df['participant_id'].nunique():,} unique participants")

    return X, C, ids, times, scaler_x, scaler_c, behavior_cols, df

# ============================================================
# 2. Random Fourier Feature Map
# ============================================================

def compute_rff(X, D=100, gamma=1.0):
    rff = RBFSampler(gamma=gamma, n_components=D, random_state=42)
    Phi = rff.fit_transform(X)

    norms = np.linalg.norm(Phi, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    Phi  /= norms

    log(f"RFF dimension: D={Phi.shape[1]}")
    return Phi, rff


# ============================================================
# 3. Model Definition
# ============================================================

class QuantumTravelerModel(nn.Module):
    """
    Quantum-inspired traveler behavioral profiling model.

    Each traveler has a personal density matrix ρ_i that evolves
    across their activity sequence. Population-level profiles ρ_k
    serve as situational attractors — not fixed categories.

    Learnable parameters:
      - V_k (K matrices, D×rank): profile factors
      - beta (K×q): context activation weights
      - logit_eta: behavioral adaptation strength
    """

    def __init__(self, K, D, q, rank=100, alpha_init=0.2, eta_init=0.1):
        super().__init__()

        self.K    = K
        self.D    = D
        self.rank = rank

        # Profile factors V_k
        self.Vs = nn.ParameterList([
            nn.Parameter(torch.randn(D, rank) * 0.1) for _ in range(K)
        ])

        # Context activation weights β_k
        self.beta = nn.Parameter(torch.randn(K, q) * 0.05)

        self.fixed_alpha = alpha_init
        # REPRODUCIBILITY NOTE: eta is initialized here as a PyTorch parameter.
        # However, the traveler state is detached after each episode below, so
        # gradients from later episodes do not propagate through the eta update.
        # In the current implementation eta therefore remains effectively at its
        # initialization value for the forward state recursion.
        self.logit_eta   = nn.Parameter(
            torch.tensor(np.log(eta_init / (1.0 - eta_init)),
                         dtype=torch.float32)
        )

    def get_alpha(self):
        return self.fixed_alpha

    def get_eta(self):
        return torch.sigmoid(self.logit_eta)

    def build_profiles(self):
        """Build density matrices ρ_k = V_k V_k^T / Tr(V_k V_k^T)."""
        profiles = []
        for V in self.Vs:
            M  = V @ V.T
            tr = torch.trace(M)
            if tr < 1e-12:
                profiles.append(torch.eye(self.D) / self.D)
            else:
                profiles.append(M / tr)
        return profiles

    # NOTE: The function name is retained from development; the implementation
    # below uses normalized sigmoid activation, as described in the manuscript.
    def softmax_activation(self, c):
        """
        Normalized Sigmoid activation — replaces softmax.
        Sigmoid is bounded (0,1) regardless of logit size.
        No exponential explosion — multiple profiles can be
        simultaneously active without one dominating.
        """
        logits = self.beta @ c
        sig    = torch.sigmoid(logits)
        return sig / sig.sum().clamp(min=1e-12)

    def forward_chunk(self, Phi_chunk, C_chunk, ids_chunk, traveler_states):
        """
        Process a chunk of sequential activity episodes.

        For each episode:
          1. Context activation π_k(c)
          2. State evolution with α blending
          3. Born-rule likelihood
          4. Behavior-driven adaptation with η
        """
        alpha    = self.get_alpha()
        eta      = self.get_eta()
        profiles = self.build_profiles()

        chunk_size     = Phi_chunk.shape[0]
        total_negloglik = torch.tensor(0.0)
        identity        = torch.eye(self.D) / self.D

        for i in range(chunk_size):
            traveler = ids_chunk[i]
            phi      = Phi_chunk[i]
            c        = C_chunk[i]

            # Initialize new traveler state from identity matrix
            if traveler not in traveler_states:
                traveler_states[traveler] = identity.clone()

            rho_prev = traveler_states[traveler]

            # Step 1: Context activation
            pi = self.softmax_activation(c)

            # Step 2: State evolution
            mixture = torch.zeros(self.D, self.D)
            for k in range(self.K):
                mixture = mixture + pi[k] * profiles[k]
            rho_t = (1.0 - alpha) * rho_prev + alpha * mixture

            # Step 3: Born-rule likelihood
            p = phi @ rho_t @ phi
            p = torch.clamp(p, min=1e-12)
            total_negloglik = total_negloglik - torch.log(p)

            # Step 4: Behavior-driven adaptation
            outer = torch.outer(phi, phi)
            rho_t = (1.0 - eta) * rho_t + eta * outer

            # Detaching bounds the computation graph across the long activity sequence
            # and keeps memory use manageable during chunked training.
            traveler_states[traveler] = rho_t.detach()

        return total_negloglik, traveler_states


# ============================================================
# 4. Density Matrix Enforcement
# ============================================================

def enforce_density_matrix_np(rho):
    rho = (rho + rho.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    total   = np.sum(eigvals)
    if total < 1e-12:
        return np.eye(rho.shape[0]) / rho.shape[0]
    eigvals /= total
    return (eigvecs * eigvals) @ eigvecs.T


# ============================================================
# 5. Checkpointing
# ============================================================

def save_checkpoint(path, epoch, model, optimizer, best_loss):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss":            best_loss,
    }, path)
    log(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, weights_only=False)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        log(f"  WARNING: Checkpoint shape mismatch — {e}")
        log(f"  Deleting incompatible checkpoint and starting fresh.")
        os.remove(path)
        best_path = os.path.join(os.path.dirname(path), "best_model.pt")
        if os.path.exists(best_path):
            os.remove(best_path)
        return 0, float('inf')
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["best_loss"]


# ============================================================
# 6. Training
# ============================================================

def train_model(Phi, C, ids, K, D, rank, epochs, lr,
                alpha_init, eta_init, chunk_size,
                checkpoint_dir, resume=True, seed=42,
                lambda_entropy=1e-4,
                lambda_beta=0.05):

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = Phi.shape[0]
    q = C.shape[1]

    Phi_t = torch.tensor(Phi, dtype=torch.float32)
    C_t   = torch.tensor(C,   dtype=torch.float32)

    model     = QuantumTravelerModel(K=K, D=D, q=q, rank=rank,
                                     alpha_init=alpha_init, eta_init=eta_init)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_loss   = float('inf')
    ckpt_path   = os.path.join(checkpoint_dir, "checkpoint.pt")

    if resume and os.path.exists(ckpt_path):
        start_epoch, best_loss = load_checkpoint(ckpt_path, model, optimizer)
        log(f"Resumed from epoch {start_epoch} | best loss: {best_loss:,.2f}")
        start_epoch += 1

    log(f"\nTraining | N={N:,} | K={K} | D={D} | q={q} | "
        f"epochs={epochs} | chunk={chunk_size} | lr={lr} | seed={seed}")

    for epoch in range(start_epoch, epochs):
        epoch_start    = time.time()
        epoch_loss     = 0.0
        traveler_states = {}

        n_chunks = (N + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end   = min(start + chunk_size, N)

            Phi_chunk = Phi_t[start:end]
            C_chunk   = C_t[start:end]
            ids_chunk = ids[start:end]

            optimizer.zero_grad()

            chunk_loss, traveler_states = model.forward_chunk(
                Phi_chunk, C_chunk, ids_chunk, traveler_states
            )

            # Entropy regularization
            entropy_penalty = torch.tensor(0.0)
            for V in model.Vs:
                rho     = V @ V.T
                tr      = torch.trace(rho)
                rho     = rho / tr.clamp(min=1e-12)
                eigvals = torch.linalg.eigvalsh(rho)
                eigvals = torch.clamp(eigvals, min=1e-12)
                entropy = -(eigvals * torch.log(eigvals)).sum()
                entropy_penalty += entropy

            chunk_loss = chunk_loss - lambda_entropy * entropy_penalty

            # ── Beta L2 regularization ──────────────────────────
            beta_reg   = lambda_beta * (model.beta ** 2).sum()
            chunk_loss = chunk_loss + beta_reg

            chunk_loss.backward()
            optimizer.step()
            
            epoch_loss += chunk_loss.item()

            if (chunk_idx + 1) % 50 == 0 or (chunk_idx + 1) == n_chunks:
                avg_so_far = epoch_loss / end
                log(f"  Chunk {chunk_idx+1}/{n_chunks} | "
                    f"Processed: {end:,} | "
                    f"Running avg NLL: {avg_so_far:.6f}")

        avg_nll = epoch_loss / N
        elapsed = time.time() - epoch_start

        log(f"  Epoch {epoch+1} complete | "
            f"Total loss: {epoch_loss:,.2f} | "
            f"Avg NLL: {avg_nll:.6f} | "
            f"Time: {elapsed/60:.1f} min | "
            f"α={model.get_alpha():.4f} | "
            f"η={model.get_eta().item():.4f} | "
            f"β_max={model.beta.abs().max().item():.2f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, "best_model.pt"))
            log(f"  >> New best loss! Saved best_model.pt")

        save_checkpoint(ckpt_path, epoch, model, optimizer, best_loss)

    # Load best model
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=False))
        log("Loaded best model parameters.")

    alpha_final = model.get_alpha()
    eta_final   = model.get_eta().item()
    beta_final  = model.beta.detach().numpy()
    Vs_final    = [V.detach().numpy() for V in model.Vs]

    rho_profiles = []
    for V_np in Vs_final:
        M   = V_np @ V_np.T
        tr  = np.trace(M)
        rho = M / tr if tr > 1e-12 else np.eye(D) / D
        rho = enforce_density_matrix_np(rho)
        rho_profiles.append(rho)

    log(f"\n{'='*60}")
    log(f"Training complete.")
    log(f"  Best loss: {best_loss:,.2f}")
    log(f"  Final α={alpha_final:.4f}, η={eta_final:.4f}")
    log(f"  Context variables: see load_data output above")

    log(f"  Final β (rows=profiles, cols=context vars):\n{beta_final}")
    log(f"{'='*60}")

    return rho_profiles, Vs_final, beta_final, alpha_final, eta_final


# ============================================================
# 7. Interpretation
# ============================================================

def interpret_profiles(rho_profiles):
    log(f"\n{'='*60}")
    log("Profile Interpretation (Eigenanalysis)")
    log(f"{'='*60}")
    for k, rho in enumerate(rho_profiles):
        eigvals, eigvecs = np.linalg.eigh(rho)
        order   = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]

        top_n = 10
        log(f"\n  Profile {k+1}:")
        log(f"    Top {top_n} eigenvalues: {np.array2string(eigvals[:top_n], precision=6)}")
        log(f"    Cumulative weight (top 5):  {np.sum(eigvals[:5]):.4f}")
        log(f"    Cumulative weight (top 10): {np.sum(eigvals[:10]):.4f}")
        log(f"    Trace: {np.sum(eigvals):.6f}")


# ============================================================
# 8. Save Results
# ============================================================

def save_results(rho_profiles, Vs, beta, alpha, eta, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    for k, rho in enumerate(rho_profiles):
        np.save(os.path.join(output_dir, f"rho_profile_{k+1}.npy"), rho)

    np.savez(os.path.join(output_dir, "model_params.npz"),
             beta=beta,
             alpha=np.array(alpha.item() if hasattr(alpha, 'item') else alpha),
             eta=np.array(eta.item() if hasattr(eta, 'item') else eta),
             **{f"V_{k}": V for k, V in enumerate(Vs)})

    log(f"Results saved to {output_dir}/")

# ============================================================
# 9. Decode Profiles Using Actual Data
# ============================================================

def decode_profiles(rho_profiles, Phi, df, output_dir="results"):

    act_base  = df['activity_new'].value_counts(normalize=True) * 100
    mode_base = df['mode_choice_new'].value_counts(normalize=True) * 100

    log(f"\n{'='*60}")
    log("Profile Decoding (Actual Data)")
    log(f"{'='*60}")

    for k, rho in enumerate(rho_profiles):
        eigvals, eigvecs = np.linalg.eigh(rho)
        order   = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        log(f"\n  Profile {k+1}:")

        n_modes = min(3, np.sum(eigvals > 0.01))
        if n_modes == 0:
            n_modes = 1

        for m in range(n_modes):
            v           = eigvecs[:, m]
            activations = (Phi @ v) ** 2
            activations /= activations.sum()

            act_pct  = pd.Series(activations).groupby(
                df['activity_new'].values).sum() * 100
            mode_pct = pd.Series(activations).groupby(
                df['mode_choice_new'].values).sum() * 100

            act_diff  = (act_pct  - act_base).sort_values(ascending=False)
            mode_diff = (mode_pct - mode_base).sort_values(ascending=False)

            duration_mean = np.sum(activations * df['event_duration_new'].values)

            log(f"\n    Mode {m+1} (λ={eigvals[m]:.4f}):")
            log(f"      Activities (profile% vs dataset%):")
            for act in act_diff.head(4).index:
                p = act_pct.get(act, 0)
                b = act_base.get(act, 0)
                log(f"        {act:<30} {p:5.1f}%  vs  {b:5.1f}%  ({p-b:+.1f}%)")
            log(f"      Modes (profile% vs dataset%):")
            for mode in mode_diff.head(4).index:
                p = mode_pct.get(mode, 0)
                b = mode_base.get(mode, 0)
                log(f"        {mode:<15} {p:5.1f}%  vs  {b:5.1f}%  ({p-b:+.1f}%)")
            log(f"      Avg duration : {duration_mean:.1f} min")
  


# ============================================================
# 10. Main Pipeline
# ============================================================

if __name__ == "__main__":

    log(f"Python:  {sys.version}")
    log(f"NumPy:   {np.__version__}")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA:    {torch.cuda.is_available()}")
    log(f"Start:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Configuration ─────────────────────────────────────────
    # Input must be the fully prepared episode-level file described in README.md.
    # The executable path is intentionally left unchanged for exact reproducibility.
    DATA_PATH  = "activities_merged4.csv"
    D          = 200   # RFF dimensions
    K          = 3     # number of behavioral profiles
    RANK       = 10     # low-rank profile factor (D=200, rank=10)
    EPOCHS     = 8
    CHUNK_SIZE = 1000  
    LR         = 0.005

    # ── Load data ─────────────────────────────────────────────
    X, C, ids, times, scaler_x, scaler_c, behavior_cols, df = load_data(DATA_PATH)


    # ── Compute RFF features ──────────────────────────────────
    Phi, rff_sampler = compute_rff(X, D=D, gamma=1.0)

    # ── Train model ───────────────────────────────────────────
    SEED = 42
    rho_profiles, Vs, beta, alpha, eta = train_model(
        Phi, C, ids,
        K=K, D=D, rank=RANK, epochs=EPOCHS,
        lr=LR, alpha_init=0.3, eta_init=0.1,
        chunk_size=CHUNK_SIZE,
        checkpoint_dir=f"checkpoints_seed{SEED}",
        resume=True,
        seed=SEED,
        lambda_entropy=1e-4,
        lambda_beta=0.3,
    )

    # ── Interpret profiles ────────────────────────────────────
    interpret_profiles(rho_profiles)

    # ── Save results ──────────────────────────────────────────
    save_results(rho_profiles, Vs, beta, alpha, eta,
                 output_dir=f"results_seed{SEED}")

    # ── Decode profiles using actual data ─────────────────────
    decode_profiles(rho_profiles, Phi, df,
                    output_dir=f"results_seed{SEED}")

    log(f"\nEnd:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("Done.")