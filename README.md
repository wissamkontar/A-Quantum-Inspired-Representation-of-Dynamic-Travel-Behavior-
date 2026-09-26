# A Quantum-Inspired Representation of Dynamic Travel Behavior

Reproducible implementation of the traveler behavioral profiling framework developed in:

> **A Quantum-Inspired Representation of Dynamic Travel Behavior**  
> Omid Armantalab, Mohammad Elayan, Li Zhao, Nathan Huynh, and Wissam Kontar  
> Department of Civil & Environmental Engineering, University of Nebraska–Lincoln

This repository implements a quantum-inspired framework that represents travel behavior as a **continuous, evolving behavioral state** rather than a fixed traveler class. Activity type, transport mode, and activity duration are represented jointly; population-level behavioral profiles are learned as density matrices; episode-level context adjusts the activation of those profiles; and each traveler's personal state is updated sequentially as new activity episodes are observed.

---

## Model

For traveler \(i\) at activity episode \(t\), the observed behavioral state is

```text
x_it = [activity type, transport mode, activity duration]
```

and the episode context is

```text
c_it = [time of day, day type, temperature, precipitation, trip distance].
```

The main modeling sequence is:

```text
Observed activity episode
        |
        v
Behavioral state x_it
(activity + mode + duration)
        |
        v
Random Fourier Feature mapping
        |
        v
L2-normalized feature vector phi(x_it)
        |
        +-----------------------------+
        |                             |
        v                             v
Population density-matrix       Episode context c_it
profiles rho_k                  determines profile weights
        |                             |
        +--------------+--------------+
                       |
                       v
            Predicted personal state
                       |
                       v
               Born-rule likelihood
                       |
                       v
          Observation-driven state update
```

### 1. Nonlinear behavioral representation

Activity type and transport mode are one-hot encoded and combined with activity duration. The resulting behavioral vector is standardized and mapped into a nonlinear Random Fourier Feature (RFF) space. The RFF vector is L2-normalized before entering the density-matrix model.

### 2. Population-level behavioral profiles

Each behavioral profile \(k\) is represented by a density matrix

```text
rho_k = V_k V_k^T / Tr(V_k V_k^T)
```

which is symmetric, positive semidefinite, and trace-normalized by construction.

The eigenstructure of each profile allows a single profile to contain multiple behavioral sub-patterns rather than forcing each profile to represent only one activity-mode pattern.

### 3. Context-dependent profile activation

The context vector contains:

- time of day: Morning, Midday, Afternoon, Evening, or Night;
- day type: Weekday, Saturday, or Sunday;
- temperature;
- precipitation; and
- trip distance.

For each episode, the model computes normalized sigmoid activation weights

```text
pi_k(c_it) = sigmoid(beta_k^T c_it) /
             sum_j sigmoid(beta_j^T c_it)
```

so that context adjusts the relative activation of the population-level behavioral profiles.

### 4. Personal behavioral state evolution

Before observing the activity episode, the traveler's predicted state combines behavioral inertia with the context-weighted population profiles:

```text
rho_i^pred(t)
    = (1 - alpha) rho_i(t-1)
      + alpha sum_k pi_k(c_it) rho_k
```

The probability of the observed behavioral state is evaluated through the Born rule:

```text
p(x_it)
    = phi(x_it)^T rho_i^pred(t) phi(x_it)
```

After the episode is observed, the traveler's state is updated:

```text
rho_i(t)
    = (1 - eta) rho_i^pred(t)
      + eta phi(x_it) phi(x_it)^T
```

This allows each traveler to remain a continuous mixture of behavioral profiles while carrying information from previous episodes forward.

### 5. Estimation and interpretation

The profile factors \(V_k\) and context coefficients \(\beta_k\) are optimized with Adam using the negative log-likelihood, together with entropy regularization on profile eigenvalues and L2 regularization on the context coefficients.

After estimation, each density matrix is eigendecomposed. The dominant eigenmodes are then interpreted using the observed activity, transport-mode, and duration data.

---

## Repository Files

The repository is organized around the main stages of the analysis:

| File | Purpose |
| --- | --- |
| `preprocess_timeuse.py` | Cleans and prepares the TimeUse+ and weather data and produces `activities_merged4.csv`. |
| `quantum_traveler_profiling.py` | Estimates the main quantum-inspired traveler behavioral profiling model. |
| `ablation_study.py` | Runs the three ablation experiments used in the paper. |
| `comparison_models.py` | Estimates the comparison MNL, latent class, and HMM models. |
| `paper_figures.py` | Contains reusable functions for reproducing selected figures from the paper. |
| `requirements.txt` | Lists the Python packages required to run the repository code. |

---

## Data

### TimeUse+ main study

The empirical application uses the **TimeUse+ main study**, a longitudinal smartphone-based travel and time-use diary conducted in German-speaking Switzerland. Participants tracked and validated their activity and travel behavior over 28 days.

The anonymized TimeUse+ main-study dataset is distributed by ETH Zurich's Research Collection:

<https://www.research-collection.ethz.ch/handle/20.500.11850/639241>

The accompanying data documentation is:

> Winkler, C., Meister, A., Isenschmid, U., Lerdo de Tejada Acosta, K., Le, B. A., and Axhausen, K. W.  
> *TimeUse+ Main Study: Data and variable description.*

The data paper is:

> Winkler, C., Meister, A., and Axhausen, K. W.  
> *The TimeUse+ data set: 4 weeks of time use and expenditure data based on GPS tracks.*  
> Transportation, 53, 909–935 (2026).  
> DOI: 10.1007/s11116-024-10517-1

The TimeUse+ data are **not included in this repository**.

### Weather data

Temperature and precipitation are obtained from NOAA's **Global Surface Summary of the Day (GSOD)** data through the Google BigQuery public dataset:

```text
bigquery-public-data.noaa_gsod
```

Weather observations are matched to each activity episode using the activity location and episode start time.

---

## Data Preprocessing

The raw TimeUse+ activity records and weather information must first be prepared using:

```bash
python preprocess_timeuse.py
```

The preprocessing script performs the steps used in the empirical analysis, including:

- removing unusable activity records;
- consolidating home and work activities;
- assigning trip mode and distance to destination activities;
- constructing local temporal variables;
- matching weather observations;
- calculating activity duration;
- merging consecutive records corresponding to the same activity and location; and
- creating the final sequential episode dataset.

The resulting file is:

```text
activities_merged4.csv
```

For full implementation details, see `preprocess_timeuse.py`.

The empirical dataset used in the manuscript contains approximately **146,318 activity episodes from 1,310 participants** after preprocessing.

---

## Installation

Python 3.10 or newer is recommended.

Create and activate a virtual environment if desired, then install all required packages with:

```bash
pip install -r requirements.txt
```

The requirements file covers all scripts in this repository.

---

## Reproducing the Main Model

After preprocessing, place `activities_merged4.csv` in the repository root and run:

```bash
python quantum_traveler_profiling.py
```

The main script:

1. loads and orders the activity episodes;
2. one-hot encodes activity type and transport mode;
3. constructs the context variables;
4. standardizes the behavioral and context inputs;
5. generates normalized Random Fourier Features;
6. estimates the density-matrix behavioral profiles;
7. sequentially updates traveler states;
8. performs eigendecomposition of the learned profiles; and
9. decodes the dominant behavioral sub-patterns using the observed activity data.

The default model configuration used in the paper is:

| Setting | Value |
| --- | ---: |
| RFF dimension `D` | 200 |
| Behavioral profiles `K` | 3 |
| Profile rank | 10 |
| Epochs | 8 |
| Chunk size | 1000 |
| Learning rate | 0.005 |
| `alpha` | 0.30 |
| `eta` | 0.10 |
| Entropy regularization | \(10^{-4}\) |
| L2 regularization on `beta` | 0.30 |

---

## Ablation Study

The three ablation experiments used in the paper are combined in:

```text
ablation_study.py
```

Run the ablations with:

```bash
python ablation_study.py
```

The script evaluates:

- **No context activation:** \(\alpha = 0\);
- **No behavioral adaptation/state update:** \(\eta = 0\); and
- **No within-profile sub-pattern structure:** rank \(r = 1\).

To also estimate the full model in the same run and calculate changes in negative log-likelihood relative to the full specification:

```bash
python ablation_study.py --include-full
```

Results are saved under:

```text
ablation_results/
```

including an `ablation_summary.csv` file.

---

## Comparison Models

The benchmark models used in the paper are implemented in:

```text
comparison_models.py
```

Run all comparison models with:

```bash
python comparison_models.py
```

The script estimates:

- **Multinomial Logit (MNL)**;
- **Joint Latent Class Model** with participant-level class membership; and
- **Hidden Markov Model (HMM)** with dynamic latent states.

Results are saved under:

```text
comparison_results/
```

with separate subdirectories for the MNL, latent class, and HMM models, together with a combined summary file.

---

## Reproducing Paper Figures

Selected figure-generation routines used in the paper are provided in:

```text
paper_figures.py
```

This file contains reusable functions for:

- the individual temperature-shock counterfactual figure;
- the population temperature-shock summary;
- the context-activation coefficient figure;
- density-matrix profile fingerprint/eigenmode figures; and
- traveler state evolution with optional HMM-state comparison.

The figure script does **not** re-estimate the model. It uses arrays, estimated parameters, and tables produced by the model and comparison analyses.

Functions can be imported as needed:

```python
from paper_figures import (
    plot_individual_temperature_shock,
    plot_population_temperature_shock,
    plot_context_activation,
    plot_profile_fingerprints,
    plot_state_evolution,
)
```

Each function contains a docstring describing the required inputs and saves the corresponding figure in both PDF and PNG formats.

---

## Reproducibility

To keep stochastic components reproducible, **random seed 42 is used throughout the repository wherever a random seed is required**.

This includes, where applicable:

- NumPy random initialization;
- PyTorch random initialization;
- Random Fourier Feature generation;
- latent class model initialization;
- HMM initialization; and
- random participant selection for traveler-level visualization.

The RFF mapping also uses:

```text
random_state = 42
```

Using the same seed, software versions, input data, and model settings is recommended when reproducing the reported results.

---

## Typical Workflow

A complete repository workflow is:

```text
1. Obtain TimeUse+ and NOAA GSOD data
            |
            v
2. preprocess_timeuse.py
            |
            v
   activities_merged4.csv
            |
            +-----------------------------+
            |              |              |
            v              v              v
3. Main model      4. Ablation       5. Comparison
   quantum_           ablation_          comparison_
   traveler_          study.py           models.py
   profiling.py
            |              |              |
            +--------------+--------------+
                           |
                           v
                  6. paper_figures.py
```

In command-line form:

```bash
python preprocess_timeuse.py
python quantum_traveler_profiling.py
python ablation_study.py --include-full
python comparison_models.py
```

The outputs from these analyses can then be passed to the plotting functions in `paper_figures.py` to reproduce selected manuscript figures.


## Contact

For questions about the modeling framework or repository, please contact the corresponding author through the contact information provided in the manuscript.
