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

The main steps are:

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

Activity type and transport mode are one-hot encoded and combined with activity duration. The resulting behavioral vector is standardized and mapped into a nonlinear Random Fourier Feature (RFF) space.

The RFF vector is L2-normalized before entering the density-matrix model.

### 2. Population-level behavioral profiles

Each behavioral profile \(k\) is represented by a density matrix

```text
rho_k = V_k V_k^T / Tr(V_k V_k^T)
```

which is symmetric, positive semidefinite, and trace-normalized by construction.

The eigenstructure of each profile allows one profile to contain multiple behavioral sub-patterns rather than forcing each profile to represent only one activity-mode pattern.

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

### 5. Estimation and profile interpretation

The profile factors \(V_k\) and context coefficients \(\beta_k\) are optimized with Adam using the negative log-likelihood, together with:

- entropy regularization on profile eigenvalues; and
- L2 regularization on the context coefficients.

After estimation, each density matrix is eigendecomposed. The code then identifies the activity types, transport modes, and activity durations associated with the dominant eigenmodes using the observed TimeUse+ activity records.

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

## Preparing the Input Data

The model script expects a cleaned, episode-level CSV. The preprocessing used for the empirical application is described below.

### Step 1 — Remove untracked activities

Remove activity records that contain no usable activity information.

### Step 2 — Consolidate home activities

All activities recorded at the home location are consolidated under a single:

```text
Home
```

activity category.

### Step 3 — Merge consecutive records

Within each participant's chronological record, merge consecutive rows when they correspond to:

- the same participant;
- the same activity; and
- the same location.

The merged rows form one activity episode. Sum the activity durations across the merged rows.

### Step 4 — Recover missing activity durations

For episodes with missing duration, calculate duration using the recorded start and end times.

Activity duration should be stored in **minutes**.

### Step 5 — Match weather information

Match NOAA GSOD temperature and precipitation information to each activity episode based on:

- activity location; and
- episode start time.

Append the resulting weather variables to the activity data.

### Step 6 — Retain trip distance

Retain the trip distance associated with each activity episode. In the empirical application, trip distance is represented in **kilometers**.

### Step 7 — Construct the final episode-level file

The final dataset should contain one row per activity episode and include at least the following columns:

| Column | Description |
| --- | --- |
| `participant_id` | Participant identifier |
| `started_at_utc_global` | Episode start timestamp used to order each participant's sequence |
| `activity_new` | Activity type |
| `mode_choice_new` | Transport mode used to reach the activity |
| `event_duration_new` | Activity duration in minutes |
| `time_of_day_new` | Local hour of day, 0–23 |
| `day_of_week_new` | Day of week, coded 1–7 with Saturday = 6 and Sunday = 7 |
| `temp_celsius_new` | Temperature in degrees Celsius |
| `precip_depth_mm_new` | Precipitation depth in millimeters |
| `trip_distance_new` | Trip distance in kilometers |

The original TimeUse+ activity and transport-mode categories are retained, except for the consolidation of activities recorded at home.

The empirical dataset used in the manuscript contains approximately **146,318 activity episodes from 1,310 participants** after preprocessing.

### Input filename

The current released script intentionally preserves the filename used in the analysis:

```text
activities_merged4.csv
```

Save or copy the final prepared dataset under that filename and place it in the same directory as the Python script.

---

## What the Script Does to the Prepared Data

Once `activities_merged4.csv` has been created, no additional manual feature engineering is required.

The script:

1. removes rows missing required model variables;
2. sorts episodes by participant and timestamp;
3. converts hour of day into five time periods;
4. converts day of week into Weekday, Saturday, or Sunday;
5. one-hot encodes activity type and transport mode;
6. builds the context matrix;
7. standardizes the behavioral and context variables;
8. generates normalized Random Fourier Features;
9. estimates the density-matrix behavioral profiles;
10. sequentially updates traveler states;
11. performs profile eigendecomposition; and
12. decodes the dominant profile sub-patterns using the observed activity data.


## Run

Place these two files in the same directory:

```text
quantum_traveler_profiling.py
activities_merged4.csv
```

Then run:

```bash
python quantum_traveler_profiling.py
```

The default configuration reproduces the core model specification used in the manuscript:

| Setting | Value |
| --- | ---: |
| RFF dimension `D` | 200 |
| Behavioral profiles `K` | 3 |
| Profile rank | 10 |
| Epochs | 8 |
| Chunk size | 1000 |
| Learning rate | 0.005 |
| `alpha` initialization | 0.30 |
| `eta` initialization | 0.10 |
| Entropy regularization | \(10^{-4}\) |
| L2 regularization on `beta` | 0.30 |
| Random seed | 42 |

The RFF mapping itself uses:

```text
random_state = 42
```


## Contact

For questions about the modeling framework or repository, please contact the corresponding author through the contact information provided in the manuscript.
