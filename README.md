# Color Recipe Builder

> Reference implementation for the paper
> **"Genetic Algorithm for Color Recipe Prediction in Industrial Settings"** — Barta et al.

Given a **target color** — a reflectance spectrum or CIELAB values — the Color
Recipe Builder searches for a **recipe**: a mixture of **inorganic pigments** and
their concentrations that reproduces that color as closely as possible, while
keeping the number of pigments low and staying within concentrations seen in
historical production recipes. It targets an **industrial setting**, where recipes
must be manufacturable and consistent with established production practice.

The repository holds three approaches to that search:

| Module  | Approach | Role |
|---------|----------|------|
| [`GA/`](#ga--genetic-algorithm)     | Genetic Algorithm            | Evolves pigment mixtures toward a target color. |
| [`NN/`](#nn--neural-network)        | Neural Network               | Multi-label classifier that predicts *which* pigments a recipe should contain. |
| [`GANN/`](#gann--genetic-algorithm--neural-network) | Genetic Algorithm + Neural Network | Uses the NN prediction to seed and constrain the GA search space. |
| [`Utils/`](#utils--shared-helpers)  | Shared helpers               | Color-science conversions, data splitting, plotting, data-inspection notebooks. |

> [!WARNING]
> **This code is not runnable as-is.** The datasets, trained models, and
> environment values are proprietary and are not included. What remains is
> published to make the logic behind the Color Recipe Builder easier to follow,
> not to be executed.

---

## Contents

- [Repository layout](#repository-layout)
- [Core concepts](#core-concepts)
- [`GA/` – Genetic Algorithm](#ga--genetic-algorithm)
- [`NN/` – Neural Network](#nn--neural-network)
- [`GANN/` – Genetic Algorithm + Neural Network](#gann--genetic-algorithm--neural-network)
- [`Utils/` – Shared helpers](#utils--shared-helpers)
- [Workflow](#workflow)
- [Dependencies](#dependencies)
- [Citation](#citation)

---

## Repository layout

```
.
├── GA/
│   ├── GA_scripts.py          # the genetic algorithm
│   ├── utils.py               # normalization, recipe→Lab, results aggregation
│   └── GA.ipynb               # driver notebook
├── NN/
│   ├── NN_tuner.py            # data loading, model, training, evaluation, tuning
│   ├── pigment_prediction.ipynb
│   └── NN_result_analysis.ipynb
├── GANN/
│   ├── ga_nn.py               # end-to-end NN-guided GA pipeline
│   └── ga_nn_notebook.ipynb   # same pipeline + evaluation plots
└── Utils/
    ├── util_methods.py        # UtilMethods: color science, splits, plotting
    ├── read_recipes.ipynb     # build train/test/val splits
    └── spectrum_viewer.ipynb  # plot reflectance spectra
```

The imports assume this directory sits as a package named **`Code/`** inside a
larger project whose root contains a `.rootfolder` marker file, a `.env`, and a
`Dataset/` folder. `UtilMethods.find_project_root()` walks up the tree looking for
that marker.

---

## Core concepts

| Term | Meaning |
|------|---------|
| **Recipe / individual** | A vector of pigment concentrations, one entry per available pigment. Most entries are zero — a recipe uses only a handful of pigments. |
| **Target color** | A reflectance curve (400–740 nm in 10 nm steps) or CIELAB values. Reflectance is the supported path; pure Lab targets are deprecated. |
| **Predictive model** | A pre-trained pipeline (not included) mapping a recipe to its predicted reflectance/Lab **and** an uncertainty estimate. |
| **Color difference** | Recipes are scored with the CIE 1994 color difference, ΔE\*<sub>94</sub>, against the target. |
| **Metamerism** | Colors are compared under three illuminants — D65, FL2, Apple Studio LED. The score is the weighted average `0.4·D65 + 0.3·FL2 + 0.3·StudioLED`; the spread across illuminants is reported as a metamerism index (`dEm`). |

---

## `GA/` – Genetic Algorithm

| File | Purpose |
|------|---------|
| `GA_scripts.py` | The algorithm. |
| `utils.py` | Recipe normalization, `recipe_to_lab`, and results / metrics aggregation over saved runs. |
| `GA.ipynb` | Driver notebook: load models + data, call `run_ga`, summarize. |

**Key functions in `GA_scripts.py`:**

- **`fitness(individuals, expectation, model, …)`** — predicts each recipe's color
  with `model`, computes the illuminant-weighted ΔE94 to the target, and combines
  three terms:

  | Term | Formula | Meaning |
  |------|---------|---------|
  | `delta_fitness` | `exp(-0.5 · ΔE94)` | closeness to the target color |
  | `ingredients_fitness` | `1 − 0.008 · n_pigments²` (0 above 10 pigments) | sparsity |
  | `incertitude_fitness` | `1 − 0.05 · uncertainty²` (0 above 4.5) | model confidence |

  Recipes whose concentrations sum to > 1 are killed (fitness 0). Returns fitness,
  ΔE94, pigment count, model uncertainty, and metamerism `dEm`.

- **`create_population(...)`** — random initial recipes, masked by each pigment's
  historical occurrence rate; supports `forced_columns` / `unused_columns`.
- **`tournament_selection(...)`** — block tournament selection; also tracks the
  global best.
- **`crossover(...)`** — arithmetic crossover on positions where at least one
  parent is non-zero, then `enforce_sparse_limit` caps the number of active
  pigments.
- **`mutate(...)`** — multiplicative mutation with a range that anneals toward 1.0
  over generations, plus random pigment drop / random new pigment
  (Beta-distributed concentration), respecting mandatory / forbidden pigments and
  per-pigment caps.
- **`run_ga(...)`** — the loop: *select → crossover → mutate → elitist
  recombination*, with optional early stopping, fitness plots, CIELAB
  visualizations, and saving of hyperparameters, per-generation best recipes, and
  summary metrics. `.notdone` / `.done` marker files track run completion.

---

## `NN/` – Neural Network

| File | Purpose |
|------|---------|
| `NN_tuner.py` | Data loading, model definition, training, evaluation, and a random hyperparameter search harness. |
| `pigment_prediction.ipynb` | Runs the tuning / summarization. |
| `NN_result_analysis.ipynb` | Deeper analysis and re-evaluation of trained models. |

**Task:** multi-label classification. Input `X` = target color (L, a, b); labels
`y` = pigment presence (`y > 0` binarized to 0/1). It predicts the *set* of
pigments a recipe should use — not their concentrations.

- **`load_data()`** — reads the `Dataset/traintest` CSVs, binarizes `y`.
- **`train_model(...)`** — Keras `Sequential`: a `Dense(relu)` stack with
  `BatchNormalization` after the first layer and `Dropout` between layers, sigmoid
  output. Adam optimizer with **binary focal loss** (`gamma=2.0, alpha=0.8`) to
  handle heavy class imbalance. Early stopping on `val_loss`.
- **`evaluate_model(...)`** — thresholds probabilities (default 0.4), saves
  predictions, computes precision / recall / binary accuracy, a per-recipe
  comparison of expected vs. predicted pigment sets (`compare_binary_dataframes`),
  and per-pigment confidence stats (`compute_confidence`).
- **`run_NN_tuning(...)`** — trains many models over manual + randomly sampled
  hyperparameter combinations, each in its own folder with a `.done` marker;
  `results_summarizer` / `re_evaluate_models` aggregate them.

---

## `GANN/` – Genetic Algorithm + Neural Network

| File | Purpose |
|------|---------|
| `ga_nn.py` | End-to-end pipeline script. |
| `ga_nn_notebook.ipynb` | The same pipeline plus an evaluation section that aggregates runs and builds performance plots. |

**Idea:** run the NN first to predict which pigments a target color needs, then run
the GA (`GA.GA_scripts.run_ga`) with the search space shaped by that prediction.
Narrowing the GA to a promising subset of pigments is meant to make the search
faster and the recipes more realistic.

**Pipeline:**

1. **Load models** — the trained NN pigment classifier (`model.keras`, with the
   custom `binary_focal_loss`) and the recipe → reflectance model
   (`pipeline_dict_file['model']['curve']`).
2. **Predict pigments** — run the NN over every target color `X` for per-pigment
   probabilities (`NN_result_df`); threshold at `THRESHOLD` (0.4) for the binary
   pigment set (`NN_result_bin_df`).
3. **Iterate targets** — loop over the recipes in `X` in strides of `step` (10),
   skipping targets that use ≤ 1 pigment and any already carrying a `.done`
   marker. For each target, `columns_to_use` is the set of NN-predicted pigments.
4. **Variants** — two switches produce the labelled sub-runs (`sp-co`, `sp`, `co`,
   `-`); the script currently executes the strict / confidence combination:
   - `STRICT_PIGMENT_USE` — restrict the GA to only the NN-predicted pigments
     (pass the reduced `max_pigment_values`), versus allowing all pigments.
   - `CONFIDENCE_OCCURENCE` — seed the population from the NN's prediction
     probabilities (`NN_result_df.loc[i, columns_to_use]`) instead of the
     historical occurrence rates (`(y > 0).sum() / len(y)`).
5. **Run** — each variant calls `run_ga` three times (200 generations,
   population 300, tournament 30, `mutation_rate=0.5`, early stopping), writing
   results into a per-variant folder; a `.done` marker is written per target.
6. **Summarize** — `summarize_results_GANN` / `metrics_counter` roll every run up
   into `summary.csv` / `metrics.json`; the notebook then plots fitness-bucket
   counts and ΔE94 buckets against pigment count, fitness, and metamerism.

---

## `Utils/` – Shared helpers

**`util_methods.py`** — the `UtilMethods` class:

- `find_project_root` — locate the `.rootfolder` marker.
- `select_prescriptive_x_y`, `divide_train_and_test_data` (KMeans-cluster-aware
  split), `binarize_dataset`, `normalize_recipes`.
- `CalculateLab` / `getLab` / `addLabcols` — reflectance → CIELAB via the `colour`
  library, including a registered **Apple Studio LED** illuminant.
- `visualize_lab` — 2D / interactive / 3D CIELAB scatter plots (matplotlib / plotly).

**Notebooks:**

- `read_recipes.ipynb` — build the train/test/val splits from the raw recipe data
  (split by process card or randomly; optionally drop rarely used pigments).
- `spectrum_viewer.ipynb` — plot reflectance spectra for selected rows.

---

## Workflow

**Step 0 — data.** `Utils/read_recipes.ipynb` prepares the `Dataset/traintest/*`
splits.

**Track A — GA only** (independent of the NN):

- `GA/GA.ipynb` — run the GA on target colors. Uses only the recipe → color
  predictive model; **no trained NN is required.**

**Track B — NN-guided GA** (pigment prediction on top):

1. `NN/pigment_prediction.ipynb` — tune and train the pigment-prediction network.
2. `NN/NN_result_analysis.ipynb` — pick the best model.
3. `GANN/ga_nn_notebook.ipynb` — run the GA seeded / constrained by that NN's
   prediction.

---

## Dependencies

`numpy` · `pandas` · `scikit-learn` · `scikit-image` · `scipy` · `matplotlib` ·
`plotly` · `colour-science` · `tensorflow` / `keras` · `python-dotenv` · `tqdm`

---

## Citation

If you use or refer to this code, please cite:

```bibtex
@article{barta_ga_color_recipe,
  title  = {Genetic Algorithm for Color Recipe Prediction in Industrial Settings},
  author = {Lazar Barta, Willem Godlieb, Yusuf Can Semerci, Anna Wilbik, Marcin Pietrasik},
  year = {2026}
}
```
## Contact
[ld.barta@alumni.maastrichtuniversity.nl](mailto:ld.barta@alumni.maastrichtuniversity.nl)