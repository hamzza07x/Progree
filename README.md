# Data Science Mini-Projects: Scraping, Segmentation, and Forecasting

Three end-to-end Python projects that follow one data science workflow: **collect** raw data (Task 2), **find structure** in it (Task 3), and **forecast and explain** it (Task 4). Each project is a self-contained Jupyter notebook with step-by-step Markdown explanations, readable code, and saved outputs.

---

## Table of Contents

1. [Projects at a Glance](#1-projects-at-a-glance)
2. [Repository Structure](#2-repository-structure)
3. [Installation and Setup](#3-installation-and-setup)
4. [Task 2: Web Scraping and Data Extraction Pipeline](#4-task-2-web-scraping-and-data-extraction-pipeline)
5. [Task 3: Customer Segmentation (Clustering)](#5-task-3-customer-segmentation-clustering)
6. [Task 4: Time-Series Forecasting and Interpretability](#6-task-4-time-series-forecasting-and-interpretability)
7. [Concepts Explained](#7-concepts-explained)
8. [Reproducibility](#8-reproducibility)
9. [Limitations and Honest Caveats](#9-limitations-and-honest-caveats)
10. [Troubleshooting](#10-troubleshooting)
11. [Tech Stack](#11-tech-stack)
12. [Author and License](#12-author-and-license)

---

## 1. Projects at a Glance

| Task | Problem | Core techniques | Output |
|---|---|---|---|
| **2. Web scraping pipeline** | Turn messy web listings into a clean, structured dataset | `requests`, BeautifulSoup, regex cleaning, deduplication | `books.csv`, `books.db` |
| **3. Customer segmentation** | Group customers by purchasing behavior with no labels | Log transform, `StandardScaler`, PCA, Elbow Method, K-Means | `customer_segments.csv` |
| **4. Forecasting + interpretability** | Forecast a climate series and explain what drives predictions | ADF/KPSS tests, seasonal differencing, SARIMA, Gradient Boosting, SHAP | `figures/` folder (9 plots, 3 tables) |

### How the three tasks connect

```
 Task 2                    Task 3                         Task 4
 ┌──────────────┐         ┌───────────────────┐          ┌──────────────────────┐
 │ Raw web HTML │  clean  │ Customer metrics  │ segment  │ Historical time      │
 │ → structured │ ──────► │ → scaled → PCA    │ ───────► │ series → stationary  │
 │ CSV / SQLite │  data   │ → K-Means groups  │  insight │ → SARIMA → SHAP      │
 └──────────────┘         └───────────────────┘          └──────────────────────┘
   COLLECT                    DISCOVER                       PREDICT + EXPLAIN
```

The tasks use three different datasets. The link between them is the workflow, not shared data.

---

## 2. Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── task2_web_scraping.ipynb           # Task 2 notebook
├── task3_customer_segmentation.ipynb  # Task 3 notebook
├── task4_forecasting_shap.ipynb       # Task 4 notebook
│
│   ── Generated when the notebooks run ──
├── books.csv                          # Task 2 output (flat file)
├── books.db                           # Task 2 output (SQLite database)
├── customer_segments.csv              # Task 3 output (customers + segment label)
└── figures/                           # Task 4 outputs
    ├── 01_raw_series.png
    ├── 02_decomposition.png
    ├── 03_acf_pacf.png
    ├── 04_residual_diagnostics.png
    ├── 05_sarima_test_forecast.png
    ├── 06_future_forecast.png
    ├── 07_shap_bar.png
    ├── 08_shap_beeswarm.png
    ├── 09_shap_waterfall.png
    ├── stationarity_table.csv
    ├── model_scores.csv
    └── shap_importance.csv
```

Rename the notebook files to match your actual file names if they differ.

---

## 3. Installation and Setup

### Requirements
- Python 3.10 or newer
- An internet connection for Task 2 (scraping) and Task 3 (dataset download)
- Task 4 needs no internet. Its dataset ships inside `statsmodels`.

### Steps

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd <repository-folder>

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### `requirements.txt`

```
requests>=2.31
beautifulsoup4>=4.12
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
statsmodels>=0.14
shap>=0.45
jupyter>=1.0
```

### Running order
The three notebooks are independent, so run them in any order. **Inside each notebook, run the cells top to bottom** (or use *Kernel → Restart & Run All*). Many cells depend on variables created in earlier cells.

---

## 4. Task 2: Web Scraping and Data Extraction Pipeline

### Goal
Build an automated pipeline that harvests raw listings from a public website, extracts the key fields with text-filtering logic, cleans whitespace artifacts, and stores the result in a CSV file and a database.

### Data source
[books.toscrape.com](https://books.toscrape.com) is a public sandbox website built specifically for scraping practice, so scraping it is explicitly allowed. It lists 1,000 books across 50 pages.

### Pipeline

```
 Fetch page ──► Parse HTML ──► Extract fields ──► Clean text ──► Deduplicate ──► Save
 (requests)     (BeautifulSoup)   (regex + class     (whitespace,   (by URL)       (CSV +
                                   parsing)          encoding)                     SQLite)
```

| Stage | What it does | Why it matters |
|---|---|---|
| **Fetch** | Downloads each page with up to 3 retries and exponential backoff | Networks fail occasionally, and one failed request should not kill the run |
| **Parse** | Finds every `article.product_pod` block on the page | Isolates one listing at a time |
| **Extract** | Pulls title, price, rating, stock status, stock quantity, and URL | Regex handles text like `In stock (22 available)`, and class names like `star-rating Three` are converted to numbers |
| **Clean** | Replaces `\xa0`, removes zero-width characters, collapses repeated whitespace, forces UTF-8 | Prevents artifacts such as `Â£` and stray newlines in the data |
| **Deduplicate** | Tracks seen URLs in a `set` | Pagination overlaps would otherwise create duplicate rows |
| **Save** | Writes `books.csv` and a SQLite table `books` with `url` as primary key | Flat file for quick use, database for querying |

### Extracted fields

| Column | Type | Example | How it is extracted |
|---|---|---|---|
| `title` | text | `A Light in the Attic` | `title` attribute of the link (the visible text is truncated) |
| `price_gbp` | float | `51.77` | Regex finds the first number in `£51.77` |
| `rating` | int (1-5) | `3` | Word in the CSS class (`Three`) mapped to a number |
| `in_stock` | bool | `True` | Looks for the phrase `in stock` |
| `stock_qty` | int | `22` | Regex `\((\d+)\s*available\)` |
| `url` | text | `https://books.toscrape.com/catalogue/...` | Relative link converted to absolute with `urljoin` |

### Politeness and safety
- A configurable delay (`delaySeconds`) separates requests so the server is not hammered.
- A descriptive `User-Agent` header identifies the script.
- `maxPages = 5` is the default for quick testing. Set `maxPages = 0` to scrape all 50 pages.

### Results

> Fill in after running with `maxPages = 0`.

| Metric | Value |
|---|---|
| Pages scraped | [N] |
| Records collected | [N] |
| Duplicate URLs | [0] |
| Missing values | [0] |

---

## 5. Task 3: Customer Segmentation (Clustering)

### Goal
Group customers by purchasing behavior without any predefined labels, then describe each group so it can be used for business decisions.

### Dataset
**Wholesale Customers** from the UCI Machine Learning Repository: 440 customers, with yearly spending on six product categories.

| Column | Meaning | Used for clustering |
|---|---|---|
| `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen` | Annual spending per category | Yes |
| `Channel`, `Region` | Category codes (Horeca/Retail, region ID) | No, these are labels, not spending |

### Pipeline

```
 Load ──► Log transform ──► StandardScaler ──► PCA ──► Elbow Method ──► K-Means ──► Profile
                (skew)        (equal scale)   (reduce)   (choose k)      (cluster)   (interpret)
```

### Step-by-step reasoning

**1. Log transform.** Spending data is heavily right-skewed: a few very large customers dwarf the rest. K-Means uses distances, so those outliers would pull cluster centers toward themselves. `log1p` compresses large values. The notebook prints skewness before and after so you can verify the effect.

**2. Standardization.** `StandardScaler` rescales every feature to mean 0 and standard deviation 1. Without it, a feature measured in larger numbers would dominate the distance calculation.

**3. PCA.** The six spending columns are correlated, because customers who buy more Grocery often buy more Detergents_Paper. PCA rewrites them as uncorrelated components ordered by variance. The notebook keeps the smallest number of components that explains at least **85%** of total variance. A loadings table shows which original columns drive each component.

**4. Elbow Method.** K-Means is run for k = 1 to 10. Inertia (within-cluster sum of squares) always decreases as k grows, so the useful signal is where the curve bends: the point after which extra clusters add little. The elbow is located automatically as the point farthest from the straight line joining the first and last points.

**5. Silhouette cross-check.** Reading an elbow by eye is subjective, so the silhouette score is plotted alongside as a second opinion. If the two methods disagree, that disagreement should be reported, not hidden.

**6. K-Means.** The final model is fitted on the PCA-reduced data with `n_init=10`, which runs 10 random starts and keeps the best, avoiding poor initialization.

**7. Segment profiling.** Cluster numbers mean nothing by themselves. The notebook computes average spending per segment in original units and a heatmap of spending relative to the overall average (1.0 = average, above 1.0 = over-spends).

### Results

> Fill in after running on the real dataset.

| Metric | Value |
|---|---|
| Components kept / variance explained | [n] / [X]% |
| Optimal k (Elbow) | [k] |
| Silhouette score | [value] |
| Segment sizes | [e.g. 210 / 130 / 100] |

| Segment | Profile (from the heatmap) | Possible business use |
|---|---|---|
| 0 | [e.g. high Grocery and Detergents_Paper, low Fresh] | [e.g. bulk retail promotions] |
| 1 | [...] | [...] |

### Output
`customer_segments.csv`: the original customer rows plus a `Segment` column.

---

## 6. Task 4: Time-Series Forecasting and Interpretability

### Goal
Fit a seasonal time-series model to a historical climate series, forecast future values with confidence intervals, and quantify which variables drive predictions using SHAP.

### Dataset
Monthly atmospheric CO2 concentration (ppm) measured at Mauna Loa, March 1958 to December 2001 (526 months), built from weekly readings. The data is bundled with `statsmodels`, so no download is needed. The series has a strong upward trend and a clear yearly cycle.

### Pipeline

```
 Load ► Stationarity tests ► Differencing ► ACF/PACF ► SARIMA grid search ► Diagnostics
                                                                                │
        SHAP ◄── Gradient Boosting on lag features ◄── Compare models ◄── Forecast + 95% CI
```

### Step 1: Stationarity analysis
A stationary series has a constant mean and variance over time. SARIMA needs the differenced series to be stationary. Two tests are used because they have opposite null hypotheses:

| Test | Null hypothesis | Small p-value means |
|---|---|---|
| ADF | Series is non-stationary | Stationary |
| KPSS | Series is stationary | Non-stationary |

A verdict is trusted only when both tests agree.

| Series | ADF p | KPSS p | Verdict |
|---|---|---|---|
| Original | 0.9989 | 0.01 | Not stationary |
| First difference (d=1) | 0.0001 | 0.10 | Stationary |
| Seasonal difference (D=1, lag 12) | 0.0005 | 0.01 | Tests disagree |
| Both (d=1, D=1) | 0.0000 | 0.10 | Stationary |

**Reading this table:** the raw series is clearly non-stationary. Seasonal differencing alone is not enough because a trend remains (the tests disagree). The notebook uses both differences. Note that `statsmodels` clips KPSS p-values to the range 0.01 to 0.10, so "0.10" means "at least 0.10".

### Step 2: SARIMA model selection
SARIMA is written as **SARIMA(p, d, q)(P, D, Q, s)**:

| Symbol | Meaning | Chosen value |
|---|---|---|
| p, q | Non-seasonal autoregressive and moving-average orders | 0, 1 |
| d | Non-seasonal differencing | 1 |
| P, Q | Seasonal autoregressive and moving-average orders | 1, 1 |
| D | Seasonal differencing | 1 |
| s | Season length in months | 12 |

A grid of 36 combinations (p, q in 0-2, P, Q in 0-1) was fitted on the training data, and the lowest AIC won: **SARIMA(0,1,1)(1,1,1,12)**, AIC = 207.7.

**Train/test split:** the last 24 months (January 2000 to December 2001) are held out and never used during model selection. Training covers March 1958 to December 1999.

### Step 3: Residual diagnostics
If the model captured the structure, the residuals should look like random noise. The notebook produces the standard four-panel diagnostic plot and runs the **Ljung-Box test**, which checks for leftover autocorrelation.

| Lags | Ljung-Box p-value | Interpretation |
|---|---|---|
| 12 | 0.62 | No evidence of leftover autocorrelation |
| 24 | 0.63 | No evidence of leftover autocorrelation |

### Step 4: Forecasts with confidence intervals
- **Test-period forecast:** 24 months ahead in one go, with a 95% confidence interval. All 24 actual values fell inside the interval.
- **Future forecast:** the model is refitted on all data and projects 36 months beyond the end of the series, with a 95% interval that widens as uncertainty accumulates.

### Step 5: Model comparison

| Model | Forecast type | MAE (ppm) | RMSE (ppm) | MAPE (%) |
|---|---|---|---|---|
| Seasonal naive (baseline) | Multi-step | 1.314 | 1.374 | 0.355 |
| SARIMA | Multi-step, with 95% CI | 0.357 | 0.431 | 0.096 |
| SARIMA | One-step-ahead | 0.210 | 0.272 | 0.057 |
| Gradient Boosting | One-step-ahead | 0.314 | 0.414 | 0.085 |

**How to read it:**
- The **seasonal naive baseline** simply predicts each month using the value from 12 months earlier. Any useful model must beat it, and SARIMA does by a wide margin.
- **Multi-step** forecasts predict all 24 months without seeing any of the test data. **One-step-ahead** forecasts see each new real value as it arrives, which is easier. Compare multi-step with multi-step, and one-step with one-step.
- The only like-for-like comparison between the two models is the **one-step** rows: SARIMA (RMSE 0.272) beats Gradient Boosting (RMSE 0.414).

### Step 6: Interpretability with SHAP

**Why a second model?** SARIMA is built from past values and error terms. It has no feature columns, so SHAP has nothing to attribute. To measure feature importance, a **Gradient Boosting Regressor** is trained on engineered features of the same series, and SHAP explains *that* model.

| Feature | Meaning |
|---|---|
| `YearlyChange_lag1/2/3` | Yearly change (value minus the value 12 months earlier) 1, 2, and 3 months ago |
| `YearlyChange_lag12` | Yearly change 12 months ago |
| `YearlyChange_avg6` | Average yearly change over the previous 6 months |
| `Momentum_lag1` | Last month's movement (value one month ago minus two months ago) |
| `Month_sin`, `Month_cos` | Month of year encoded as a circle, so December and January stay close |

The **target** is the yearly change, which is stationary and easy for trees to learn. Predictions are converted back to CO2 levels by adding the predicted change to last year's value.

**SHAP importance weights** (mean absolute SHAP value on the test set):

| Rank | Feature | Mean \|SHAP\| | Share |
|---|---|---|---|
| 1 | `YearlyChange_lag1` | 0.2143 | 41.3% |
| 2 | `YearlyChange_lag12` | 0.0941 | 18.2% |
| 3 | `YearlyChange_avg6` | 0.0822 | 15.9% |
| 4 | `YearlyChange_lag2` | 0.0516 | 10.0% |
| 5 | `YearlyChange_lag3` | 0.0427 | 8.2% |
| 6 | `Momentum_lag1` | 0.0207 | 4.0% |
| 7 | `Month_sin` | 0.0077 | 1.5% |
| 8 | `Month_cos` | 0.0050 | 1.0% |

**Three plots are produced:**
- **Bar plot:** global importance (the table above, visualized).
- **Beeswarm plot:** shows direction, meaning whether high or low feature values push the prediction up or down.
- **Waterfall plot:** explains one single prediction step by step from the base value.

A sanity check in the notebook confirms that SHAP values plus the base value reproduce the model's predictions (maximum error about 1e-15).

### How to interpret the SHAP result correctly
- The dominance of recent yearly changes reflects **persistence**: CO2 changes smoothly, so last month's yearly change predicts this month's. That is expected, not a surprising discovery.
- The near-zero importance of `Month_sin` and `Month_cos` does **not** mean seasonality is unimportant. The target was already seasonally differenced, so seasonality was removed before SHAP saw the data.

---

## 7. Concepts Explained

| Concept | Plain-language explanation |
|---|---|
| **Regex** | A pattern language for finding text, such as "digits inside brackets followed by the word available" |
| **Deduplication** | Removing repeated records so each real-world item appears once |
| **Skewness** | How lopsided a distribution is. Values near 0 are symmetric, and large positive values mean a long right tail of big outliers |
| **Standardization** | Rescaling each feature to mean 0 and standard deviation 1 so no feature dominates because of its units |
| **PCA** | Rewrites correlated features as a smaller set of uncorrelated components that keep most of the information |
| **Explained variance** | The share of the data's total spread that a principal component captures |
| **K-Means** | Assigns each point to the nearest of k cluster centers and moves the centers until they stop changing |
| **Inertia** | The total squared distance from points to their cluster center. Lower means tighter clusters |
| **Elbow Method** | Plot inertia against k and pick the point where the curve stops dropping sharply |
| **Silhouette score** | From -1 to 1, how well each point fits its own cluster versus the nearest other cluster. Higher is better |
| **Stationarity** | A series whose mean and variance do not change over time |
| **Differencing** | Replacing each value with its change from the previous value (or the value 12 months earlier, for seasonal differencing) |
| **ADF / KPSS** | Two statistical tests for stationarity with opposite null hypotheses |
| **AIC** | A score that rewards fit and penalizes complexity. Lower is better when comparing models on the same data |
| **Confidence interval** | A range expected to contain the true value a stated percentage of the time (95% here) |
| **Ljung-Box test** | Checks whether residuals still contain autocorrelation. A high p-value is the good outcome |
| **Seasonal naive baseline** | Forecast = the value from the same month last year. The minimum bar any model must clear |
| **SHAP** | Splits each prediction into per-feature contributions that add up exactly to the model output |

---

## 8. Reproducibility

- **Random seed:** every random operation uses `randomSeed = 42`.
- **Task 4 needs no network:** the dataset is bundled with `statsmodels`, so results are reproducible on any machine with the same package versions.
- **Tested with:** Python 3.12, numpy 2.4, pandas 3.0, matplotlib 3.10, seaborn 0.13, scikit-learn 1.8, statsmodels 0.15, shap 0.52, requests 2.33, beautifulsoup4 4.14. Small numerical differences can appear with other versions.
- **Task 2 results depend on the live website**, so record counts could change if the site changes.
- **Always use "Restart and Run All"** before trusting results. Running cells out of order is the most common cause of errors and stale outputs.

---

## 9. Limitations and Honest Caveats

### Task 2
- The target site is a practice site with a clean structure. Real websites change layouts, block scrapers, and have terms of service and `robots.txt` rules that must be checked first.
- The scraper is single-threaded. That is fine for 1,000 records, but it will not scale to millions.
- The target is semi-structured HTML, so the text-filtering layer does less work than it would on truly unstructured text.

### Task 3
- K-Means assumes roughly spherical clusters of similar size, and real customer data often is not like that.
- The final k is a judgement call. The elbow and the silhouette score can disagree, and the segments depend on the log transform and the 85% PCA cutoff.
- Clusters describe patterns in the data. They do not prove that the segments are meaningful business categories.

### Task 4
- **SHAP explains the Gradient Boosting model, not SARIMA.** SARIMA has no feature matrix to attribute.
- **The interpretable model is the weaker one.** Gradient Boosting scored RMSE 0.414 versus 0.272 for SARIMA on one-step forecasts, so the importance weights describe a less accurate model.
- **The test set is small.** 24 autocorrelated months are too few to prove the confidence intervals are well calibrated. "All 24 inside the interval" is consistent with good calibration, but it does not demonstrate it.
- **The chosen seasonal AR term is almost zero** (coefficient about -0.0008, p = 0.068). AIC selected it, but the model is practically SARIMA(0,1,1)(0,1,1,12).
- The dataset ends in 2001. Forecasts are an exercise in method, not a current CO2 projection.

---

## 10. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `NameError: name 'dbPath' is not defined` | A cell was run before the cell that defines the variable, or the kernel was restarted | Run cells from the top, or use *Restart & Run All* |
| `NameError: name 'display' is not defined` | Code run outside Jupyter | Add `from IPython.display import display` |
| Task 3 prints "WARNING: using synthetic backup data" | The UCI download failed | Download the *Wholesale customers data* CSV manually and replace the `read_csv` call with the local file path. **Do not report results from the synthetic data.** |
| Task 2 returns no data | Network blocked, or the site layout changed | Check the connection and the log messages. If the site changed, update the CSS selectors in `parsePage` |
| Task 2 text shows `Â£` | Wrong text encoding | Keep `response.encoding = "utf-8"` in `fetchPage` |
| `ModuleNotFoundError: shap` | Missing dependency | `pip install shap` |
| Task 4 grid search is slow | 36 SARIMA fits | Expect about 20-60 seconds. Shrink the grid ranges to speed it up |
| Plots do not appear | Matplotlib backend issue | Add `%matplotlib inline` at the top of the notebook |

---

## 11. Tech Stack

| Area | Libraries |
|---|---|
| Data collection | `requests`, `beautifulsoup4`, `re`, `csv`, `sqlite3` |
| Data handling | `numpy`, `pandas` |
| Machine learning | `scikit-learn` (`StandardScaler`, `PCA`, `KMeans`, `GradientBoostingRegressor`, metrics) |
| Time series | `statsmodels` (`SARIMAX`, `adfuller`, `kpss`, `seasonal_decompose`, ACF/PACF) |
| Interpretability | `shap` (`TreeExplainer`) |
| Visualization | `matplotlib`, `seaborn` |
| Environment | Python 3, Jupyter Notebook |

---

## 12. Author and License

**Author:** [Your Name]
**Contact:** [your email or LinkedIn]
**License:** [e.g. MIT]

Data sources: books.toscrape.com (practice site), UCI Machine Learning Repository (Wholesale Customers), and the Mauna Loa CO2 record bundled with `statsmodels`.