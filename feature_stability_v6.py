
"""
======================================================================
ALPHALENS V6 FEATURE STABILITY & LEAKAGE AUDIT
======================================================================

Purpose:
    1. Load the actual V4 feature dataset.
    2. Identify numeric candidate features.
    3. Check for obvious target leakage.
    4. Check missingness and near-constant features.
    5. Measure feature stability across chronological periods.
    6. Measure univariate relationship with the 10-day target.
    7. Produce SAFE / REVIEW / REMOVE classifications.
    8. Save audit files for the next V6 modelling stage.

Expected files:
    data/ml_features_v4.csv
    data/ml_targets_v3.csv

Outputs:
    data/feature_stability_v6_results.csv
    data/feature_stability_v6_summary.csv
    data/feature_stability_v6_safe_features.csv
    data/feature_stability_v6_review_features.csv
    data/feature_stability_v6_remove_features.csv
======================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

FEATURE_FILE = os.path.join(DATA_DIR, "ml_features_v4.csv")
TARGET_FILE = os.path.join(DATA_DIR, "ml_targets_v3.csv")

RESULTS_FILE = os.path.join(
    DATA_DIR,
    "feature_stability_v6_results.csv"
)

SUMMARY_FILE = os.path.join(
    DATA_DIR,
    "feature_stability_v6_summary.csv"
)

SAFE_FILE = os.path.join(
    DATA_DIR,
    "feature_stability_v6_safe_features.csv"
)

REVIEW_FILE = os.path.join(
    DATA_DIR,
    "feature_stability_v6_review_features.csv"
)

REMOVE_FILE = os.path.join(
    DATA_DIR,
    "feature_stability_v6_remove_features.csv"
)


# Number of chronological periods used for stability testing.
N_PERIODS = 5

# Missingness above this level is considered problematic.
MAX_MISSING_RATE = 0.30

# A feature with >=99.5% identical values is effectively constant.
MAX_CONSTANT_RATE = 0.995

# Minimum absolute AUC deviation from 0.50 to count as meaningful
# for the univariate screening.
MIN_AUC_SIGNAL = 0.55

# Minimum number of valid observations needed for AUC calculation.
MIN_VALID_AUC_ROWS = 100


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def normalize_date_column(df):
    """
    Find a date-like column and convert it to datetime.
    """
    candidates = [
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Timestamp",
        "timestamp"
    ]

    for col in candidates:
        if col in df.columns:
            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            if converted.notna().sum() > 0:
                df[col] = converted
                return df, col

    # Fallback: inspect object columns.
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            if converted.notna().sum() >= max(
                10,
                int(len(df) * 0.50)
            ):
                df[col] = converted
                return df, col

    return df, None


def find_target_column(df):
    """
    Find the most appropriate binary target column.
    Preference is given to the 10-day 1% target.
    """

    preferred = [
        "Target_10D_1pct",
        "Target_10D_1Pct",
        "Target_10D_1.0pct",
        "Target_10D_1%",
        "Return_AtLeast_1.0pct",
        "Return_AtLeast_1pct",
        "Target"
    ]

    for col in preferred:
        if col in df.columns:
            values = pd.to_numeric(
                df[col],
                errors="coerce"
            ).dropna()

            unique = set(values.unique())

            if len(unique) == 2:
                return col

    # Search all columns for binary target-like fields.
    target_candidates = []

    for col in df.columns:
        name = col.lower()

        if any(
            word in name
            for word in [
                "target",
                "return_atleast",
                "future",
                "forward"
            ]
        ):
            values = pd.to_numeric(
                df[col],
                errors="coerce"
            ).dropna()

            if len(values) > 0 and len(values.unique()) == 2:
                target_candidates.append(col)

    if target_candidates:
        return target_candidates[0]

    return None


def safe_auc(y, x):
    """
    Calculate AUC for a single feature.

    We use max(AUC, 1-AUC) so that both positive and negative
    monotonic relationships are treated as signal strength.
    """

    mask = (
        pd.notna(y)
        & pd.notna(x)
    )

    if mask.sum() < MIN_VALID_AUC_ROWS:
        return np.nan

    yy = pd.to_numeric(
        y[mask],
        errors="coerce"
    )

    xx = pd.to_numeric(
        x[mask],
        errors="coerce"
    )

    valid = (
        yy.notna()
        & xx.notna()
    )

    yy = yy[valid]
    xx = xx[valid]

    if len(yy) < MIN_VALID_AUC_ROWS:
        return np.nan

    if yy.nunique() != 2:
        return np.nan

    if xx.nunique() < 2:
        return np.nan

    try:
        auc = roc_auc_score(yy, xx)

        if not np.isfinite(auc):
            return np.nan

        return max(float(auc), 1.0 - float(auc))

    except Exception:
        return np.nan


def constant_rate(series):
    """
    Percentage of observations belonging to the most common value.
    """

    s = series.dropna()

    if len(s) == 0:
        return 1.0

    counts = s.value_counts(dropna=False)

    return float(counts.iloc[0] / len(s))


def stability_score(values):
    """
    Convert period AUC values into a simple stability score.

    Score interpretation:
        1.00 = highly stable
        0.50 = weak / mixed
        0.00 = unusable

    This is a screening score, not a profitability score.
    """

    vals = np.array(
        [v for v in values if np.isfinite(v)],
        dtype=float
    )

    if len(vals) == 0:
        return np.nan

    # Distance from random AUC of 0.50.
    signal = np.abs(vals - 0.50)

    mean_signal = float(signal.mean())

    # Consistency across periods.
    positive_signal = np.mean(signal >= 0.05)

    # Penalise high dispersion.
    dispersion = float(np.std(vals))

    raw = (
        0.60 * min(mean_signal / 0.15, 1.0)
        + 0.30 * positive_signal
        + 0.10 * max(
            0.0,
            1.0 - min(dispersion / 0.20, 1.0)
        )
    )

    return float(np.clip(raw, 0.0, 1.0))


def classify_feature(
    leakage_flag,
    missing_rate,
    constant_rate_value,
    mean_auc,
    stability
):
    """
    Conservative feature classification.

    REMOVE:
        obvious leakage / unusable feature

    REVIEW:
        suspicious, unstable or weak-quality feature

    SAFE:
        passes basic screening
    """

    if leakage_flag:
        return "REMOVE"

    if missing_rate >= MAX_MISSING_RATE:
        return "REMOVE"

    if constant_rate_value >= MAX_CONSTANT_RATE:
        return "REMOVE"

    if not np.isfinite(mean_auc):
        return "REVIEW"

    if not np.isfinite(stability):
        return "REVIEW"

    if stability < 0.35:
        return "REVIEW"

    return "SAFE"


# ============================================================
# START
# ============================================================

print_header(
    "ALPHALENS V6 FEATURE STABILITY & LEAKAGE AUDIT"
)

print()
print("Loading V4 features...")


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(
        "\nCould not find:\n"
        f"    {FEATURE_FILE}\n\n"
        "Available files should include:\n"
        "    data/ml_features_v4.csv\n"
    )

if not os.path.exists(TARGET_FILE):
    raise FileNotFoundError(
        "\nCould not find:\n"
        f"    {TARGET_FILE}\n\n"
        "Available files should include:\n"
        "    data/ml_targets_v3.csv\n"
    )


# ============================================================
# LOAD DATA
# ============================================================

features = pd.read_csv(FEATURE_FILE)
targets = pd.read_csv(TARGET_FILE)

print(f"Feature rows    : {len(features)}")
print(f"Feature columns : {len(features.columns)}")

print(f"Target rows     : {len(targets)}")
print(f"Target columns  : {len(targets.columns)}")


# ============================================================
# DATE HANDLING
# ============================================================

features, feature_date_col = normalize_date_column(features)
targets, target_date_col = normalize_date_column(targets)

print()
print("DATE COLUMNS")
print("-" * 70)
print(f"Feature date column : {feature_date_col}")
print(f"Target date column  : {target_date_col}")


# ============================================================
# IDENTIFY TARGET
# ============================================================

target_col = find_target_column(targets)

if target_col is None:
    raise ValueError(
        "\nCould not identify a binary target column in:\n"
        f"{TARGET_FILE}\n\n"
        "Expected something similar to:\n"
        "Target_10D_1pct\n"
    )

print()
print("TARGET")
print("-" * 70)
print(f"Target column : {target_col}")


# ============================================================
# PREPARE MERGED DATA
# ============================================================

if (
    feature_date_col is not None
    and target_date_col is not None
):
    feature_date = features[feature_date_col]
    target_date = targets[target_date_col]

    # If dates are unique, use a date merge.
    if feature_date.notna().sum() > 0:
        if target_date.notna().sum() > 0:

            feature_dates_unique = (
                feature_date.dropna().nunique()
                == len(feature_date.dropna())
            )

            target_dates_unique = (
                target_date.dropna().nunique()
                == len(target_date.dropna())
            )

            if feature_dates_unique and target_dates_unique:

                target_subset = targets[
                    [target_date_col, target_col]
                ].copy()

                target_subset = target_subset.rename(
                    columns={
                        target_date_col: "__MERGE_DATE__"
                    }
                )

                features = features.copy()

                features["__MERGE_DATE__"] = (
                    features[feature_date_col]
                )

                merged = features.merge(
                    target_subset,
                    on="__MERGE_DATE__",
                    how="inner"
                )

                merged = merged.drop(
                    columns=["__MERGE_DATE__"]
                )

            else:
                merged = None
        else:
            merged = None
    else:
        merged = None
else:
    merged = None


# ============================================================
# FALLBACK ALIGNMENT
# ============================================================

if merged is None or len(merged) < 100:

    print()
    print(
        "Date merge unavailable or too small."
    )
    print(
        "Using chronological row alignment."
    )

    n = min(
        len(features),
        len(targets)
    )

    merged = features.iloc[:n].copy()

    target_values = targets[target_col].iloc[:n].values

    merged["__TARGET__"] = target_values

else:

    merged["__TARGET__"] = merged[target_col].values

    if target_col in merged.columns:
        merged = merged.drop(
            columns=[target_col]
        )


# ============================================================
# CLEAN TARGET
# ============================================================

merged["__TARGET__"] = pd.to_numeric(
    merged["__TARGET__"],
    errors="coerce"
)

merged = merged[
    merged["__TARGET__"].isin([0, 1])
].copy()

merged = merged.reset_index(drop=True)


print()
print("MERGED DATASET")
print("-" * 70)
print(f"Rows : {len(merged)}")

print()
print("TARGET DISTRIBUTION")
print(merged["__TARGET__"].value_counts())

if len(merged) == 0:
    raise ValueError(
        "No valid binary target rows remain."
    )


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

date_col = None

for col in merged.columns:
    if col.lower() in [
        "date",
        "datetime",
        "timestamp"
    ]:
        if np.issubdtype(
            merged[col].dtype,
            np.datetime64
        ):
            date_col = col
            break

if date_col is not None:
    merged = merged.sort_values(
        date_col
    ).reset_index(drop=True)


# ============================================================
# IDENTIFY NUMERIC FEATURES
# ============================================================

exclude_columns = {
    "__TARGET__"
}

if date_col is not None:
    exclude_columns.add(date_col)

# Explicit target-like columns are excluded from candidates.
for col in merged.columns:

    lower = col.lower()

    if any(
        key in lower
        for key in [
            "target",
            "future_return",
            "forward_return",
            "future_close",
            "future_high",
            "future_low"
        ]
    ):
        exclude_columns.add(col)


numeric_features = []

for col in merged.columns:

    if col in exclude_columns:
        continue

    if pd.api.types.is_numeric_dtype(
        merged[col]
    ):
        numeric_features.append(col)


print()
print("FEATURE SET")
print("-" * 70)
print(f"Numeric candidate features : {len(numeric_features)}")


if len(numeric_features) == 0:
    raise ValueError(
        "No numeric candidate features were found."
    )


# ============================================================
# CREATE CHRONOLOGICAL PERIODS
# ============================================================

merged["__PERIOD__"] = pd.qcut(
    np.arange(len(merged)),
    q=N_PERIODS,
    labels=False,
    duplicates="drop"
)

period_count = int(
    merged["__PERIOD__"].nunique()
)

print()
print("CHRONOLOGICAL PERIODS")
print("-" * 70)
print(f"Periods : {period_count}")


# ============================================================
# AUDIT FEATURES
# ============================================================

print()
print("RUNNING FEATURE AUDIT...")
print("-" * 70)

audit_rows = []


for idx, feature in enumerate(
    numeric_features,
    start=1
):

    series = pd.to_numeric(
        merged[feature],
        errors="coerce"
    )

    missing_rate = float(
        series.isna().mean()
    )

    const_rate = constant_rate(
        series
    )

    unique_count = int(
        series.nunique(dropna=True)
    )

    # --------------------------------------------------------
    # BASIC LEAKAGE NAME CHECK
    # --------------------------------------------------------

    lower_name = feature.lower()

    leakage_name_flag = any(
        keyword in lower_name
        for keyword in [
            "target",
            "future_return",
            "forward_return",
            "future_close",
            "future_high",
            "future_low",
            "future_open",
            "future_volume"
        ]
    )

    # --------------------------------------------------------
    # FULL SAMPLE AUC
    # --------------------------------------------------------

    full_auc = safe_auc(
        merged["__TARGET__"],
        series
    )

    # --------------------------------------------------------
    # PERIOD AUC
    # --------------------------------------------------------

    period_aucs = []

    for period in sorted(
        merged["__PERIOD__"].dropna().unique()
    ):

        part = merged[
            merged["__PERIOD__"] == period
        ]

        auc = safe_auc(
            part["__TARGET__"],
            part[feature]
        )

        period_aucs.append(
            auc
        )

    valid_period_aucs = [
        x
        for x in period_aucs
        if np.isfinite(x)
    ]

    if valid_period_aucs:

        mean_period_auc = float(
            np.mean(valid_period_aucs)
        )

        median_period_auc = float(
            np.median(valid_period_aucs)
        )

        std_period_auc = float(
            np.std(valid_period_aucs)
        )

        min_period_auc = float(
            np.min(valid_period_aucs)
        )

        max_period_auc = float(
            np.max(valid_period_aucs)
        )

        stability = stability_score(
            valid_period_aucs
        )

        periods_above_055 = int(
            sum(
                x >= MIN_AUC_SIGNAL
                for x in valid_period_aucs
            )
        )

    else:

        mean_period_auc = np.nan
        median_period_auc = np.nan
        std_period_auc = np.nan
        min_period_auc = np.nan
        max_period_auc = np.nan
        stability = np.nan
        periods_above_055 = 0

    # --------------------------------------------------------
    # CORRELATION WITH TARGET
    # --------------------------------------------------------

    valid = (
        series.notna()
        & merged["__TARGET__"].notna()
    )

    if valid.sum() >= MIN_VALID_AUC_ROWS:

        try:

            correlation = float(
                np.corrcoef(
                    series[valid],
                    merged.loc[
                        valid,
                        "__TARGET__"
                    ]
                )[0, 1]
            )

            if not np.isfinite(correlation):
                correlation = np.nan

        except Exception:

            correlation = np.nan

    else:

        correlation = np.nan

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    classification = classify_feature(
        leakage_flag=leakage_name_flag,
        missing_rate=missing_rate,
        constant_rate_value=const_rate,
        mean_auc=mean_period_auc,
        stability=stability
    )

    audit_rows.append({

        "Feature": feature,

        "Data_Type": str(
            merged[feature].dtype
        ),

        "Unique_Values": unique_count,

        "Missing_Rate": missing_rate,

        "Constant_Rate": const_rate,

        "Full_Sample_AUC": full_auc,

        "Mean_Period_AUC": mean_period_auc,

        "Median_Period_AUC": median_period_auc,

        "Std_Period_AUC": std_period_auc,

        "Min_Period_AUC": min_period_auc,

        "Max_Period_AUC": max_period_auc,

        "Periods_AUC_GE_0.55": periods_above_055,

        "Stability_Score": stability,

        "Target_Correlation": correlation,

        "Leakage_Name_Flag": leakage_name_flag,

        "Classification": classification
    })

    if (
        idx == 1
        or idx % 25 == 0
        or idx == len(numeric_features)
    ):
        print(
            f"Processed {idx:>4}/{len(numeric_features)}"
        )


# ============================================================
# AUDIT DATAFRAME
# ============================================================

audit = pd.DataFrame(
    audit_rows
)


# ============================================================
# SORT
# ============================================================

classification_order = {
    "REMOVE": 0,
    "REVIEW": 1,
    "SAFE": 2
}

audit["_class_order"] = (
    audit["Classification"]
    .map(classification_order)
)

audit = audit.sort_values(
    by=[
        "_class_order",
        "Stability_Score"
    ],
    ascending=[
        True,
        False
    ]
).drop(
    columns=["_class_order"]
).reset_index(drop=True)


# ============================================================
# COUNTS
# ============================================================

safe_count = int(
    (audit["Classification"] == "SAFE").sum()
)

review_count = int(
    (audit["Classification"] == "REVIEW").sum()
)

remove_count = int(
    (audit["Classification"] == "REMOVE").sum()
)

leakage_features = audit[
    audit["Leakage_Name_Flag"] == True
]["Feature"].tolist()

high_missing = audit[
    audit["Missing_Rate"] >= MAX_MISSING_RATE
]["Feature"].tolist()

high_constant = audit[
    audit["Constant_Rate"] >= MAX_CONSTANT_RATE
]["Feature"].tolist()


# ============================================================
# SAVE MAIN RESULTS
# ============================================================

audit.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# SAVE SAFE FEATURES
# ============================================================

safe_features = audit[
    audit["Classification"] == "SAFE"
][[
    "Feature",
    "Full_Sample_AUC",
    "Mean_Period_AUC",
    "Median_Period_AUC",
    "Std_Period_AUC",
    "Stability_Score",
    "Missing_Rate",
    "Constant_Rate"
]].copy()

safe_features.to_csv(
    SAFE_FILE,
    index=False
)


# ============================================================
# SAVE REVIEW FEATURES
# ============================================================

review_features = audit[
    audit["Classification"] == "REVIEW"
].copy()

review_features.to_csv(
    REVIEW_FILE,
    index=False
)


# ============================================================
# SAVE REMOVE FEATURES
# ============================================================

remove_features = audit[
    audit["Classification"] == "REMOVE"
].copy()

remove_features.to_csv(
    REMOVE_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = [

    {
        "Metric": "Input_Feature_Rows",
        "Value": len(features)
    },

    {
        "Metric": "Input_Feature_Columns",
        "Value": len(features.columns)
    },

    {
        "Metric": "Target_Rows",
        "Value": len(targets)
    },

    {
        "Metric": "Merged_Rows",
        "Value": len(merged)
    },

    {
        "Metric": "Candidate_Numeric_Features",
        "Value": len(numeric_features)
    },

    {
        "Metric": "Safe_Features",
        "Value": safe_count
    },

    {
        "Metric": "Review_Features",
        "Value": review_count
    },

    {
        "Metric": "Remove_Features",
        "Value": remove_count
    },

    {
        "Metric": "Potential_Leakage_Name_Features",
        "Value": len(leakage_features)
    },

    {
        "Metric": "High_Missingness_Features",
        "Value": len(high_missing)
    },

    {
        "Metric": "Near_Constant_Features",
        "Value": len(high_constant)
    },

    {
        "Metric": "Mean_Stability_Score",
        "Value": audit[
            "Stability_Score"
        ].mean()
    },

    {
        "Metric": "Median_Stability_Score",
        "Value": audit[
            "Stability_Score"
        ].median()
    },

    {
        "Metric": "Best_Stability_Score",
        "Value": audit[
            "Stability_Score"
        ].max()
    },

    {
        "Metric": "Worst_Stability_Score",
        "Value": audit[
            "Stability_Score"
        ].min()
    },

    {
        "Metric": "Mean_Period_AUC",
        "Value": audit[
            "Mean_Period_AUC"
        ].mean()
    },

    {
        "Metric": "Median_Period_AUC",
        "Value": audit[
            "Mean_Period_AUC"
        ].median()
    }
]

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT REPORT
# ============================================================

print_header(
    "V6 FEATURE STABILITY AUDIT RESULTS"
)

print()
print("INPUT")
print("-" * 70)
print(f"Feature file       : {FEATURE_FILE}")
print(f"Target file        : {TARGET_FILE}")

print()
print("DATA")
print("-" * 70)
print(f"Rows analysed      : {len(merged)}")
print(f"Candidate features : {len(numeric_features)}")
print(f"Target             : {target_col}")

print()
print("CLASSIFICATION")
print("-" * 70)
print(f"SAFE                : {safe_count}")
print(f"REVIEW              : {review_count}")
print(f"REMOVE              : {remove_count}")

print()
print("QUALITY FLAGS")
print("-" * 70)
print(
    f"Potential leakage   : {len(leakage_features)}"
)
print(
    f"High missingness    : {len(high_missing)}"
)
print(
    f"Near constant      : {len(high_constant)}"
)

print()
print("STABILITY")
print("-" * 70)

mean_stability = audit[
    "Stability_Score"
].mean()

median_stability = audit[
    "Stability_Score"
].median()

best_stability = audit[
    "Stability_Score"
].max()

worst_stability = audit[
    "Stability_Score"
].min()

print(
    f"Mean stability      : {mean_stability:.3f}"
)

print(
    f"Median stability    : {median_stability:.3f}"
)

print(
    f"Best stability      : {best_stability:.3f}"
)

print(
    f"Worst stability     : {worst_stability:.3f}"
)


# ============================================================
# TOP SAFE FEATURES
# ============================================================

print()
print("TOP SAFE FEATURES")
print("-" * 70)

top_safe = audit[
    audit["Classification"] == "SAFE"
].sort_values(
    "Stability_Score",
    ascending=False
).head(20)

if len(top_safe) == 0:

    print("No features passed the SAFE screen.")

else:

    display_columns = [
        "Feature",
        "Mean_Period_AUC",
        "Stability_Score",
        "Missing_Rate",
        "Constant_Rate"
    ]

    print(
        top_safe[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# LEAKAGE FLAGS
# ============================================================

print()
print("LEAKAGE / SUSPICIOUS NAME FLAGS")
print("-" * 70)

if leakage_features:

    for feature in leakage_features[:50]:
        print(f" - {feature}")

    if len(leakage_features) > 50:
        print(
            f" ... and {len(leakage_features) - 50} more"
        )

else:

    print(
        "No obvious leakage names detected."
    )


# ============================================================
# OUTPUTS
# ============================================================

print()
print("OUTPUT FILES")
print("-" * 70)
print(f"Results  : {RESULTS_FILE}")
print(f"Summary  : {SUMMARY_FILE}")
print(f"SAFE     : {SAFE_FILE}")
print(f"REVIEW   : {REVIEW_FILE}")
print(f"REMOVE   : {REMOVE_FILE}")


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("INTERPRETATION")
print("-" * 70)

if len(leakage_features) > 0:

    print(
        "WARNING: Potential leakage-like feature names were detected."
    )

elif safe_count == 0:

    print(
        "WARNING: No features passed the conservative SAFE screen."
    )

elif safe_count < 10:

    print(
        "CAUTION: Only a small number of features passed."
    )

elif mean_stability >= 0.60:

    print(
        "PROMISING: Feature stability is relatively strong."
    )

elif mean_stability >= 0.40:

    print(
        "MIXED: Some feature stability exists, but further"
    )
    print(
        "chronological validation is required."
    )

else:

    print(
        "WEAK: Feature stability is generally limited."
    )

print()
print(
    "IMPORTANT: This audit does NOT establish profitability."
)
print(
    "It is a feature-quality and stability screening stage."
)
print(
    "The next V6 model must use strictly chronological validation."
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print(
    "V6 FEATURE STABILITY & LEAKAGE AUDIT COMPLETE"
)
print("=" * 70)

print(
    "PASS: V6 feature stability audit completed."
)

print("=" * 70)
