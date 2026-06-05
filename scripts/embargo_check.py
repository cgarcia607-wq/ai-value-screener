"""End-to-end verification of WalkForwardCV against real membership data.

Run from anywhere:
    python scripts/embargo_check.py

Expected (post calendar-month-anchor fix): 8 folds, every test window
exactly 12 months, max(train) to min(test) gap 395-399 days, 0 embargo
violations.
"""

import sys
from pathlib import Path

# Repo root on sys.path so `src.*` imports work when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_sources.sp500_membership import load_membership_table  # noqa: E402
from src.validation.walk_forward import WalkForwardCV  # noqa: E402

table = load_membership_table()
df = table.to_pandas()

cv = WalkForwardCV(train_period=36, test_period=12, embargo=12, step=12, expanding=True)
folds = list(cv.split_long(df, date_col='as_of_date'))

print("=== Per-fold counts ===")
for fold in folds:
    train_dates = df.iloc[fold.train_indices]['as_of_date']
    test_dates = df.iloc[fold.test_indices]['as_of_date']
    print(f"Fold {fold.fold_id}: train rows={len(fold.train_indices)} ({train_dates.nunique()} months), test rows={len(fold.test_indices)} ({test_dates.nunique()} months), vintage={fold.vintage_date}")

print()
print("=== Embargo verification ===")
violations = 0
for fold in folds:
    train_dates = df.iloc[fold.train_indices]['as_of_date']
    test_dates = df.iloc[fold.test_indices]['as_of_date']
    max_train = train_dates.max()
    min_test = test_dates.min()
    gap = (min_test - max_train).days
    print(f"Fold {fold.fold_id}: max_train={max_train}, min_test={min_test}, gap={gap} days")
    if gap < 365:
        violations += 1

print()
print(f"Embargo violations: {violations} / {len(folds)}")
