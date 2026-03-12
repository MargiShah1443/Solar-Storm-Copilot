import pandas as pd

def basic_eda_summary(df: pd.DataFrame) -> dict:
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
        "missing_fraction_mean": float(df.isna().mean().mean()),
        "missing_cols_gt_50pct": int((df.isna().mean() > 0.5).sum()),
    }
    return summary
