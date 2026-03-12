import pandas as pd

def add_kp_labels_for_cmes(
    df_cme: pd.DataFrame,
    df_kp: pd.DataFrame,
    horizon_hours: int = 72,
    label_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    For each CME startTime, compute max Kp within [t0, t0 + horizon_hours].
    Adds:
      - kp_max_h{horizon_hours}
      - label_storm (1 if kp_max >= threshold else 0)
    """
    df = df_cme.copy()
    if "startTime" not in df.columns:
        raise ValueError("CME dataframe must have startTime.")

    kp = df_kp[["time", "kp"]].copy()

    kp = kp.sort_values("time")
    df = df.sort_values("startTime")

    kp_max_list = []
    for t0 in df["startTime"]:
        if pd.isna(t0):
            kp_max_list.append(float("nan"))
            continue
        t1 = t0 + pd.Timedelta(hours=horizon_hours)
        window = kp[(kp["time"] >= t0) & (kp["time"] <= t1)]
        kp_max_list.append(window["kp"].max() if len(window) else float("nan"))

    col = f"kp_max_h{horizon_hours}"
    df[col] = kp_max_list
    df["label_storm"] = (df[col] >= label_threshold).astype("float")  # float to allow NaN
    return df

def merge_omni_window_features(
    df_events: pd.DataFrame,
    df_omni: pd.DataFrame,
    window_hours: int = 24,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    For each event time, aggregate OMNI variables in [t0 - window_hours, t0].
    Adds columns like: omni_mean_V, omni_mean_Bz, etc.
    """
    df = df_events.copy()
    if "startTime" in df.columns:
        time_col = "startTime"
    elif "beginTime" in df.columns:
        time_col = "beginTime"
    else:
        raise ValueError("Events df must contain startTime (CME) or beginTime (flare).")

    omni = df_omni.copy().sort_values("time")
    numeric_cols = [c for c in omni.columns if c != "time"]

    out = []
    for t0 in df[time_col]:
        if pd.isna(t0):
            out.append({f"omni_{agg}_{c}": float("nan") for c in numeric_cols})
            continue
        t_start = t0 - pd.Timedelta(hours=window_hours)
        w = omni[(omni["time"] >= t_start) & (omni["time"] <= t0)]
        if len(w) == 0:
            out.append({f"omni_{agg}_{c}": float("nan") for c in numeric_cols})
            continue

        if agg == "mean":
            vals = w[numeric_cols].mean(numeric_only=True)
        elif agg == "median":
            vals = w[numeric_cols].median(numeric_only=True)
        elif agg == "max":
            vals = w[numeric_cols].max(numeric_only=True)
        else:
            raise ValueError(f"Unsupported agg={agg}")

        out.append({f"omni_{agg}_{c}": vals.get(c, float("nan")) for c in numeric_cols})

    feats = pd.DataFrame(out)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
