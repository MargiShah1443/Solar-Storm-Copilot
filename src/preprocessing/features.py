import pandas as pd

def parse_flare_class_strength(class_type: pd.Series) -> pd.Series:
    """
    Converts flare class like 'M1.4' or 'X2.0' into an approximate numeric strength.
    Common mapping: A<B<C<M<X; treat letter as decade multiplier.
    """
    multipliers = {"A": 1e-4, "B": 1e-3, "C": 1e-2, "M": 1e-1, "X": 1.0}

    def to_strength(x):
        if not isinstance(x, str) or len(x) < 2:
            return None
        letter = x[0].upper()
        try:
            number = float(x[1:])
        except ValueError:
            return None
        return multipliers.get(letter, None) * number

    return class_type.apply(to_strength)

def add_basic_features_cme(df_cme: pd.DataFrame) -> pd.DataFrame:
    df = df_cme.copy()
    if "analysis_speed" in df.columns:
        df["analysis_speed"] = pd.to_numeric(df["analysis_speed"], errors="coerce")
    # Example: hour-of-day feature
    if "startTime" in df.columns:
        df["start_hour"] = df["startTime"].dt.hour
        df["start_dow"] = df["startTime"].dt.dayofweek
    return df

def add_basic_features_flare(df_flr: pd.DataFrame) -> pd.DataFrame:
    df = df_flr.copy()
    if "classType" in df.columns:
        df["flare_strength"] = parse_flare_class_strength(df["classType"])
    if "beginTime" in df.columns:
        df["start_hour"] = df["beginTime"].dt.hour
        df["start_dow"] = df["beginTime"].dt.dayofweek
    return df
