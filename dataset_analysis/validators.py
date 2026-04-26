def validate_no_nan(df, dataset_name=""):
    if df.isnull().values.any():
        raise ValueError(f"NaNs found in dataset: {dataset_name}")


def validate_min_samples(df, min_samples=10, dataset_name=""):
    if len(df) < min_samples:
        raise ValueError(f"Too few samples in {dataset_name}: {len(df)}")