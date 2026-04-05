import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
base_folder = r"C:\usp\tabular-transfer-learning\data\ic_downstream1"
test_folder = r"C:\usp\tabular-transfer-learning\data\ic_downstream1_Sample75_Imputation_Real_Values_exp_100_4"
output_folder = r"C:\usp\tabular-transfer-learning\data\ic_downstream1_a_updated"

os.makedirs(output_folder, exist_ok=True)

# =========================
# LOAD ALL MOLECULES
# =========================
all_molecules = pd.read_csv(os.path.join(base_folder, "exp_100_1.csv"), sep="|")

# =========================
# LOAD TEST SET (X + Y)
# =========================
test_X = pd.read_csv(os.path.join(test_folder, "ic_test_X.csv"))
test_y = pd.read_csv(os.path.join(test_folder, "ic_test_y.csv"))

# Ensure y is a column
if test_y.shape[1] == 1:
    test_y.columns = ["target"]

test_full = pd.concat([test_X, test_y], axis=1)

# =========================
# FIND COMMON COLUMNS
# =========================
common_cols = list(set(all_molecules.columns).intersection(set(test_full.columns)))

print(f"Using {len(common_cols)} common columns for matching")

# =========================
# IDENTIFY TEST MOLECULES
# =========================
# Create keys for matching
def create_key(df, cols):
    return df[cols].astype(str).agg("||".join, axis=1)

all_molecules["_key"] = create_key(all_molecules, common_cols)
test_full["_key"] = create_key(test_full, common_cols)

test_keys = set(test_full["_key"])

# Split datasets
test_set = all_molecules[all_molecules["_key"].isin(test_keys)].copy()
remaining = all_molecules[~all_molecules["_key"].isin(test_keys)].copy()

print(f"Test found: {len(test_set)}")
print(f"Remaining: {len(remaining)}")

# =========================
# TRAIN / VAL SPLIT
# =========================
train_df, val_df = train_test_split(
    remaining,
    test_size=0.1875,
    random_state=42,
    shuffle=True
)

# =========================
# LOAD ORIGINAL HEADERS
# =========================
train_X_cols = pd.read_csv(os.path.join(base_folder, "ic_train_X.csv"), nrows=0).columns
train_y_cols = pd.read_csv(os.path.join(base_folder, "ic_train_y.csv"), nrows=0).columns

# =========================
# FUNCTION TO SPLIT X / Y
# =========================
def split_X_y(df, X_cols, y_cols):
    X = df[X_cols]
    y = df[y_cols]
    return X, y

# =========================
# SPLIT DATASETS
# =========================
train_X, train_y = split_X_y(train_df, train_X_cols, train_y_cols)
val_X, val_y = split_X_y(val_df, train_X_cols, train_y_cols)
test_X_final, test_y_final = split_X_y(test_set, train_X_cols, train_y_cols)

# =========================
# SAVE FILES
# =========================
train_X.to_csv(os.path.join(output_folder, "ic_train_X.csv"), index=False)
train_y.to_csv(os.path.join(output_folder, "ic_train_y.csv"), index=False)

val_X.to_csv(os.path.join(output_folder, "ic_val_X.csv"), index=False)
val_y.to_csv(os.path.join(output_folder, "ic_val_y.csv"), index=False)

test_X_final.to_csv(os.path.join(output_folder, "ic_test_X.csv"), index=False)
test_y_final.to_csv(os.path.join(output_folder, "ic_test_y.csv"), index=False)

print("✅ Done! Files saved in:", output_folder)