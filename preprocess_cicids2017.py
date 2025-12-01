import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ===================================================================
# 🔹 CONFIGURATION
csv_path = "data\cicids2017_cleaned.csv"       # <-- change if your file name is different
seq_len = 32                     # number of flows per window
save_prefix = "processed\preprocessed_"     # output prefix
# ===================================================================

print("🔄 Loading CSV...")
df = pd.read_csv(csv_path)

# -------------------------------------------------------------------
# 🔹 Step 1: Convert Attack Type → binary label
# -------------------------------------------------------------------
df["LabelBinary"] = (df["Attack Type"] != "Normal Traffic").astype(int)

label_counts = df["LabelBinary"].value_counts()
print("\nLabel distribution (0 = Normal, 1 = Attack):")
print(label_counts)

# -------------------------------------------------------------------
# 🔹 Step 2: Extract numeric features
# -------------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in ["Attack Type", "LabelBinary"]]
X_raw = df[feature_cols].values  # shape (N, 52)
y_raw = df["LabelBinary"].values  # shape (N,)
print(f"\nFeature count: {len(feature_cols)} numeric features")

# -------------------------------------------------------------------
# 🔹 Step 3: Normalize features (StandardScaler)
# -------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# save scaler for later inference / dashboard / few-shot
joblib.dump(scaler, f"{save_prefix}scaler.pkl")
print("✔ Saved scaler")

# -------------------------------------------------------------------
# 🔹 Step 4: Build sequence windows (for Transformer + GNN)
# -------------------------------------------------------------------
N = len(X_scaled)
N_trim = (N // seq_len) * seq_len  # drop last incomplete part
X_trim = X_scaled[:N_trim]
y_trim = y_raw[:N_trim]

# reshape into windows
X_seq = X_trim.reshape(-1, seq_len, X_scaled.shape[1])  # (num_windows, 32, 52)
y_seq = y_trim.reshape(-1, seq_len)

print("\n===== Final shapes =====")
print("X_seq:", X_seq.shape)
print("y_seq:", y_seq.shape)

# -------------------------------------------------------------------
# 🔹 Step 5: Save for later model training
# -------------------------------------------------------------------
np.save(f"{save_prefix}X_seq.npy", X_seq)
np.save(f"{save_prefix}y_seq.npy", y_seq)
np.save(f"{save_prefix}feature_cols.npy", np.array(feature_cols, dtype=object))
print("✔ Saved numpy arrays:")
print(f"   {save_prefix}X_seq.npy")
print(f"   {save_prefix}y_seq.npy")
print(f"   {save_prefix}feature_cols.npy")

print("\n🎉 Preprocessing complete.")
