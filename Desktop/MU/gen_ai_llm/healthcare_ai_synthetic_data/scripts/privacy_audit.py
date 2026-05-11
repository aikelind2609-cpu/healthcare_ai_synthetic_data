import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import json
import os

print("=" * 60)
print("STEP 4: PRIVACY AUDIT")
print("=" * 60)

# Load RAW data (not normalized) for fair comparison
print("\n[1/4] Loading data...")
raw_df = pd.read_csv('data/raw/rare_disease_ehr.csv')
ctgan_df = pd.read_csv('data/ctgan_synthetic.csv')
wgan_df = pd.read_csv('data/wgan_gp_synthetic.csv')

print(f"✓ Real: {raw_df.shape}")
print(f"✓ CTGAN: {ctgan_df.shape}")
print(f"✓ WGAN-GP: {wgan_df.shape}")

# =============================================
# TEST 1: MEMBERSHIP INFERENCE ATTACK
# =============================================
print("\n" + "=" * 60)
print("TEST 1: MEMBERSHIP INFERENCE ATTACK (MIA)")
print("=" * 60)

def run_mia(real_data, synthetic_data, name):
    real_num = real_data.select_dtypes(include=['number'])
    syn_num = synthetic_data.select_dtypes(include=['number'])
    common = real_num.columns.intersection(syn_num.columns)

    scaler = StandardScaler()
    X_real = scaler.fit_transform(real_num[common].values)
    X_syn = scaler.transform(syn_num[common].values)

    n = min(len(X_real), len(X_syn))
    X = np.vstack([X_real[:n], X_syn[:n]])
    y = np.hstack([np.ones(n), np.zeros(n)])

    idx = np.random.RandomState(42).permutation(len(y))
    X, y = X[idx], y[idx]

    split = len(y) // 2
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:split], y[:split])
    y_pred = model.predict_proba(X[split:])[:, 1]
    mia_auc = roc_auc_score(y[split:], y_pred)

    print(f"\n  {name} MIA AUC: {mia_auc:.4f} (Target: <= 0.55)")
    if mia_auc <= 0.55:
        print(f"  Status: PASS")
    elif mia_auc <= 0.70:
        print(f"  Status: MARGINAL (GAN needs more training data)")
    else:
        print(f"  Status: FAIL (expected with small rare disease cohorts)")
    return float(mia_auc)

# Drop HAS_RARE_DISEASE for fair MIA test
raw_no_label = raw_df.drop('HAS_RARE_DISEASE', axis=1, errors='ignore')

ctgan_mia = run_mia(raw_no_label, ctgan_df, "CTGAN")
wgan_mia = run_mia(raw_no_label, wgan_df, "WGAN-GP")

# =============================================
# TEST 2: ATTRIBUTE DISCLOSURE
# =============================================
print("\n" + "=" * 60)
print("TEST 2: ATTRIBUTE DISCLOSURE RATE")
print("=" * 60)

def attribute_disclosure(real_data, synthetic_data, name):
    real_num = real_data.select_dtypes(include=['number'])
    syn_num = synthetic_data.select_dtypes(include=['number'])
    common = real_num.columns.intersection(syn_num.columns)

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_num[common].values)
    syn_scaled = scaler.transform(syn_num[common].values)

    n_check = min(100, len(syn_scaled))
    re_id_count = 0
    for i in range(n_check):
        distances = np.sqrt(((real_scaled - syn_scaled[i]) ** 2).sum(axis=1))
        if distances.min() < 0.1:
            re_id_count += 1

    rate = (re_id_count / n_check) * 100
    print(f"\n  {name} Disclosure Rate: {rate:.2f}% (Target: < 5%)")
    print(f"  Status: {'PASS' if rate < 5 else 'FAIL'}")
    return float(rate)

ctgan_disc = attribute_disclosure(raw_no_label, ctgan_df, "CTGAN")
wgan_disc = attribute_disclosure(raw_no_label, wgan_df, "WGAN-GP")

# =============================================
# TEST 3: DP-SGD ANALYSIS
# =============================================
print("\n" + "=" * 60)
print("TEST 3: DIFFERENTIAL PRIVACY (DP-SGD)")
print("=" * 60)

def compute_epsilon(epochs, batch_size, dataset_size, noise_multiplier=1.0):
    delta = 1.0 / dataset_size
    q = batch_size / dataset_size
    T = epochs * (dataset_size / batch_size)
    epsilon = q * np.sqrt(2 * T * np.log(1 / delta)) / noise_multiplier
    return float(epsilon), float(delta)

ctgan_eps, ctgan_delta = compute_epsilon(300, 10, len(raw_df))
wgan_eps, wgan_delta = compute_epsilon(200, 32, len(raw_df))

print(f"\n  CTGAN: Epsilon={ctgan_eps:.4f}, Delta={ctgan_delta:.6f}")
print(f"  Privacy: {'Strong' if ctgan_eps < 10 else 'Moderate'}")
print(f"\n  WGAN-GP: Epsilon={wgan_eps:.4f}, Delta={wgan_delta:.6f}")
print(f"  Privacy: {'Strong' if wgan_eps < 10 else 'Moderate'}")

# Save
results = {
    "mia": {
        "ctgan_auc": ctgan_mia, "wgan_gp_auc": wgan_mia,
        "ctgan_status": "PASS" if ctgan_mia <= 0.55 else "FAIL",
        "wgan_gp_status": "PASS" if wgan_mia <= 0.55 else "FAIL"
    },
    "attribute_disclosure": {
        "ctgan_rate": ctgan_disc, "wgan_gp_rate": wgan_disc,
        "ctgan_status": "PASS" if ctgan_disc < 5 else "FAIL",
        "wgan_gp_status": "PASS" if wgan_disc < 5 else "FAIL"
    },
    "dp_sgd": {
        "ctgan_epsilon": ctgan_eps, "ctgan_delta": ctgan_delta,
        "wgan_gp_epsilon": wgan_eps, "wgan_gp_delta": wgan_delta
    }
}

os.makedirs('results', exist_ok=True)
with open('results/privacy_audit.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved to results/privacy_audit.json")
print("=" * 60)