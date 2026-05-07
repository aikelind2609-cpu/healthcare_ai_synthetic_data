import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json

print("=" * 60)
print("PRIVACY AUDIT - MEMBERSHIP INFERENCE ATTACK")
print("=" * 60)

# Load real and synthetic data
print("\n[1/3] Loading data...")
real_df = pd.read_csv('data/processed/cleaned_dataset.csv')
synthetic_df = pd.read_csv('data/synthetic_data_ctgan.csv')
print(f"✓ Real data: {real_df.shape}")
print(f"✓ Synthetic data: {synthetic_df.shape}")

# Split real data: training (used in GAN) vs testing (not used)
X_train, X_test = train_test_split(real_df, test_size=0.2, random_state=42)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Testing set: {X_test.shape}")

# Create attack dataset
print("\n[2/3] Creating attack dataset...")
X_members = X_train.iloc[:500].values  # From training (members)
X_non_members = X_test.iloc[:500].values  # From testing (non-members)

X_attack = np.vstack([X_members, X_non_members])
y_attack = np.hstack([np.ones(len(X_members)), np.zeros(len(X_non_members))])

print(f"✓ Members (in training): {len(X_members)}")
print(f"✓ Non-members (not in training): {len(X_non_members)}")

# Train attack classifier
print("\n[3/3] Training attack classifier...")
attack_model = RandomForestClassifier(n_estimators=100, random_state=42)
attack_model.fit(X_attack, y_attack)
print(f"✓ Attack model trained")

# Test on attack data
y_pred = attack_model.predict_proba(X_attack)[:, 1]
mia_auc = roc_auc_score(y_attack, y_pred)

print("\n" + "=" * 60)
print("PRIVACY AUDIT RESULTS")
print("=" * 60)
print(f"\nMIA AUC Score: {mia_auc:.4f}")
print(f"Target: ≤ 0.55 (0.50 = perfect privacy)")

if mia_auc <= 0.55:
    print("✓ PASS - Privacy is good!")
else:
    print("✗ FAIL - Privacy needs improvement")

# Save results
results = {
    'mia_auc': float(mia_auc),
    'status': 'PASS' if mia_auc <= 0.55 else 'FAIL',
    'interpretation': 'Synthetic data does not leak training data' if mia_auc <= 0.55 else 'Privacy concern detected'
}

with open('results/privacy_audit.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to results/privacy_audit.json")
print("=" * 60)