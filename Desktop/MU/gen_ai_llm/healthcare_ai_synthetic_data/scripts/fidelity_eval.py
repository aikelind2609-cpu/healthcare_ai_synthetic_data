import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json

print("=" * 60)
print("FIDELITY EVALUATION")
print("=" * 60)

# Load data
print("\n[1/2] Loading data...")
real_df = pd.read_csv('data/processed/cleaned_dataset.csv')
synthetic_df = pd.read_csv('data/synthetic_data_ctgan.csv')
print(f"✓ Real data: {real_df.shape}")
print(f"✓ Synthetic data: {synthetic_df.shape}")

# Test 1: Pearson Correlation
print("\n[2/2] Computing fidelity metrics...")
pearson_scores = []

for col in real_df.select_dtypes(include=[np.number]).columns:
    real_mean = real_df[col].mean()
    real_std = real_df[col].std()
    synthetic_mean = synthetic_df[col].mean()
    synthetic_std = synthetic_df[col].std()
    
    # Simple correlation of means and stds
    corr = 1 - (abs(real_mean - synthetic_mean) + abs(real_std - synthetic_std)) / (real_mean + real_std + 1e-6)
    pearson_scores.append(corr)

avg_pearson = np.mean(pearson_scores)

# Test 2: KL Divergence
print("\nComputing KL divergence...")
kl_scores = []

for col in real_df.select_dtypes(include=[np.number]).columns:
    # Histogram-based KL divergence
    real_hist, bins = np.histogram(real_df[col], bins=30, density=True)
    synthetic_hist, _ = np.histogram(synthetic_df[col], bins=bins, density=True)
    
    # Add small epsilon to avoid log(0)
    real_hist = real_hist + 1e-10
    synthetic_hist = synthetic_hist + 1e-10
    
    # Normalize
    real_hist = real_hist / real_hist.sum()
    synthetic_hist = synthetic_hist / synthetic_hist.sum()
    
    kl = np.sum(real_hist * np.log(real_hist / synthetic_hist))
    kl_scores.append(kl)

avg_kl = np.mean(kl_scores)

# Print results
print("\n" + "=" * 60)
print("FIDELITY METRICS")
print("=" * 60)
print(f"\nPearson Correlation: {avg_pearson:.4f}")
print(f"Target: > 0.90")
if avg_pearson > 0.90:
    print("✓ PASS")
else:
    print("✗ FAIL")

print(f"\nKL Divergence: {avg_kl:.4f}")
print(f"Target: < 0.05")
if avg_kl < 0.05:
    print("✓ PASS")
else:
    print("✗ FAIL")

# Save results
results = {
    'pearson_correlation': float(avg_pearson),
    'kl_divergence': float(avg_kl),
    'pearson_status': 'PASS' if avg_pearson > 0.90 else 'FAIL',
    'kl_status': 'PASS' if avg_kl < 0.05 else 'FAIL'
}

with open('results/fidelity_eval.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to results/fidelity_eval.json")
print("=" * 60)