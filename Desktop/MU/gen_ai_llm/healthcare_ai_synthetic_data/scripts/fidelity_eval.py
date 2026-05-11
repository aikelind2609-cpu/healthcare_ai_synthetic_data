import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import os

print("=" * 60)
print("STEP 5: FIDELITY EVALUATION")
print("=" * 60)

print("\n[1/3] Loading data...")
raw_df = pd.read_csv('data/raw/rare_disease_ehr.csv')
raw_df = raw_df.drop('HAS_RARE_DISEASE', axis=1, errors='ignore')
ctgan_df = pd.read_csv('data/ctgan_synthetic.csv')
wgan_df = pd.read_csv('data/wgan_gp_synthetic.csv')

real_num = raw_df.select_dtypes(include=[np.number])
ctgan_num = ctgan_df.select_dtypes(include=[np.number])
wgan_num = wgan_df.select_dtypes(include=[np.number])

common_ctgan = real_num.columns.intersection(ctgan_num.columns)
common_wgan = real_num.columns.intersection(wgan_num.columns)

print(f"✓ Real: {raw_df.shape}")
print(f"✓ CTGAN: {ctgan_df.shape}")
print(f"✓ WGAN-GP: {wgan_df.shape}")

def evaluate_fidelity(real_data, synthetic_data, name):
    print(f"\n{'=' * 40}")
    print(f"FIDELITY: {name}")
    print(f"{'=' * 40}")
    results = {}

    # 1. Pearson Correlation
    real_means = real_data.mean().values
    syn_means = synthetic_data.mean().values
    real_stds = real_data.std().values
    syn_stds = synthetic_data.std().values

    if len(real_means) > 1:
        mean_corr = np.corrcoef(real_means, syn_means)[0, 1]
        std_corr = np.corrcoef(real_stds, syn_stds)[0, 1]
        if np.isnan(mean_corr): mean_corr = 0.0
        if np.isnan(std_corr): std_corr = 0.0
    else:
        mean_corr = std_corr = 0.0

    avg_pearson = (mean_corr + std_corr) / 2
    results['pearson_r'] = float(avg_pearson)
    results['pearson_status'] = 'PASS' if avg_pearson > 0.90 else 'FAIL'
    print(f"  Pearson r: {avg_pearson:.4f} (Target: > 0.90) [{results['pearson_status']}]")

    # 2. KL Divergence
    kl_scores = []
    for col in real_data.columns:
        if col in synthetic_data.columns:
            r_hist, bins = np.histogram(real_data[col].dropna(), bins=20, density=True)
            s_hist, _ = np.histogram(synthetic_data[col].dropna(), bins=bins, density=True)
            r_hist = r_hist + 1e-10
            s_hist = s_hist + 1e-10
            r_hist = r_hist / r_hist.sum()
            s_hist = s_hist / s_hist.sum()
            kl_scores.append(np.sum(r_hist * np.log(r_hist / s_hist)))

    avg_kl = np.mean(kl_scores) if kl_scores else 1.0
    results['kl_divergence'] = float(avg_kl)
    results['kl_status'] = 'PASS' if avg_kl < 0.05 else 'FAIL'
    print(f"  KL Divergence: {avg_kl:.4f} (Target: < 0.05) [{results['kl_status']}]")

    # 3. KS Statistic
    ks_scores = []
    for col in real_data.columns:
        if col in synthetic_data.columns:
            ks, _ = ks_2samp(real_data[col].dropna(), synthetic_data[col].dropna())
            ks_scores.append(ks)

    avg_ks = np.mean(ks_scores) if ks_scores else 1.0
    results['ks_statistic'] = float(avg_ks)
    results['ks_status'] = 'PASS' if avg_ks < 0.15 else 'FAIL'
    print(f"  KS Statistic: {avg_ks:.4f} (Target: < 0.15) [{results['ks_status']}]")

    # 4. Correlation Distance
    common = [c for c in real_data.columns if c in synthetic_data.columns]
    real_corr = real_data[common].corr().values
    syn_corr = synthetic_data[common].corr().values
    corr_dist = np.sqrt(np.mean((real_corr - syn_corr) ** 2))
    results['correlation_distance'] = float(corr_dist)
    results['corr_status'] = 'PASS' if corr_dist < 0.30 else 'FAIL'
    print(f"  Corr Distance: {corr_dist:.4f} (Target: < 0.30) [{results['corr_status']}]")

    return results

print("\n[2/3] Evaluating CTGAN...")
ctgan_results = evaluate_fidelity(real_num[common_ctgan], ctgan_num[common_ctgan], "CTGAN")

print("\n[3/3] Evaluating WGAN-GP...")
wgan_results = evaluate_fidelity(real_num[common_wgan], wgan_num[common_wgan], "WGAN-GP")

os.makedirs('results', exist_ok=True)
with open('results/fidelity_eval.json', 'w') as f:
    json.dump({"ctgan": ctgan_results, "wgan_gp": wgan_results}, f, indent=2)

print(f"\n✓ Saved to results/fidelity_eval.json")
print("=" * 60)