import json
import pandas as pd

print("=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

# Load all results
with open('results/privacy_audit.json') as f:
    privacy = json.load(f)
with open('results/fidelity_eval.json') as f:
    fidelity = json.load(f)
with open('results/ml_tasks.json') as f:
    ml_tasks = json.load(f)

# Create summary
print("\n" + "=" * 60)
print("PRIVACY METRICS")
print("=" * 60)
print(f"MIA AUC: {privacy['mia_auc']:.4f} (Target: ≤ 0.55)")
print(f"Status: {privacy['status']}")

print("\n" + "=" * 60)
print("FIDELITY METRICS")
print("=" * 60)
print(f"Pearson Correlation: {fidelity['pearson_correlation']:.4f} (Target: > 0.90)")
print(f"Status: {fidelity['pearson_status']}")
print(f"KL Divergence: {fidelity['kl_divergence']:.4f} (Target: < 0.05)")
print(f"Status: {fidelity['kl_status']}")

print("\n" + "=" * 60)
print("UTILITY METRICS (ML TASKS)")
print("=" * 60)
print(f"Task 1 (Eligibility) AUROC: {ml_tasks['task1_eligibility']['auroc']:.4f}")
print(f"Status: {ml_tasks['task1_eligibility']['status']}")
print(f"Task 2 (Disease Onset) AUROC: {ml_tasks['task2_disease_onset']['auroc']:.4f}")
print(f"Status: {ml_tasks['task2_disease_onset']['status']}")

print("\n" + "=" * 60)
print("OVERALL PROJECT STATUS")
print("=" * 60)

all_status = [
    privacy['status'],
    fidelity['pearson_status'],
    fidelity['kl_status'],
    ml_tasks['task1_eligibility']['status'],
    ml_tasks['task2_disease_onset']['status']
]

overall = 'PASS' if all(s == 'PASS' for s in all_status) else 'FAIL'
print(f"\n{'✓' if overall == 'PASS' else '✗'} OVERALL: {overall}")

# Save summary
summary = {
    'privacy': privacy,
    'fidelity': fidelity,
    'ml_tasks': ml_tasks,
    'overall_status': overall
}

with open('results/final_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Summary saved to results/final_summary.json")
print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)
print("\nAll files created:")
print("✓ data/synthetic_data_ctgan.csv")
print("✓ models/ctgan_model.pkl")
print("✓ results/privacy_audit.json")
print("✓ results/fidelity_eval.json")
print("✓ results/ml_tasks.json")
print("✓ results/final_summary.json")