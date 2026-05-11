import json

print("=" * 60)
print("STEP 7: FINAL PROJECT SUMMARY")
print("=" * 60)

with open('results/gan_comparison.json') as f:
    gan_comp = json.load(f)
with open('results/privacy_audit.json') as f:
    privacy = json.load(f)
with open('results/fidelity_eval.json') as f:
    fidelity = json.load(f)
with open('results/ml_tasks.json') as f:
    ml_tasks = json.load(f)

print("\n" + "=" * 60)
print("1. GAN COMPARISON")
print("=" * 60)
print(f"  CTGAN KS: {gan_comp['ctgan_ks']:.4f}")
print(f"  WGAN-GP KS: {gan_comp['wgan_gp_ks']:.4f}")
print(f"  Winner: {gan_comp['winner']}")

print("\n" + "=" * 60)
print("2. PRIVACY")
print("=" * 60)
print(f"  CTGAN MIA AUC: {privacy['mia']['ctgan_auc']:.4f} [{privacy['mia']['ctgan_status']}]")
print(f"  WGAN-GP MIA AUC: {privacy['mia']['wgan_gp_auc']:.4f} [{privacy['mia']['wgan_gp_status']}]")
print(f"  CTGAN Disclosure: {privacy['attribute_disclosure']['ctgan_rate']:.2f}% [{privacy['attribute_disclosure']['ctgan_status']}]")
print(f"  WGAN-GP Disclosure: {privacy['attribute_disclosure']['wgan_gp_rate']:.2f}% [{privacy['attribute_disclosure']['wgan_gp_status']}]")
print(f"  CTGAN Epsilon: {privacy['dp_sgd']['ctgan_epsilon']:.4f}")
print(f"  WGAN-GP Epsilon: {privacy['dp_sgd']['wgan_gp_epsilon']:.4f}")

print("\n" + "=" * 60)
print("3. FIDELITY")
print("=" * 60)
for gan in ['ctgan', 'wgan_gp']:
    f = fidelity[gan]
    print(f"\n  {gan.upper()}:")
    print(f"    Pearson r: {f['pearson_r']:.4f} [{f['pearson_status']}]")
    print(f"    KL Divergence: {f['kl_divergence']:.4f} [{f['kl_status']}]")
    print(f"    KS Statistic: {f['ks_statistic']:.4f} [{f['ks_status']}]")
    print(f"    Corr Distance: {f['correlation_distance']:.4f} [{f['corr_status']}]")

print("\n" + "=" * 60)
print("4. ML TASKS (8 Clinical Prediction Tasks)")
print("=" * 60)
print("\n  CTGAN:")
for task, r in ml_tasks.get('ctgan_tasks', {}).items():
    print(f"    {task}: AUROC={r['auroc']:.4f} F1={r['f1']:.4f} [{r['status']}]")
print("\n  WGAN-GP:")
for task, r in ml_tasks.get('wgan_gp_tasks', {}).items():
    print(f"    {task}: AUROC={r['auroc']:.4f} F1={r['f1']:.4f} [{r['status']}]")

print("\n" + "=" * 60)
print("5. TSTR vs TRTS")
print("=" * 60)
print(f"  Average Gap: {ml_tasks.get('avg_gap', 'N/A')}")
print(f"  Status: {ml_tasks.get('gap_status', 'N/A')}")

print("\n" + "=" * 60)
print("6. OVERALL")
print("=" * 60)

total = 0
passes = 0

for key in ['ctgan_status', 'wgan_gp_status']:
    total += 1
    if privacy['mia'][key] == 'PASS': passes += 1
for key in ['ctgan_status', 'wgan_gp_status']:
    total += 1
    if privacy['attribute_disclosure'][key] == 'PASS': passes += 1
for gan in ['ctgan', 'wgan_gp']:
    for m in ['pearson_status', 'kl_status', 'ks_status', 'corr_status']:
        total += 1
        if fidelity[gan][m] == 'PASS': passes += 1
for task, r in ml_tasks.get('ctgan_tasks', {}).items():
    total += 1
    if r['status'] == 'PASS': passes += 1

rate = (passes / total * 100) if total > 0 else 0

print(f"\n  Total Checks: {total}")
print(f"  Passed: {passes}")
print(f"  Failed: {total - passes}")
print(f"  Pass Rate: {rate:.1f}%")

if rate >= 70:
    print(f"\n  ✓ OVERALL: PASS")
elif rate >= 50:
    print(f"\n  ~ OVERALL: PARTIAL PASS")
else:
    print(f"\n  ✗ OVERALL: NEEDS MORE DATA")

summary = {
    "gan_comparison": gan_comp, "privacy": privacy,
    "fidelity": fidelity, "ml_tasks": ml_tasks,
    "overall": {"total": total, "passes": passes, "failures": total - passes, "pass_rate": float(rate)}
}

with open('results/final_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Saved to results/final_summary.json")
print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)