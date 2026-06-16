# 🛡️ Privacy-Preserving Synthetic Patient Records
### Leveraging CTGAN & WGAN-GP for Rare Disease Clinical Trials

![Python](https://shields.io)
![Privacy](https://shields.io)
![Status](https://shields.io)

## 👥 The Team
- **Aikel Indurkhya**
- **Aamna Khan**

## 📊 Dataset Overview: SYNTHEA
Our pipeline focuses on high-fidelity synthesis of rare disease data, ensuring patient privacy while maintaining clinical utility.
*   **Total Patients:** 1,171
*   **Features:** 15 clinical variables
*   **Rare Disease Subset:** 125 patients (Target Class)

---

## ⚙️ Project Pipeline
The workflow is modularized into specialized scripts for data engineering, model training, and rigorous auditing:


| Phase | Script | Description |
| :--- | :--- | :--- |
| **1. Data** | `build_rare_disease_dataset.py` | Merges SYNTHEA relational tables |
| **2. Clean** | `preprocessing.py` | Cleaning, encoding, and normalization |
| **3. Model** | `train_ctgan.py` | Training CTGAN + WGAN-GP architectures |
| **4. Audit** | `privacy_audit.py` | MIA, Attribute Disclosure & DP-SGD analysis |
| **5. Fidelity** | `fidelity_eval.py` | Statistical metrics (Pearson, KL, KS) |
| **6. ML Tasks** | `ml_tasks.py` | TSTR/TRTS evaluation across 8 tasks |
| **7. Report** | `final_summary.py` | Compilation of all performance metrics |

---

## 🚀 Execution Guide
Run the pipeline sequentially to reproduce the results:

```bash
# Data Preparation
python scripts/build_rare_disease_dataset.py
python scripts/preprocessing.py

# Model Training & Auditing
python scripts/train_ctgan.py
python scripts/privacy_audit.py

# Evaluation & Summary
python scripts/fidelity_eval.py
python scripts/ml_tasks.py
python scripts/final_summary.py
```

---

## 📈 Key Results
Our model successfully balances the "Privacy-Utility Trade-off."


| Metric | Result | Status |
| :--- | :--- | :--- |
| **ML Performance** | **AUROC 0.99+** | ✅ PASS |
| **Attribute Disclosure** | **0.00% Risk** | ✅ PASS |
| **Privacy Budget** | **ε = 3.47** | ✅ STRONG |

### 🔍 Performance Highlights
- **High Utility:** Synthetic data maintains statistical distributions suitable for ML training (TSTR).
- **Strong Privacy:** DP-SGD implementation ensures epsilon values remain within the "Strong Privacy" threshold.
- **Leakage Prevention:** Zero attribute disclosure risk detected during adversarial audits.
