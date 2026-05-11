======================================================================
     PRIVACY-PRESERVING SYNTHETIC PATIENT RECORDS (RARE DISEASE)
======================================================================

TEAM
----------------------------------------------------------------------
* Aamna Khan
* Aikel Indurkhya


DATASET SUMMARY: SYNTHEA
----------------------------------------------------------------------
Total Patients:       1,171
Features:             15
Rare Disease Cases:   125


PIPELINE ARCHITECTURE
----------------------------------------------------------------------
1. build_rare_disease_dataset.py  - Merges raw SYNTHEA tables
2. preprocessing.py               - Data cleaning, encoding, & normalization
3. train_ctgan.py                 - Trains CTGAN + WGAN-GP models
4. privacy_audit.py               - MIA, Attribute Disclosure, & DP-SGD
5. fidelity_eval.py               - Statistical metrics (Pearson, KL, KS)
6. ml_tasks.py                    - 8 ML tasks + TSTR/TRTS evaluation
7. final_summary.py               - Compiles final performance reports


EXECUTION STEPS
----------------------------------------------------------------------
Run scripts in the following order from the root directory:

  python scripts/build_rare_disease_dataset.py
  python scripts/preprocessing.py
  python scripts/train_ctgan.py
  python scripts/privacy_audit.py
  python scripts/fidelity_eval.py
  python scripts/ml_tasks.py
  python scripts/final_summary.py


KEY PERFORMANCE METRICS
----------------------------------------------------------------------
[Metric]                  [Result]         [Status]
ML Task Accuracy (AUROC):  0.99+            PASS
Attribute Disclosure:      0.00%            PASS
DP-SGD Privacy Budget:     Epsilon 3.47     STRONG PRIVACY


SUMMARY OF FINDINGS
----------------------------------------------------------------------
The pipeline successfully generates high-fidelity synthetic data
while maintaining strict privacy guarantees. The low Epsilon value
(3.47) confirms that the model is resilient against membership 
inference and attribute disclosure attacks.
======================================================================
