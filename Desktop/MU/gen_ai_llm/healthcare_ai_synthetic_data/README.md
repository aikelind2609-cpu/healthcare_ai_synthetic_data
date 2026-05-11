\# Privacy-Preserving Synthetic Patient Records for Rare Disease Clinical Trials



\## Team

\- Aamna Khan

\- Aikel Indurkhya



\## Dataset

SYNTHEA - 1,171 patients, 15 features, 125 rare disease patients



\## Pipeline

1\. build\_rare\_disease\_dataset.py - Merge SYNTHEA tables

2\. preprocessing.py - Clean, encode, normalize

3\. train\_ctgan.py - Train CTGAN + WGAN-GP

4\. privacy\_audit.py - MIA + Attribute Disclosure + DP-SGD

5\. fidelity\_eval.py - Pearson + KL + KS metrics

6\. ml\_tasks.py - 8 ML tasks + TSTR/TRTS

7\. final\_summary.py - Compile all results



\## How to Run

python scripts/build\_rare\_disease\_dataset.py

python scripts/preprocessing.py

python scripts/train\_ctgan.py

python scripts/privacy\_audit.py

python scripts/fidelity\_eval.py

python scripts/ml\_tasks.py

python scripts/final\_summary.py



\## Results

\- ML Tasks: AUROC 0.99+ (PASS)

\- Attribute Disclosure: 0.00% (PASS)

\- DP-SGD: Epsilon 3.47 (Strong Privacy)

