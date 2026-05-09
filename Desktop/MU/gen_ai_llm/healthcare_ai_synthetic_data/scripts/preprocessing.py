import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print("=" * 60)
print("DATA PREPROCESSING PIPELINE")
print("=" * 60)

# Load data
print("\n[1/4] Loading data...")
admissions = pd.read_csv('data/raw/admissions.csv')
print(f"✓ Loaded admissions: {admissions.shape}")

# Handle missing values
print("\n[2/4] Handling missing values...")
imputer = SimpleImputer(strategy='median')
numeric_cols = admissions.select_dtypes(include=[np.number]).columns
admissions[numeric_cols] = imputer.fit_transform(admissions[numeric_cols])
print(f"✓ Missing values filled")

# Normalize features
print("\n[3/4] Normalizing features...")
scaler = StandardScaler()
admissions[numeric_cols] = scaler.fit_transform(admissions[numeric_cols])
print(f"✓ Features normalized")

# Save cleaned data
print("\n[4/4] Saving cleaned data...")
admissions.to_csv('data/processed/cleaned_dataset.csv', index=False)
print(f"✓ Saved to data/processed/cleaned_dataset.csv")
print(f"✓ Shape: {admissions.shape}")

# First fix nulls in preprocessing
df = pd.read_csv('data/processed/cleaned_dataset.csv')
print('Nulls before:', df.isnull().sum().sum())
df = df.dropna()
print('Nulls after:', df.isnull().sum().sum())
df.to_csv('data/processed/cleaned_dataset.csv', index=False)
print('Saved. Shape:', df.shape)

print("\nPREPROCESSING COMPLETE!")