import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

print("=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)

print("\n[1/5] Loading data...")
df = pd.read_csv('data/raw/rare_disease_ehr.csv')
print(f"✓ Loaded: {df.shape}")

print("\n[2/5] Handling missing values...")
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
print(f"✓ Nulls: {df.isnull().sum().sum()}")

print("\n[3/5] Encoding categorical variables...")
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"  {col}: {len(le.classes_)} categories")

print("\n[4/5] Normalizing...")
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(f"✓ Normalized")

print("\n[5/5] Saving...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

df.to_csv('data/processed/cleaned_dataset.csv', index=False)
train_df.to_csv('data/processed/train_dataset.csv', index=False)
test_df.to_csv('data/processed/test_dataset.csv', index=False)

print(f"✓ Full: {df.shape}")
print(f"✓ Train: {train_df.shape}")
print(f"✓ Test: {test_df.shape}")
print("=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)