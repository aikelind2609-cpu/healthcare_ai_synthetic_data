import pandas as pd
import numpy as np
from ctgan import CTGAN
import pickle
import os

# Disable parallel processing
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

print("=" * 60)
print("CTGAN TRAINING")
print("=" * 60)

# Load cleaned data
print("\n[1/3] Loading data...")
df = pd.read_csv('data/processed/cleaned_dataset.csv')
print(f"✓ Loaded: {df.shape}")

# Initialize CTGAN
print("\n[2/3] Initializing CTGAN...")
ctgan = CTGAN(
    epochs=50,
    batch_size=10,
    enable_gpu=False
)
print(f"✓ CTGAN initialized")

# Train
print("\n[3/3] Training CTGAN (this may take 5-10 mins)...")
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {cat_cols}")
ctgan.fit(df, discrete_columns=cat_cols, epochs=50)
print(f"✓ Training complete")

# Save model
print("\nSaving model...")
with open('models/ctgan_model.pkl', 'wb') as f:
    pickle.dump(ctgan, f)
print(f"✓ Model saved")

# Generate synthetic data
print("\nGenerating synthetic data...")
synthetic_data = ctgan.sample(n=5000)
synthetic_data.to_csv('data/synthetic_data_ctgan.csv', index=False)
print(f"✓ Generated {len(synthetic_data)} synthetic records")

print("\n" + "=" * 60)
print("GAN TRAINING COMPLETE!")
print("=" * 60)