import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ctgan import CTGAN
from scipy.stats import ks_2samp
import json

os.environ['LOKY_MAX_CPU_COUNT'] = '1'

print("=" * 60)
print("STEP 3: GAN TRAINING (CTGAN + WGAN-GP)")
print("=" * 60)

# =============================================
# LOAD RAW DATA (not normalized - better for CTGAN)
# =============================================
print("\n[1/6] Loading RAW data...")
df_raw = pd.read_csv('data/raw/rare_disease_ehr.csv')

# Drop HAS_RARE_DISEASE (it's a label, not a feature)
gan_train = df_raw.drop('HAS_RARE_DISEASE', axis=1)
cat_cols = gan_train.select_dtypes(include=['object']).columns.tolist()
print(f"✓ Training data: {gan_train.shape}")
print(f"✓ Categorical: {cat_cols}")

# =============================================
# PART A: CTGAN
# =============================================
print("\n" + "=" * 60)
print("PART A: CTGAN TRAINING (300 epochs)")
print("=" * 60)

ctgan = CTGAN(epochs=300, batch_size=10, enable_gpu=False)
ctgan.fit(gan_train, discrete_columns=cat_cols, epochs=300)
print("✓ CTGAN training complete")

os.makedirs('models', exist_ok=True)
with open('models/ctgan_model.pkl', 'wb') as f:
    pickle.dump(ctgan, f)

synthetic_ctgan = ctgan.sample(n=len(gan_train))
synthetic_ctgan.to_csv('data/ctgan_synthetic.csv', index=False)
print(f"✓ Generated {len(synthetic_ctgan)} CTGAN records")

# =============================================
# PART B: WGAN-GP
# =============================================
print("\n" + "=" * 60)
print("PART B: WGAN-GP TRAINING (200 epochs)")
print("=" * 60)

X = gan_train.select_dtypes(include=[np.number]).values.astype(np.float32)

# Normalize for WGAN-GP
from sklearn.preprocessing import StandardScaler
wgan_scaler = StandardScaler()
X_scaled = wgan_scaler.fit_transform(X)

input_dim = X_scaled.shape[1]
latent_dim = 64

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

def gradient_penalty(disc, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1).to(device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = disc(interpolates)
    fake_out = torch.ones(batch_size, 1).to(device)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake_out, create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

device = torch.device('cpu')
gen = Generator(latent_dim, input_dim).to(device)
disc = Discriminator(input_dim).to(device)
g_opt = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.9))
d_opt = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.9))

dataset = TensorDataset(torch.FloatTensor(X_scaled))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(200):
    for batch_idx, (real_data,) in enumerate(dataloader):
        real_data = real_data.to(device)
        bs = real_data.size(0)
        for _ in range(5):
            disc.zero_grad()
            z = torch.randn(bs, latent_dim).to(device)
            fake_data = gen(z)
            d_loss = -disc(real_data).mean() + disc(fake_data.detach()).mean() + 10 * gradient_penalty(disc, real_data, fake_data.detach(), device)
            d_loss.backward()
            d_opt.step()
        gen.zero_grad()
        z = torch.randn(bs, latent_dim).to(device)
        g_loss = -disc(gen(z)).mean()
        g_loss.backward()
        g_opt.step()
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/200 | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("✓ WGAN-GP training complete")

torch.save(gen.state_dict(), 'models/wgan_gp_generator.pt')
torch.save(disc.state_dict(), 'models/wgan_gp_discriminator.pt')

gen.eval()
with torch.no_grad():
    z = torch.randn(len(gan_train), latent_dim)
    wgan_synthetic_scaled = gen(z).numpy()

wgan_synthetic = wgan_scaler.inverse_transform(wgan_synthetic_scaled)
num_cols = gan_train.select_dtypes(include=[np.number]).columns.tolist()
wgan_df = pd.DataFrame(wgan_synthetic, columns=num_cols)
wgan_df.to_csv('data/wgan_gp_synthetic.csv', index=False)
print(f"✓ Generated {len(wgan_df)} WGAN-GP records")

# =============================================
# PART C: COMPARE
# =============================================
print("\n" + "=" * 60)
print("PART C: GAN COMPARISON")
print("=" * 60)

real_num = gan_train.select_dtypes(include=[np.number])

ctgan_ks = []
for col in real_num.columns:
    if col in synthetic_ctgan.columns:
        ks, _ = ks_2samp(real_num[col], synthetic_ctgan[col])
        ctgan_ks.append(ks)

wgan_ks = []
for col in real_num.columns:
    if col in wgan_df.columns:
        ks, _ = ks_2samp(real_num[col], wgan_df[col])
        wgan_ks.append(ks)

ctgan_avg = np.mean(ctgan_ks) if ctgan_ks else 1.0
wgan_avg = np.mean(wgan_ks) if wgan_ks else 1.0

winner = "CTGAN" if ctgan_avg < wgan_avg else "WGAN-GP"
print(f"  CTGAN avg KS: {ctgan_avg:.4f}")
print(f"  WGAN-GP avg KS: {wgan_avg:.4f}")
print(f"  Winner: {winner}")

os.makedirs('results', exist_ok=True)
with open('results/gan_comparison.json', 'w') as f:
    json.dump({"ctgan_ks": float(ctgan_avg), "wgan_gp_ks": float(wgan_avg), "winner": winner}, f, indent=2)

print("\n" + "=" * 60)
print("GAN TRAINING COMPLETE!")
print("=" * 60)