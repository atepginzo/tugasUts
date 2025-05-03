import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path input dan output
input_file = 'E:/Tugas UTS/preprocessed_data.csv'
output_dir = 'E:/Tugas UTS/static'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(input_file)

# Konversi kolom timestamp menjadi datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Tambahan kolom waktu
df['tanggal'] = df['timestamp'].dt.date
df['jam'] = df['timestamp'].dt.hour

# Set style seaborn
sns.set(style='whitegrid')

# =======================
# 1. Distribusi Sentimen
# =======================
plt.figure(figsize=(6, 4))
sns.countplot(x='sentimen', data=df, palette='Set2')
plt.title('Distribusi Sentimen')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Komentar')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'visual_sentimen.png'))
plt.close()

# ===========================
# 2. Aktivitas Komentar per Hari
# ===========================
plt.figure(figsize=(8, 4))
komentar_per_hari = df.groupby('tanggal').size()
komentar_per_hari.plot(kind='line', marker='o', color='blue')
plt.title('Aktivitas Komentar per Hari')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Komentar')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'visual_per_hari.png'))
plt.close()

# ===========================
# 3. Aktivitas Komentar per Jam
# ===========================
plt.figure(figsize=(8, 4))
sns.countplot(x='jam', data=df, palette='coolwarm')
plt.title('Aktivitas Komentar per Jam')
plt.xlabel('Jam (0–23)')
plt.ylabel('Jumlah Komentar')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'visual_per_jam.png'))
plt.close()

# ===========================
# 4. Jumlah Komentar per Platform
# ===========================
plt.figure(figsize=(6, 4))
sns.countplot(x='platform', data=df, palette='pastel')
plt.title('Jumlah Komentar per Platform')
plt.xlabel('Platform')
plt.ylabel('Jumlah Komentar')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'visual_platform.png'))
plt.close()

print(" Semua visualisasi berhasil disimpan ke folder static/")
