# preprocess.py
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download resource NLTK (jika belum ada)
nltk.download('stopwords')

# Load stopwords Indonesia
stop_words = set(stopwords.words('indonesian'))

# Inisialisasi stemmer bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # hapus link
    text = re.sub(r'[^\w\s]', '', text)  # hapus tanda baca
    text = re.sub(r'\d+', '', text)  # hapus angka
    text = text.strip()
    return text

# Fungsi preprocessing

def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = cleaned.split()
    tokens = [t for t in tokens if t not in stop_words]
    stemmed = [stemmer.stem(t) for t in tokens]
    return ' '.join(stemmed)

# Path file
input_file = 'E:/Tugas UTS/data/data_platform_gusmiftah.csv'
output_file = 'E:/Tugas UTS/preprocessed_data.csv'

# Cek apakah file ada
if not os.path.exists(input_file):
    raise FileNotFoundError(f"File {input_file} tidak ditemukan.")

# Baca dan proses
df = pd.read_csv(input_file)
df['cleaned_komentar'] = df['komentar'].astype(str).apply(preprocess_text)

# Simpan hasil
df.to_csv(output_file, index=False)
print(f"✅ Preprocessing selesai. File disimpan sebagai:\n{output_file}")