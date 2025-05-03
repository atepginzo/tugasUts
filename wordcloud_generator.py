import pandas as pd
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Path input dan output
input_file = 'E:/Tugas UTS/preprocessed_data.csv'
output_dir = 'E:/Tugas UTS/static'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(input_file)

# Fungsi bantu untuk membuat wordcloud
def generate_wordcloud(text, sentiment_label):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'wordcloud_{sentiment_label}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Wordcloud {sentiment_label} disimpan di {output_path}")

# Pisahkan komentar berdasarkan sentimen
for label in df['sentimen'].unique():
    filtered = df[df['sentimen'] == label]
    generate_wordcloud(filtered['cleaned_komentar'], label)
