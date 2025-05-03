import pandas as pd
import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical

# Path input dan output
input_file = 'E:/Tugas UTS/preprocessed_data.csv'
model_dir = 'E:/Tugas UTS/model'
model_output = os.path.join(model_dir, 'lstm_model.h5')
tokenizer_output = os.path.join(model_dir, 'tokenizer.pkl')
label_classes_output = os.path.join(model_dir, 'label_classes.npy')

# Buat folder model jika belum ada
os.makedirs(model_dir, exist_ok=True)

# Load data
df = pd.read_csv(input_file)

# Encode label sentimen
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['sentimen'])
y = to_categorical(df['label'])

# Tokenisasi teks
MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_komentar'])
X = tokenizer.texts_to_sequences(df['cleaned_komentar'])
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model LSTM
model = Sequential()
model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=128, input_length=MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

# Kompilasi dan training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
model.summary()
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n📊 Akurasi data test: {accuracy:.4f}")

# ==========================
# Tambahan: Confusion Matrix dan Metrics
# ==========================

# Prediksi kelas
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
labels = label_encoder.classes_  # ['negatif', 'netral', 'positif'] atau sesuai data

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=labels))

# ==========================
# Simpan model dan tools
# ==========================
model.save(model_output)

with open(tokenizer_output, 'wb') as f:
    pickle.dump(tokenizer, f)

np.save(label_classes_output, label_encoder.classes_)

print(f"✅ Model disimpan di {model_output}")
print(f"✅ Tokenizer disimpan di {tokenizer_output}")
print(f"✅ Label encoder classes disimpan di {label_classes_output}")
