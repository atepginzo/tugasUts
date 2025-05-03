from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Inisialisasi Flask
app = Flask(__name__)

# =======================
# Load model & tokenizer
# =======================
model = tf.keras.models.load_model('model/lstm_model.h5')

with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load dataset
df = pd.read_csv('preprocessed_data.csv')
sample_data = df[['platform', 'komentar', 'timestamp', 'sentimen']].head(10).to_dict(orient='records')
# =======================
# Fungsi prediksi sentimen
# =======================
def predict_comment(comment, max_len=100):
    sequence = tokenizer.texts_to_sequences([comment])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded)[0]
    label_index = np.argmax(prediction)
    labels = ['Negatif', 'Netral', 'Positif']
    return labels[label_index]

# =======================
# Route utama
# =======================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    user_comment = ""

    if request.method == 'POST':
        user_comment = request.form['comment']
        prediction_result = predict_comment(user_comment)

    return render_template('index.html',
                           comment=user_comment,
                           prediction=prediction_result,
                           sample_data=sample_data,
                           charts={
                               'sentimen': 'static/visual_sentimen.png',
                               'platform': 'static/visual_platform.png',
                               'per_jam': 'static/visual_per_jam.png',
                               'per_hari': 'static/visual_per_hari.png',
                               'confusion': 'static/confusionMatrix.png',
                               'wordcloud_positif': 'static/wordcloud_positif.png',
                               'wordcloud_negatif': 'static/wordcloud_negatif.png',
                               'wordcloud_netral': 'static/wordcloud_netral.png'
                           })

# =======================
# Jalankan aplikasi
# =======================
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
