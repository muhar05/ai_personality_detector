# Arsitektur Sistem AI Personality Detector

Dokumen ini menjelaskan arsitektur sistem **AI Personality Detector**, yang menganalisis pola komunikasi dalam percakapan untuk mendeteksi kepribadian berdasarkan model OCEAN (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).

## Arsitektur Tingkat Tinggi

![Arsitektur Sistem](../diagrams/architecture.png)

### Komponen Utama

#### 1. **Browser / Web UI** (`src/templates/index.html`)
Antarmuka web yang memungkinkan pengguna untuk:
- Upload file chat WhatsApp dalam format `.txt`
- Input teks percakapan secara langsung
- Visualisasi hasil analisis OCEAN dalam bentuk radar chart

#### 2. **Flask App** (`src/web_app.py`)
Server aplikasi web yang menangani:
- **Endpoint `/`**: Menampilkan halaman utama
- **Endpoint `/analyze`**: Menerima file atau teks untuk dianalisis
- Orkestrasi proses analisis dari parsing hingga prediksi
- Mengembalikan hasil dalam format JSON

#### 3. **Chat Parser** (`src/chat_utils.py`)
Modul yang bertugas:
- Mem-parsing file chat WhatsApp dengan format `[DD/MM/YY, HH.MM.SS] Nama: Pesan`
- Mengelompokkan pesan berdasarkan pengirim
- Fungsi utama: `parse_chat_per_user(file_path)` → `{user: [msg...]}`

#### 4. **Cleaner** (`web_app.clean_chat_text()`)
Fungsi pembersih untuk input teks langsung:
- Menghapus timestamp dan nama pengirim dari format chat
- Mengekstrak konten pesan murni

#### 5. **Preprocessing NLP** (`src/preprocessing.py`)
Modul preprocessing teks dengan tahapan:
- Tokenisasi teks
- Normalisasi slang menggunakan `slangwords.txt`
- Penghapusan stopwords menggunakan `stopwords.txt`
- Stemming (opsional, menggunakan Sastrawi)
- Deteksi negasi
- Fungsi utama: `simple_preprocess(text)` → teks yang sudah diproses

#### 6. **TF-IDF Vectorizer**
Ekstraksi fitur teks menggunakan:
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Parameter: `max_features=2000`, `ngram_range=(1,2)`
- Mengubah teks menjadi vektor numerik untuk model ML

#### 7. **MultiOutputClassifier (Logistic Regression)**
Model machine learning dengan karakteristik:
- Multi-label classification untuk 5 trait OCEAN
- Base classifier: Logistic Regression dengan solver `liblinear`
- Parameter: `C=0.5`, `class_weight='balanced'`
- Fungsi utama: `predict_with_model(text, model_path)` → `{O,C,E,A,N}`

### Resources

- **stopwords.txt**: Daftar kata umum yang dihapus saat preprocessing
- **slangwords.txt**: Pemetaan kata slang ke kata formal
- **personality_clf.joblib**: Model ML terlatih (TF-IDF vectorizer + classifier)

## Alur Endpoint `/analyze`

![Sequence Diagram /analyze](../diagrams/analyze-sequence.png)

### Langkah-langkah Proses

1. **User mengirim request** ke `/analyze` dengan:
   - File chat (.txt), atau
   - Teks langsung melalui form

2. **Jika upload file**:
   - File disimpan sementara ke direktori `uploads/` dengan `secure_filename`
   - File di-parse menggunakan `parse_chat_per_user(file_path)`
   - Hasil: `{user1: [msg...], user2: [msg...], ...}`
   - File sementara dihapus setelah parsing

3. **Jika input teks langsung**:
   - Teks dibersihkan menggunakan `clean_chat_text(text)`
   - Hasil: `{"User": gabungan_teks}`

4. **Loop untuk setiap user**:
   - Gabungkan semua pesan user menjadi satu teks
   - Panggil `predict_with_model(text, MODEL_PATH)`:
     - Load model dari `personality_clf.joblib`
     - Preprocess teks menggunakan `simple_preprocess(text)`
     - Transform dengan TF-IDF vectorizer
     - Prediksi menggunakan classifier
     - Return: `{openness: 0/1, conscientiousness: 0/1, extraversion: 0/1, agreeableness: 0/1, neuroticism: 0/1}`

5. **Response JSON** dikirim ke user:
   ```json
   {
     "success": true,
     "results": {
       "User1": {"openness": 1, "conscientiousness": 0, ...},
       "User2": {"openness": 0, "conscientiousness": 1, ...}
     }
   }
   ```

## Catatan Penting

### ⚠️ Hardcoded Model Path

Saat ini, `MODEL_PATH` di `web_app.py` masih **hardcoded** ke path absolut lokal:

```python
MODEL_PATH = r'E:\muhar\Tugas Matkul\S5\KECERDASAN BUATAN\ai_personality_detector\src\personality_clf.joblib'
```

**Rekomendasi**: Untuk deployment yang lebih robust, gunakan path relatif:

```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'personality_clf.joblib')
```

### Model Machine Learning

- **Algoritma**: Logistic Regression dengan Multi-Output Classification
- **Input**: Teks percakapan dalam Bahasa Indonesia
- **Output**: 5 label binary untuk trait OCEAN (0 = tidak dominan, 1 = dominan)
- **Training**: Menggunakan data labeled dalam format CSV
- **Evaluasi**: F1-score dan classification report

### Preprocessing

Preprocessing menggunakan pendekatan **lightweight** untuk mempertahankan lebih banyak informasi:
- Hanya menghapus stopwords yang sangat umum (dan, atau, di, ke, dari, dll)
- Tidak melakukan stemming yang terlalu agresif
- Mendukung deteksi negasi untuk meningkatkan akurasi

### Upload Security

- Menggunakan `secure_filename` dari Werkzeug untuk mencegah path traversal
- File sementara dihapus setelah diproses
- Direktori `uploads/` dibuat otomatis jika belum ada

## Pengembangan Lebih Lanjut

Beberapa area yang dapat ditingkatkan:

1. **Model Path**: Gunakan path relatif atau environment variable
2. **Error Handling**: Tambahkan validasi untuk format file chat yang tidak sesuai
3. **Caching**: Cache model ML agar tidak perlu load ulang setiap request
4. **Async Processing**: Untuk file besar, gunakan background job (Celery, RQ)
5. **Logging**: Tambahkan logging yang lebih terstruktur
6. **API Documentation**: Tambahkan Swagger/OpenAPI documentation
7. **Testing**: Tambahkan unit test dan integration test

## Referensi

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Sastrawi Stemmer](https://github.com/sastrawi/sastrawi)
- [OCEAN Personality Model](https://en.wikipedia.org/wiki/Big_Five_personality_traits)
