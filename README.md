# Sistem Analisis Pola Komunikasi Percakapan Berbasis AI Dengan Konsep OCEAN

## Deskripsi Proyek

Sistem ini merupakan aplikasi berbasis AI yang menganalisis pola komunikasi percakapan (chat) untuk mendeteksi kepribadian pengguna berdasarkan model OCEAN (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism). Sistem dapat digunakan untuk menganalisis chat WhatsApp atau teks percakapan lain, baik secara manual maupun melalui antarmuka web.

## Fitur Utama

- **Analisis kepribadian OCEAN** dari teks percakapan.
- **Preprocessing otomatis**: normalisasi slang, stopwords, stemming, dan deteksi negasi.
- **Model Machine Learning**: Multi-label classification menggunakan Logistic Regression.
- **Antarmuka Web**: Upload file chat dan dapatkan hasil analisis secara interaktif.
- **Visualisasi hasil**: Radar chart OCEAN pada web.

## Arsitektur Sistem

Untuk memahami arsitektur dan alur kerja sistem secara detail, silakan lihat [Dokumentasi Arsitektur](docs/ARCHITECTURE.md).

## Struktur Folder

```
ai_personality_detector/
│
├── README.md
├── requirements.txt
├── slangwords.txt
├── stopwords.txt
├── data/
│   └── chat_sample_labeled.txt
├── diagrams/
│   ├── architecture.mmd
│   ├── architecture.png
│   ├── architecture.jpg
│   ├── analyze-sequence.mmd
│   ├── analyze-sequence.png
│   └── analyze-sequence.jpg
├── docs/
│   └── ARCHITECTURE.md
├── src/
│   ├── app.py
│   ├── chat_utils.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── web_app.py
│   └── templates/
│       └── index.html
├── uploads/
```

## Setup & Instalasi

1. **Clone repository**

   ```
   git clone <repo-url>
   cd ai_personality_detector
   ```

2. **Buat dan aktifkan virtual environment (opsional)**

   ```
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate # Linux/Mac
   ```

3. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

4. **Download NLTK stopwords (otomatis di app.py), atau manual:**
   ```
   python -m nltk.downloader stopwords
   ```

## Library yang Digunakan

- **nltk**: Stopwords dan preprocessing teks.
- **Sastrawi**: Stemming Bahasa Indonesia.
- **scikit-learn**: Machine Learning (TF-IDF, Logistic Regression, MultiOutputClassifier).
- **joblib**: Menyimpan dan memuat model ML.
- **matplotlib**: Visualisasi (opsional).
- **flask**: Web server.
- **werkzeug**: Keamanan upload file.

## Cara Penggunaan

### 1. Analisis Chat via Script

- Jalankan [src/app.py](src/app.py) untuk analisis file chat:
  ```
  python src/app.py --chat data/chat_sample.txt --model personality_clf.joblib
  ```
  - Hasil analisis akan ditampilkan di terminal.

### 2. Melatih Model Machine Learning

- Jalankan [src/model.py](src/model.py) untuk melatih model:

  ```
  python src/model.py
  ```

  - Model akan disimpan sebagai `personality_clf.joblib`.

- Format data latih: CSV dengan kolom `id,text,openness,conscientiousness,extraversion,agreeableness,neuroticism`.

### 3. Analisis Chat via Web

- Jalankan web server:
  ```
  python src/web_app.py
  ```
- Buka browser ke `http://localhost:5000`
- Upload file chat WhatsApp (.txt) atau masukkan teks manual.
- Hasil analisis OCEAN akan ditampilkan beserta radar chart.

## Penjelasan Model Machine Learning

- **Preprocessing**: Teks diubah menjadi token, slang dinormalisasi, stopwords dihapus, kata distem menggunakan Sastrawi, dan negasi dideteksi.
- **Ekstraksi Fitur**: Menggunakan TF-IDF Vectorizer (max_features=4000, ngram_range=(1,2)).
- **Klasifikasi**: Multi-label classification dengan Logistic Regression (solver='liblinear', max_iter=200), dibungkus dengan MultiOutputClassifier.
- **Evaluasi**: Menggunakan F1-score dan classification report.
- **Prediksi**: Model memprediksi skor 0/1 untuk setiap trait OCEAN.

## Format File Chat

- File chat WhatsApp (.txt) dengan format:
  ```
  [tanggal, jam] Nama Pengirim: Pesan
  ```
- Contoh: [data/chat_sample_labeled.txt](data/chat_sample_labeled.txt)

## Konsep OCEAN

- **Openness**: Keterbukaan terhadap pengalaman baru, kreativitas.
- **Conscientiousness**: Disiplin, ketelitian, tanggung jawab.
- **Extraversion**: Sosialisasi, semangat, bicara.
- **Agreeableness**: Keramahan, empati, kerja sama.
- **Neuroticism**: Kecemasan, emosi negatif.

## Kontribusi & Pengembangan

- Tambahkan kata slang dan stopwords di `slangwords.txt` dan `stopwords.txt`.
- Kembangkan lexicon OCEAN di [src/app.py](src/app.py).
- Ganti/latih model dengan data yang lebih besar untuk akurasi lebih baik.

## Lisensi

Proyek ini untuk keperluan akademik dan pembelajaran.

---

**Kontak:**  
Untuk pertanyaan dan saran, silakan hubungi pengembang melalui email atau issues di repository.
