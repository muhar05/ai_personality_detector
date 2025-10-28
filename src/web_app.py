from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from chat_utils import read_chat_file
from preprocessing import preprocess_text
from lexicon import lexicon_scores

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Buat folder upload jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk analisis personality (API)
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        chat_text = ""
        
        # Cek apakah ada file yang di-upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Baca file berdasarkan ekstensi
                if filename.lower().endswith('.txt'):
                    # Proses file .txt langsung
                    name, messages, text = read_chat_file(file_path)
                    chat_text = text
                elif filename.lower().endswith('.csv'):
                    # Proses file .csv (ambil kolom 'text' atau 'message')
                    import csv
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        texts = []
                        for row in reader:
                            # Coba ambil dari kolom yang mungkin
                            if 'text' in row:
                                texts.append(row['text'])
                            elif 'message' in row:
                                texts.append(row['message'])
                            elif 'content' in row:
                                texts.append(row['content'])
                        chat_text = ' '.join(texts)
                
                # Hapus file setelah diproses
                os.remove(file_path)
            else:
                return jsonify({'success': False, 'error': 'File tidak didukung. Gunakan .txt atau .csv'})
        
        # Jika tidak ada file, ambil dari textarea
        elif 'chat_text' in request.form and request.form['chat_text'].strip():
            chat_text = request.form['chat_text']
        else:
            return jsonify({'success': False, 'error': 'Mohon masukkan teks atau upload file'})
        
        # Jika menggunakan textarea, buat file sementara
        if 'chat_text' in request.form and request.form['chat_text'].strip():
            temp_file = 'temp_chat.txt'
            with open(temp_file, 'w', encoding='utf-8') as f:
                lines = chat_text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        f.write(f"[10/11, 08:{i:02d}] User: {line.strip()}\n")
            
            name, messages, text = read_chat_file(temp_file)
            os.remove(temp_file)
        else:
            # Jika dari file, langsung proses
            text = chat_text
        
        # Proses analisis
        tokens = preprocess_text(text)
        raw_scores, normalized = lexicon_scores(tokens)
        
        # Tentukan trait dominan
        trait_labels = {
            "openness": "Openness (Terbuka pada pengalaman baru)",
            "conscientiousness": "Conscientiousness (Teliti & Disiplin)",
            "extraversion": "Extraversion (Ekstrovert)",
            "agreeableness": "Agreeableness (Mudah bergaul & ramah)",
            "neuroticism": "Neuroticism (Mudah cemas)"
        }
        max_trait = max(normalized, key=normalized.get)
        
        result = {
            'scores': normalized,
            'dominant_trait': trait_labels[max_trait] if normalized[max_trait] > 0 else "Tidak ada trait dominan",
            'success': True
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)