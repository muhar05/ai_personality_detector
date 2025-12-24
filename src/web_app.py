from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from chat_utils import parse_chat_per_user
from model import predict_with_model
import traceback
import json
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = r'E:\muhar\Tugas Matkul\S5\KECERDASAN BUATAN\ai_personality_detector\src\personality_clf.joblib'

def clean_chat_text(text):
    """
    Menghapus timestamp dan nama pengirim dari format chat WhatsApp.
    """
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        m = re.match(r"\[\d{2}/\d{2}/\d{2}, \d{2}\.\d{2}\.\d{2}\] [^:]+: (.+)", line.strip())
        if m:
            cleaned.append(m.group(1))
        else:
            cleaned.append(line.strip())
    return ' '.join(cleaned)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Jika file diupload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            user_msgs = parse_chat_per_user(file_path)
            os.remove(file_path)
            results = {}
            for name, msgs in user_msgs.items():
                text = ' '.join(msgs)
                print(f"\n=== ANALYZING USER: {name} ===")
                print(f"Text length: {len(text)} characters")
                print(f"Text sample: {text[:200]}...")
                pred = predict_with_model(text, MODEL_PATH)
                print(f"Prediction result: {pred}")
                print(f"Prediction type: {type(pred)}")
                # Ensure all values are regular Python integers
                clean_pred = {k: int(v) for k, v in pred.items()}
                print(f"Cleaned prediction: {clean_pred}")
                results[name] = clean_pred
            
            response_data = {'success': True, 'results': results}
            print(f"Final response: {response_data}")
            return jsonify(response_data)
            
        # Jika teks dikirim langsung
        elif 'text' in request.form and request.form['text'].strip():
            text = request.form['text']
            print(f"\n=== ANALYZING DIRECT TEXT ===")
            print(f"Text length: {len(text)} characters")
            print(f"Text sample: {text[:200]}...")
            cleaned_text = clean_chat_text(text)
            print(f"Cleaned text: {cleaned_text[:200]}...")
            # (opsional) processed_text = simple_preprocess(cleaned_text)
            pred = predict_with_model(cleaned_text, MODEL_PATH)
            print(f"Prediction result: {pred}")
            print(f"Prediction type: {type(pred)}")
            
            # Ensure all values are regular Python integers  
            clean_pred = {k: int(v) for k, v in pred.items()}
            print(f"Cleaned prediction: {clean_pred}")
            results = {"User": clean_pred}
            
            response_data = {'success': True, 'results': results}
            print(f"Final response: {response_data}")
            return jsonify(response_data)
        else:
            return jsonify({'success': False, 'error': 'Mohon upload file chat atau masukkan teks.'})
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)