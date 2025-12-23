from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from chat_utils import parse_chat_per_user
from model import predict_with_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'personality_clf.joblib'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'success': False, 'error': 'Mohon upload file chat.'})
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        user_msgs = parse_chat_per_user(file_path)
        os.remove(file_path)
        results = {}
        for name, msgs in user_msgs.items():
            text = ' '.join(msgs)
            pred = predict_with_model(text, MODEL_PATH)
            results[name] = pred
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)