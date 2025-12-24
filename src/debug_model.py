import sys
import os

# Add proper path
sys.path.append('.')

from model import predict_with_model
from preprocessing import simple_preprocess
import joblib

print("=== DEBUGGING MODEL PREDICTIONS ===")
print(f"Current working directory: {os.getcwd()}")

# Fix path - model is in current directory, not src subdirectory
model_path = 'personality_clf.joblib'

if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    print("Available files:")
    for f in os.listdir('.'):
        if f.endswith('.joblib'):
            print(f"  📁 {f}")
    exit(1)

try:
    model_data = joblib.load(model_path)
    print(f"✅ Model loaded successfully from {model_path}!")
    print(f"Model keys: {model_data.keys()}")
    
    if 'vec' in model_data:
        vec = model_data['vec']
        print(f"Vocabulary size: {len(vec.vocabulary_)}")
        print(f"Sample vocabulary: {list(vec.vocabulary_.items())[:10]}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Test with chat dari game
chat_text = """saldo bisa dicari skin limited susah kalimat berbahaya ini wkwkwk fakta lapangan bang yaudah malam ini gas spin dikit ingat rumusnya noted kalo dapet epic aku traktir kopi deal screenshot jangan lupa lah dapet elite doang lanjut lagi bang tanggung ini nih setannya muncul bentar topup dulu wkwkwk aman terkendali OKE DAPET EPIC NAHHH kan aku bilang mental langsung naik fix winrate ikut naik besok rank bareng ya gas jangan first pick aneh aneh santai guin martis aman siap epical glory jangan dihina dong becanda bang yang penting skin nyala skin nyala gameplay menyusul"""

print(f"\n=== TESTING WITH GAME CHAT ===")
result = predict_with_model(chat_text, model_path)
print(f"Game chat result: {result}")
print(f"Any non-zero?: {any(v > 0 for v in result.values())}")

# Test individual traits
test_texts = [
    "Saya suka mencoba hal baru dan kreatif",
    "Saya selalu disiplin tepat waktu terorganisir", 
    "Saya senang berbicara berinteraksi banyak orang",
    "Saya suka membantu orang lain ramah baik peduli",
    "Saya sering merasa khawatir cemas takut gelisah"
]

print(f"\n=== TESTING INDIVIDUAL TRAITS ===")
for i, text in enumerate(test_texts):
    result = predict_with_model(text, model_path)
    print(f"Text {i+1}: {text}")
    print(f"Result: {result}")
    print(f"Any non-zero?: {any(v > 0 for v in result.values())}")
    print()