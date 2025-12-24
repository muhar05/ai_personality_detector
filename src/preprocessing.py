import os
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load stopwords dari file
with open(os.path.join(BASE_DIR, 'stopwords.txt'), encoding='utf-8') as f:
    
    STOPWORDS = set(line.strip() for line in f if line.strip())

# Load slang dari file
SLANG = {}
with open(os.path.join(BASE_DIR, 'slangwords.txt'), encoding='utf-8') as f:
    for line in f:
        if line.strip() and ':' in line:
            slang, formal = line.strip().split(':', 1)
            SLANG[slang.strip()] = formal.strip()

NEGATIONS = {'tidak', 'bukan', 'nggak', 'gak', 'tdk', 'tak'}

def normalize_token(tok):
    return SLANG.get(tok, tok)

def tokenize(text):
    toks = re.findall(r'\w+', text.lower(), flags=re.UNICODE)
    toks = [normalize_token(t) for t in toks]
    return toks

def apply_negation(tokens, window=3):
    tokens = tokens[:]
    for i, t in enumerate(tokens):
        if t in NEGATIONS:
            for j in range(1, window+1):
                if i + j < len(tokens):
                    if not tokens[i + j].startswith('NOT_'):
                        tokens[i + j] = 'NOT_' + tokens[i + j]
    return tokens

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[@#]\w+', ' ', text)
    tokens = tokenize(text)
    tokens = apply_negation(tokens)
    processed = []
    for t in tokens:
        neg = False
        if t.startswith('NOT_'):
            neg = True
            t = t[4:]
        if not t or t.isdigit():
            continue
        stem = stemmer.stem(t)
        if stem in STOPWORDS:
            continue
        if neg:
            stem = 'NOT_' + stem
        processed.append(stem)
    return processed

# Preprocessing yang lebih ringan - jangan terlalu agresif
def simple_preprocess(text):
    # Basic cleaning only
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[@#]\w+', ' ', text)
    # Tokenize tapi jangan hapus terlalu banyak kata
    tokens = re.findall(r'\w+', text.lower())
    # Filter hanya kata yang sangat umum seperti 'dan', 'atau', 'di', 'ke', 'dari'
    basic_stops = {'dan', 'atau', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'yang', 'ini', 'itu'}
    filtered = [t for t in tokens if t not in basic_stops and len(t) > 2]
    return ' '.join(filtered)

def train_classifier(csv_path, model_out='personality_clf.joblib'):
    texts, labels, names = [], [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            names.append(row['name'])
            labels.append([
                int(row['openness']), int(row['conscientiousness']),
                int(row['extraversion']), int(row['agreeableness']), int(row['neuroticism'])
            ])
    
    print(f"Total data: {len(texts)}")
    
    # Preprocess dengan cara yang lebih ringan
    processed_texts = [simple_preprocess(t) for t in texts]
    
    # Check sample processed texts
    sample_processed = processed_texts[:5]
    print(f"Sample processed texts: {sample_processed}")
    
    # TF-IDF dengan parameter yang lebih permisif
    vec = TfidfVectorizer(
        max_features=1000, 
        ngram_range=(1,2), 
        lowercase=True, 
        min_df=1,        # Gunakan semua kata
        max_df=0.95      # Hanya buang kata yang terlalu sering
    )
    X = vec.fit_transform(processed_texts)
    y = np.array(labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Check label distribution
    for i, trait in enumerate(['openness','conscientiousness','extraversion','agreeableness','neuroticism']):
        trait_labels = y[:, i]
        print(f"{trait}: {np.sum(trait_labels)} positive out of {len(trait_labels)}")
    
    # Split dengan random state berbeda dan stratify yang lebih simple
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=123)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Classifier dengan parameter yang lebih aggressive untuk learning
    base = LogisticRegression(
        solver='liblinear', 
        max_iter=1000, 
        C=10.0,           # Lebih aggressive learning
        random_state=42,
        class_weight='balanced'  # Handle imbalanced data
    )
    clf = MultiOutputClassifier(base, n_jobs=1)
    
    print("Training classifier...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, 
                              target_names=['openness','conscientiousness','extraversion','agreeableness','neuroticism'],
                              zero_division=0))
    
    # Additional metrics
    print("\nValidation set label distribution:")
    for i, trait in enumerate(['openness','conscientiousness','extraversion','agreeableness','neuroticism']):
        val_labels = y_val[:, i]
        pred_labels = y_pred[:, i]
        print(f"{trait}: True={np.sum(val_labels)}, Predicted={np.sum(pred_labels)}")
    
    # Test dengan contoh teks manual
    print("\n=== Manual Test ===")
    test_texts = [
        "Saya suka mencoba hal baru dan kreatif",
        "Saya selalu disiplin dan tepat waktu", 
        "Saya senang berbicara dengan orang lain",
        "Saya suka membantu orang",
        "Saya sering merasa khawatir"
    ]
    
    for test_text in test_texts:
        processed = simple_preprocess(test_text)
        X_test = vec.transform([processed])
        pred = clf.predict(X_test)[0]
        print(f"Text: '{test_text}' -> {pred}")
    
    joblib.dump({'vec': vec, 'clf': clf, 'preprocess_func': simple_preprocess}, model_out, compress=3)
    print(f"\nModel saved to {model_out}")

def predict_with_model(text, model_path):
    data = joblib.load(model_path)
    vec = data['vec']
    clf = data['clf']
    
    # Gunakan fungsi preprocessing yang sama seperti saat training
    if 'preprocess_func' in data:
        proc = data['preprocess_func'](text)
    else:
        proc = simple_preprocess(text)  # fallback
    
    X = vec.transform([proc])
    pred = clf.predict(X)[0]
    names = ['openness','conscientiousness','extraversion','agreeableness','neuroticism']
    return dict(zip(names, map(int, pred)))

if __name__ == "__main__":
    train_classifier("../data/train.csv")