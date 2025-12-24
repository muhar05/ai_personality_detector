import re
import os
import argparse
import csv
import sys
from collections import Counter
import matplotlib.pyplot as plt

# Minimal external libs
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import joblib

from chat_utils import read_chat_file
from preprocessing import preprocess_text

# ------------------ Utilities ------------------
# Ensure NLTK stopwords available (download if missing)
try:
    _ = nltk.corpus.stopwords.words('indonesian')
except Exception:
    nltk.download('stopwords')

STOPWORDS = set(nltk.corpus.stopwords.words('indonesian'))

# Simple slang normalization (extend as needed)
SLANG = {
    'gk': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'ga': 'tidak',
    'gw': 'saya', 'gue': 'saya', 'lo': 'kamu', 'kmu': 'kamu',
    'sm': 'sama', 'sama2': 'sama', 'tdk': 'tidak'
}

NEGATIONS = {'tidak', 'bukan', 'nggak', 'gak', 'tdk', 'tak'}

def normalize_token(tok):
    return SLANG.get(tok, tok)

def tokenize(text):
    # keep unicode word characters, drop punctuation
    toks = re.findall(r'\w+', text.lower(), flags=re.UNICODE)
    toks = [normalize_token(t) for t in toks]
    return toks

def apply_negation(tokens, window=3):
    # prefix next `window` tokens with NOT_ to capture negation effect
    tokens = tokens[:]  # copy
    for i, t in enumerate(tokens):
        if t in NEGATIONS:
            for j in range(1, window+1):
                if i + j < len(tokens):
                    if not tokens[i + j].startswith('NOT_'):
                        tokens[i + j] = 'NOT_' + tokens[i + j]
    return tokens

# ------------------ Preprocessing ------------------
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    # remove urls and mentions quickly
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

# ------------------ Lexicon (simple, stemmed) ------------------
# Note: stem your lexicon words with Sastrawi if you expand it
RAW_TRAITS = {
    "openness": ["imajinasi", "ide", "baru", "pikir", "kreatif"],
    "conscientiousness": ["disiplin", "tepat", "rapi", "teratur", "kerja"],
    "extraversion": ["teman", "bicara", "senang", "ramai", "ngobrol", "koneksi"],
    "agreeableness": ["bantu", "baik", "peduli", "teman", "sopan"],
    "neuroticism": ["cemas", "khawatir", "takut", "gelisah", "panik"]
}

# Stem the lexicon once at startup
TRAITS = {}
for trait, words in RAW_TRAITS.items():
    stemmed = []
    for w in words:
        w_stem = stemmer.stem(w)
        stemmed.append(w_stem)
    TRAITS[trait] = stemmed

def lexicon_scores(tokens):
    counts = Counter(tokens)
    scores = {}
    for trait, words in TRAITS.items():
        score = sum(counts[w] for w in words if w in counts)
        scores[trait] = score
    # normalize to 0..1 by dividing by max (safe)
    mx = max(scores.values()) if max(scores.values()) > 0 else 1
    norm = {k: v / mx for k, v in scores.items()}
    return scores, norm

# ------------------ Lightweight classifier (optional) ------------------
def train_classifier(csv_path, model_out='personality_clf.joblib'):
    # CSV expected columns: id,text,openness,conscientiousness,extraversion,agreeableness,neuroticism
    texts = []
    labels = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = ['text', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for c in required:
            if c not in reader.fieldnames:
                print(f"CSV missing required column: {c}")
                return
        for row in reader:
            texts.append(row['text'])
            labels.append([int(row[c]) for c in required[1:]] if False else [
                int(row['openness']), int(row['conscientiousness']),
                int(row['extraversion']), int(row['agreeableness']), int(row['neuroticism'])
            ])
    if not texts:
        print("No data found in CSV.")
        return

    # Preprocess texts (lightweight, may be slower on CPU but OK for small datasets)
    processed_texts = [' '.join(preprocess_text(t)) for t in texts]

    # TF-IDF vectorizer limited to reduce memory usage
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1,2), lowercase=True)
    X = vec.fit_transform(processed_texts)

    y = labels

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # Multi-output classifier with light solver; single-threaded to save RAM/CPU
    base = LogisticRegression(solver='liblinear', max_iter=200)
    clf = MultiOutputClassifier(base, n_jobs=1)
    print("Training classifier (this may take some minutes on CPU)...")
    clf.fit(X_train, y_train)

    # Evaluate quickly
    y_pred = clf.predict(X_val)
    print("Validation F1 (macro) per label:")
    try:
        print(f1_score(y_val, y_pred, average=None))
    except Exception:
        pass
    print(classification_report(y_val, y_pred, target_names=['openness','conscientiousness','extraversion','agreeableness','neuroticism']))

    # Save model and vectorizer
    joblib.dump({'vec': vec, 'clf': clf}, model_out, compress=3)
    print(f"Model saved to {model_out}")

def predict_with_model(text, model_path):
    data = joblib.load(model_path)
    vec = data['vec']
    clf = data['clf']
    proc = ' '.join(preprocess_text(text))
    X = vec.transform([proc])
    pred = clf.predict(X)[0]
    names = ['openness','conscientiousness','extraversion','agreeableness','neuroticism']
    return dict(zip(names, map(int, pred)))

def interpret_trait(trait):
    mapping = {
        "openness": "Terbuka pada pengalaman baru",
        "conscientiousness": "Teliti & Disiplin",
        "extraversion": "Ekstrovert",
        "agreeableness": "Mudah bergaul & ramah",
        "neuroticism": "Mudah cemas"
    }
    return mapping.get(trait, trait)

# ------------------ Main script behavior ------------------
def analyze_chat_file(chat_path, model_path=None, show_plot=False):
    name, messages, text = read_chat_file(chat_path)
    tokens = preprocess_text(text)
    raw_scores, normalized = lexicon_scores(tokens)

    print(f"\nNama Pengirim: {name}")
    print("\n=== Personality Scores (normalized) ===")
    for k, v in normalized.items():
        print(f"{k.capitalize()}: {v:.2f}")

    max_trait = max(normalized, key=normalized.get)
    if normalized[max_trait] > 0:
        print(f"\nInterpretasi: {name} cenderung memiliki kepribadian {interpret_trait(max_trait)}.")
    else:
        print(f"\nInterpretasi: Tidak ditemukan kecenderungan kepribadian yang dominan pada {name}.")

    if model_path:
        print("\nModel-based prediction:")
        try:
            pred = predict_with_model(text, model_path)
            for k, v in pred.items():
                print(f"{k}: {v}")
        except Exception as e:
            print("Model prediction failed:", e)

def main():
    parser = argparse.ArgumentParser(description="Lightweight Personality Detector (CPU-friendly)")
    parser.add_argument('--chat', help='Path to chat text file', required=True)
    parser.add_argument('--train_csv', help='Optional: path to labeled CSV to train model')
    parser.add_argument('--model_out', help='Output path for saved model', default='personality_clf.joblib')
    parser.add_argument('--use_model', help='Optional: path to saved model to run predictions', default=None)
    args = parser.parse_args()

    if args.train_csv:
        train_classifier(args.train_csv, model_out=args.model_out)

    analyze_chat_file(args.chat, model_path=args.use_model)

if __name__ == '__main__':
    main()