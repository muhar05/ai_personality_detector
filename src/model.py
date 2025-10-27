import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from preprocessing import preprocess_text

def train_classifier(csv_path, model_out='personality_clf.joblib'):
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
            labels.append([
                int(row['openness']), int(row['conscientiousness']),
                int(row['extraversion']), int(row['agreeableness']), int(row['neuroticism'])
            ])
    if not texts:
        print("No data found in CSV.")
        return

    processed_texts = [' '.join(preprocess_text(t)) for t in texts]
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1,2), lowercase=True)
    X = vec.fit_transform(processed_texts)
    y = labels

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    base = LogisticRegression(solver='liblinear', max_iter=200)
    clf = MultiOutputClassifier(base, n_jobs=1)
    print("Training classifier (this may take some minutes on CPU)...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Validation F1 (macro) per label:")
    try:
        print(f1_score(y_val, y_pred, average=None))
    except Exception:
        pass
    print(classification_report(y_val, y_pred, target_names=['openness','conscientiousness','extraversion','agreeableness','neuroticism']))

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