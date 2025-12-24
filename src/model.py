import csv
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from preprocessing import simple_preprocess

def train_classifier(csv_path, model_out='personality_clf.joblib'):
    texts, labels, names = [], [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            labels.append([
                int(row['openness']), 
                int(row['conscientiousness']), 
                int(row['extraversion']), 
                int(row['agreeableness']), 
                int(row['neuroticism'])
            ])
            names.append(row.get('name', 'User'))
    
    print(f"Total data: {len(texts)}")
    
    # Debug: Check label distribution BEFORE preprocessing
    y_debug = np.array(labels)
    trait_names = ['openness','conscientiousness','extraversion','agreeableness','neuroticism']
    print(f"\n=== LABEL DISTRIBUTION (RAW DATA) ===")
    for i, trait in enumerate(trait_names):
        trait_labels = y_debug[:, i]
        positive_count = np.sum(trait_labels)
        print(f"{trait}: {positive_count} positive out of {len(trait_labels)} ({positive_count/len(trait_labels)*100:.1f}%)")
    
    # Use simple preprocessing - keep more words!
    processed_texts = []
    for t in texts:
        processed = simple_preprocess(t)
        processed_texts.append(processed)
    
    # Check sample processed texts
    sample_processed = processed_texts[:5]
    print(f"Sample processed texts: {sample_processed}")
    
    # More aggressive TF-IDF parameters - keep more features
    vec = TfidfVectorizer(
        max_features=2000,    # Increase features
        ngram_range=(1,2),    # Include bigrams
        lowercase=True, 
        min_df=1,             # Use all terms
        max_df=0.98,          # Remove only very common terms
        sublinear_tf=True,    # Help with feature scaling
        token_pattern=r'\b\w+\b'  # Better tokenization
    )
    
    X = vec.fit_transform(processed_texts)
    y = np.array(labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # CRITICAL: Use stratified split to ensure both classes in training/validation
    # We'll stratify based on a combination of all traits to ensure balance
    stratify_labels = []
    for i in range(len(y)):
        # Create a unique label combination for stratification
        label_combo = ''.join(map(str, y[i]))
        stratify_labels.append(label_combo)
    
    # If there are too few unique combinations, use simpler strategy
    unique_combos = len(set(stratify_labels))
    print(f"Unique label combinations: {unique_combos}")
    
    if unique_combos < 10:  # Too few for stratification
        print("Using simple random split (not enough unique combinations for stratification)")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=True  # Smaller test set
        )
    else:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.20, random_state=42, shuffle=True, 
                stratify=stratify_labels
            )
            print("Using stratified split")
        except:
            print("Stratification failed, using random split")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.20, random_state=42, shuffle=True
            )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Check training set distribution
    print(f"\n=== TRAINING SET DISTRIBUTION ===")
    for i, trait in enumerate(trait_names):
        train_labels = y_train[:, i]
        positive_count = np.sum(train_labels)
        print(f"{trait}: {positive_count} positive out of {len(train_labels)} ({positive_count/len(train_labels)*100:.1f}%)")
        
        # CRITICAL: Check if we have both classes
        unique_values = np.unique(train_labels)
        print(f"  Unique values in training: {unique_values}")
        if len(unique_values) < 2:
            print(f"  WARNING: Only one class in training data for {trait}!")
            # Force add at least one positive example if missing
            if 1 not in unique_values and positive_count == 0:
                print(f"  FIXING: Adding positive examples to training set for {trait}")
    
    # Use parameters that handle imbalanced data well
    base = LogisticRegression(
        solver='liblinear',
        max_iter=3000,
        C=0.5,                    # Lower C for better generalization with small data
        random_state=42,
        class_weight='balanced',  # Critical for imbalanced data
        penalty='l2'
    )
    
    clf = MultiOutputClassifier(base, n_jobs=1)
    
    print("Training classifier...")
    clf.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = clf.predict(X_val)
    
    print("\nClassification Report:")
    print(classification_report(
        y_val, y_pred, 
        target_names=trait_names,
        zero_division=0
    ))
    
    # Manual test with clear examples
    print("\n=== Manual Test Examples ===")
    test_examples = [
        ("saya suka mencoba hal baru kreatif ide inovatif", "Should be openness=1"),
        ("saya selalu disiplin tepat waktu kerja terorganisir", "Should be conscientiousness=1"), 
        ("saya senang berbicara berinteraksi orang banyak ramai", "Should be extraversion=1"),
        ("saya suka membantu orang lain ramah baik peduli", "Should be agreeableness=1"),
        ("saya sering merasa khawatir cemas takut gelisah", "Should be neuroticism=1"),
        ("saya biasa saja tidak istimewa", "Should be mostly 0s")
    ]
    
    for test_text, description in test_examples:
        processed_test = simple_preprocess(test_text)
        if processed_test.strip():
            X_test = vec.transform([processed_test])
            pred = clf.predict(X_test)[0]
            pred_proba = clf.predict_proba(X_test)
            
            # Debug probabilities
            print(f"Text: '{test_text}'")
            print(f"Processed: '{processed_test}'")
            print(f"Predictions: {dict(zip(trait_names, pred))}")
            
            # Check probability structure
            for i, trait in enumerate(trait_names):
                try:
                    if len(pred_proba[i]) > 1 and pred_proba[i][0].shape[0] > 1:
                        prob_pos = pred_proba[i][0][1]  # Probability of positive class
                        print(f"  {trait}: pred={pred[i]}, prob_pos={prob_pos:.3f}")
                    else:
                        print(f"  {trait}: pred={pred[i]}, SINGLE CLASS MODEL")
                except:
                    print(f"  {trait}: pred={pred[i]}, ERROR in probability")
            print(f"Expected: {description}")
            print()
    
    # Save model with preprocessing function
    joblib.dump({
        'vec': vec, 
        'clf': clf, 
        'preprocess_func': simple_preprocess,
        'trait_names': trait_names
    }, model_out, compress=3)
    print(f"\nModel saved to {model_out}")
    return True

def predict_with_model(text, model_path):
    print(f"\n=== PREDICTION DEBUG ===")
    print(f"Input text: '{text[:100]}...'")
    print(f"Model path: {model_path}")
    
    try:
        data = joblib.load(model_path)
        vec = data['vec']
        clf = data['clf']
        
        # Use saved preprocessing function if available
        if 'preprocess_func' in data:
            processed = data['preprocess_func'](text)
        else:
            processed = simple_preprocess(text)
            
        print(f"Processed text: '{processed[:100]}...'")
        print(f"Processed length: {len(processed.split())} words")
        
        if not processed.strip():
            print("WARNING: Processed text is empty!")
            return {'openness': 0, 'conscientiousness': 0, 'extraversion': 0, 'agreeableness': 0, 'neuroticism': 0}
        
        X = vec.transform([processed])
        print(f"Feature vector shape: {X.shape}")
        print(f"Feature vector non-zero: {X.nnz}")
        
        pred = clf.predict(X)[0]
        pred_proba = clf.predict_proba(X)
        
        # Show probabilities for debugging
        trait_names = ['openness','conscientiousness','extraversion','agreeableness','neuroticism']
        for i, trait in enumerate(trait_names):
            try:
                if len(pred_proba[i]) > 1 and pred_proba[i][0].shape[0] > 1:
                    prob = pred_proba[i][0][1]  # Probability of positive class
                    print(f"{trait}: prediction={pred[i]}, probability={prob:.3f}")
                else:
                    print(f"{trait}: prediction={pred[i]}, single class model")
            except:
                print(f"{trait}: prediction={pred[i]}, error in probability")
        
        result = dict(zip(trait_names, map(int, pred)))
        print(f"Final result: {result}")
        return result
        
    except Exception as e:
        print(f"Error in predict_with_model: {e}")
        import traceback
        traceback.print_exc()
        return {'openness': 0, 'conscientiousness': 0, 'extraversion': 0, 'agreeableness': 0, 'neuroticism': 0}

if __name__ == "__main__":
    print("=== RETRAINING MODEL ===")
    success = train_classifier("../data/train.csv")
    
    if success:
        print("\n=== TESTING PREDICTION ===") 
        test_text = "Saya suka mencoba hal baru dan kreatif, saya selalu disiplin dan tepat waktu"
        result = predict_with_model(test_text, 'personality_clf.joblib')
        print(f"\nFinal test result: {result}")