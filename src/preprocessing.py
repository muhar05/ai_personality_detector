import os
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

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