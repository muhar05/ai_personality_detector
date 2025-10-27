import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Ensure NLTK stopwords available (download if missing)
try:
    _ = nltk.corpus.stopwords.words('indonesian')
except Exception:
    nltk.download('stopwords')

STOPWORDS = set(nltk.corpus.stopwords.words('indonesian'))

SLANG = {
    'gk': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'ga': 'tidak',
    'gw': 'saya', 'gue': 'saya', 'lo': 'kamu', 'kmu': 'kamu',
    'sm': 'sama', 'sama2': 'sama', 'tdk': 'tidak'
}
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