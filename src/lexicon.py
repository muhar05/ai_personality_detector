from collections import Counter
from preprocessing import preprocess_text, stemmer

RAW_TRAITS = {
    "openness": ["imajinasi", "ide", "baru", "pikir", "kreatif", "open minded", "kepo", "penasaran"],
    "conscientiousness": ["disiplin", "tepat", "rapi", "teratur", "kerja", "on time", "niat", "serius", "telaten"],
    "extraversion": ["teman", "bicara", "senang", "ramai", "ngobrol", "koneksi", "nongkrong", "mabar", "gabut", "hangout"],
    "agreeableness": ["bantu", "baik", "peduli", "teman", "sopan", "ramah", "support", "care", "solid"],
    "neuroticism": ["cemas", "khawatir", "takut", "gelisah", "panik", "galau", "baper", "overthinking"]
}

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
    mx = max(scores.values()) if max(scores.values()) > 0 else 1
    norm = {k: v / mx for k, v in scores.items()}
    return scores, norm