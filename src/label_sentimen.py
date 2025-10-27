from transformers import pipeline
import os

# Inisialisasi pipeline sentiment (gunakan model multilingual)
sentiment_pipeline = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")

def get_sentiment_label(text):
    result = sentiment_pipeline(text[:512])[0]
    label = result['label'].lower()
    if label == "positive":
        return "+"
    elif label == "negative":
        return "-"
    else:
        return "="

def label_chat_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    labeled_lines = []
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) >= 3:
            message = parts[2].strip()
            label = get_sentiment_label(message)
            labeled_line = line.strip() + f" [{label}]\n"
            labeled_lines.append(labeled_line)
        else:
            labeled_lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(labeled_lines)

if __name__ == "__main__":
    # Ambil path folder project (parent dari src)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_file = os.path.join(BASE_DIR, 'data', 'chat_sample.txt')
    output_file = os.path.join(BASE_DIR, 'data', 'chat_sample_labeled.txt')
    label_chat_file(input_file, output_file)
    print(f"Labeling selesai. Hasil disimpan di {output_file}")