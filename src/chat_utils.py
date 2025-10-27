def read_chat_file(chat_path):
    with open(chat_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    messages = []
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) >= 3:
            messages.append(parts[2].strip())
    text = ' '.join(messages)
    # Ambil nama pengirim dari baris pertama
    if lines:
        first_line = lines[0]
        name = first_line.split(']', 1)[-1].split(':', 1)[0].strip()
    else:
        name = "Unknown"
    return name, messages, text