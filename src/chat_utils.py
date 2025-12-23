import re

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

def read_import_file(import_path):
    data = {}
    with open(import_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                name, text = line.split(':', 1)
                name = name.strip()
                text = text.strip()
                if name not in data:
                    data[name] = []
                data[name].append(text)
    return data  # {nama: [list teks]}

def parse_chat_per_user(chat_path):
    user_msgs = {}
    with open(chat_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r"\[\d{2}/\d{2}/\d{2}, \d{2}\.\d{2}\.\d{2}\] ([^:]+): (.+)", line.strip())
            if m:
                name, msg = m.group(1).strip(), m.group(2).strip()
                if name not in user_msgs:
                    user_msgs[name] = []
                user_msgs[name].append(msg)
    return user_msgs  # {name: [msg, ...]}