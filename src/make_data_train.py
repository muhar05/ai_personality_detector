import pandas as pd
import random

# Data latih dengan variasi yang sangat beragam
openness_data = [
    # Label 1 (tinggi openness)
    "Saya suka mencoba hal-hal baru dan eksperimen",
    "Saya selalu tertarik dengan ide-ide kreatif dan inovatif",
    "Saya senang berimajinasi dan berpikir out of the box",
    "Saya mudah terbuka terhadap pengalaman baru",
    "Saya gemar mengeksplorasi konsep-konsep yang unik",
    "Saya menikmati diskusi tentang teori dan filosofi",
    "Saya sering mencari solusi alternatif untuk masalah",
    "Saya tertarik pada seni, musik, dan budaya yang beragam",
    "Saya selalu ingin belajar keterampilan baru",
    "Saya senang bereksperimen dengan pendekatan berbeda",
    "Saya mudah terinspirasi oleh hal-hal di sekitar",
    "Saya menikmati tantangan intelektual yang kompleks",
    "Saya suka berpikir abstrak dan konseptual",
    "Saya selalu terbuka pada perspektif dan pandangan baru",
    "Saya gemar mencoba teknologi dan metode terbaru",
    "Saya senang mengembangkan ide-ide original",
    "Saya mudah beradaptasi dengan perubahan",
    "Saya tertarik mengeksplorasi budaya dan tradisi berbeda",
    "Saya suka berkreasi dan membuat sesuatu yang unik",
    "Saya menikmati aktivitas yang menantang kreativitas",
    # Label 0 (rendah openness)  
    "Saya lebih suka rutinitas yang sudah pasti dan terprediksi",
    "Saya merasa nyaman dengan cara-cara yang sudah terbukti",
    "Saya tidak terlalu tertarik mencoba hal-hal yang belum pernah",
    "Saya lebih memilih metode tradisional yang sudah teruji",
    "Saya kurang suka dengan perubahan yang mendadak",
    "Saya lebih fokus pada hal-hal praktis daripada teoretis",
    "Saya tidak terlalu tertarik dengan seni dan budaya",
    "Saya lebih suka mengikuti prosedur yang sudah ada",
    "Saya merasa lebih aman dengan hal-hal yang familiar",
    "Saya tidak begitu suka bereksperimen atau mencoba-coba",
    "Saya lebih suka bekerja dengan cara yang sudah saya kuasai",
    "Saya tidak terlalu suka diskusi yang terlalu abstrak",
    "Saya lebih memilih aktivitas yang konkret dan jelas",
    "Saya kurang tertarik dengan ide-ide yang terlalu baru",
    "Saya lebih nyaman dengan lingkungan yang stabil",
    "Saya tidak terlalu suka mengambil risiko dengan hal baru",
    "Saya lebih suka mengikuti aturan yang sudah ditetapkan",
    "Saya kurang tertarik dengan filosofi atau teori kompleks",
    "Saya lebih fokus pada hasil yang nyata dan terukur",
    "Saya tidak terlalu suka dengan ketidakpastian"
]

conscientiousness_data = [
    # Label 1 (tinggi conscientiousness)
    "Saya selalu menyelesaikan pekerjaan tepat waktu",
    "Saya sangat terorganisir dalam mengatur jadwal",
    "Saya selalu mempersiapkan segala sesuatu dengan matang",
    "Saya disiplin dalam menjalankan komitmen",
    "Saya selalu memeriksa detail pekerjaan sebelum submit",
    "Saya memiliki sistem filing yang rapi dan teratur",
    "Saya selalu menepati janji dan deadline",
    "Saya bekerja dengan standar kualitas yang tinggi",
    "Saya membuat to-do list dan mengikutinya konsisten",
    "Saya selalu double-check hasil kerja sebelum diserahkan",
    "Saya memiliki rutinitas harian yang terstruktur",
    "Saya selalu mencatat dan mendokumentasikan dengan baik",
    "Saya mengerjakan tugas step by step secara sistematis",
    "Saya selalu backup data dan file penting",
    "Saya memiliki planning yang detail untuk setiap project",
    "Saya konsisten dalam menjaga kualitas output",
    "Saya selalu review dan evaluasi hasil kerja",
    "Saya membuat jadwal yang realistis dan mengikutinya",
    "Saya sangat hati-hati dalam mengambil keputusan",
    "Saya selalu bertanggung jawab atas hasil kerja",
    # Label 0 (rendah conscientiousness)
    "Saya sering menunda-nunda pekerjaan sampai menit terakhir",
    "Saya kurang terorganisir dalam mengatur jadwal",
    "Saya sering lupa dengan appointment atau deadline",
    "Saya kadang tidak teliti dalam mengerjakan tugas",
    "Saya sering kehilangan dokumen atau file penting",
    "Saya tidak terlalu suka membuat planning yang detail",
    "Saya sering terburu-buru dalam menyelesaikan pekerjaan",
    "Saya kadang tidak konsisten dalam kualitas output",
    "Saya tidak terlalu suka dengan rutinitas yang ketat",
    "Saya sering mengerjakan sesuatu secara spontan",
    "Saya kadang tidak backup data dengan baik",
    "Saya tidak terlalu detail dalam dokumentasi",
    "Saya sering multitasking tanpa fokus yang jelas",
    "Saya kadang tidak menepati jadwal yang sudah dibuat",
    "Saya sering mengubah rencana di tengah jalan",
    "Saya tidak terlalu suka dengan sistem yang terlalu rigid",
    "Saya kadang tidak thorough dalam review hasil",
    "Saya sering bekerja dengan deadline yang mepet",
    "Saya tidak terlalu suka dengan administrative tasks",
    "Saya kadang tidak konsisten dalam follow-up"
]

extraversion_data = [
    # Label 1 (tinggi extraversion)
    "Saya senang berinteraksi dan ngobrol dengan banyak orang",
    "Saya mudah memulai percakapan dengan orang baru",
    "Saya merasa energik saat berada di kerumunan",
    "Saya suka menjadi pusat perhatian dalam diskusi",
    "Saya aktif dalam kegiatan sosial dan komunitas",
    "Saya mudah berteman dan bergaul dengan siapa saja",
    "Saya senang berbagi cerita dan pengalaman",
    "Saya suka bekerja dalam tim dan kolaborasi",
    "Saya mudah mengekspresikan pendapat di depan umum",
    "Saya enjoy dengan networking dan bertemu orang baru",
    "Saya sering memimpin diskusi atau presentasi",
    "Saya mudah terbuka dan sharing dengan orang lain",
    "Saya senang dengan acara-acara yang ramai",
    "Saya aktif dalam media sosial dan komunikasi online",
    "Saya mudah mengajak orang lain untuk berpartisipasi",
    "Saya senang dengan brainstorming session dalam kelompok",
    "Saya mudah beradaptasi dalam lingkungan sosial baru",
    "Saya suka mengorganisir acara atau gathering",
    "Saya mudah memotivasi dan menginspirasi orang lain",
    "Saya merasa nyaman berbicara di forum publik",
    # Label 0 (rendah extraversion)
    "Saya lebih suka bekerja sendiri daripada dalam kelompok",
    "Saya butuh waktu untuk recharge setelah bersosialisasi",
    "Saya lebih nyaman mendengarkan daripada berbicara",
    "Saya tidak terlalu suka menjadi pusat perhatian",
    "Saya lebih memilih lingkungan yang tenang dan damai",
    "Saya perlu waktu untuk membuka diri pada orang baru",
    "Saya lebih suka komunikasi tertulis daripada verbal",
    "Saya merasa lelah setelah interaksi sosial yang intens",
    "Saya lebih memilih gathering kecil daripada acara besar",
    "Saya tidak terlalu aktif dalam media sosial",
    "Saya lebih suka observasi daripada langsung berpartisipasi",
    "Saya butuh me-time untuk menyendiri dan refleksi",
    "Saya tidak terlalu suka dengan small talk atau basa-basi",
    "Saya lebih memilih berkualitas daripada kuantitas dalam pertemanan",
    "Saya tidak terlalu comfortable dengan spontaneous social event",
    "Saya lebih suka mengekspresikan ide melalui tulisan",
    "Saya membutuhkan preparation sebelum presentasi atau meeting",
    "Saya lebih memilih bekerja di belakang layar",
    "Saya tidak terlalu suka dengan networking yang terlalu formal",
    "Saya merasa lebih produktif saat bekerja dalam suasana tenang"
]

agreeableness_data = [
    # Label 1 (tinggi agreeableness)
    "Saya selalu siap membantu teman yang kesulitan",
    "Saya mudah berempati dengan perasaan orang lain",
    "Saya suka berkolaborasi dan bekerja sama dalam tim",
    "Saya selalu berusaha menghindari konflik dengan orang lain",
    "Saya senang berbagi dan memberikan support pada orang lain",
    "Saya mudah memaafkan kesalahan orang lain",
    "Saya selalu mencari solusi win-win dalam masalah",
    "Saya peduli dengan kesejahteraan dan perasaan orang lain",
    "Saya suka mendengarkan curhat dan memberikan advice",
    "Saya selalu berusaha fair dan adil dalam berinteraksi",
    "Saya mudah trusty dan percaya pada niat baik orang",
    "Saya suka memberikan compliment dan appreciation",
    "Saya selalu considerate terhadap kebutuhan orang lain",
    "Saya senang volunteer dan membantu kegiatan sosial",
    "Saya mudah compromise dan mencari jalan tengah",
    "Saya selalu supportive terhadap ide dan usaha orang lain",
    "Saya suka memediasi konflik dan mencari solusi damai",
    "Saya peduli dengan community dan lingkungan sekitar",
    "Saya mudah melakukan teamwork dan koordinasi",
    "Saya selalu respectful dan menghargai pendapat orang lain",
    # Label 0 (rendah agreeableness)
    "Saya lebih fokus pada kepentingan diri sendiri",
    "Saya tidak terlalu mudah percaya pada orang baru",
    "Saya kadang skeptis terhadap motif orang lain",
    "Saya lebih suka berkompetisi daripada berkolaborasi",
    "Saya tidak terlalu patient dengan kesalahan orang lain",
    "Saya lebih direct dan blunt dalam berkomunikasi",
    "Saya tidak terlalu suka mengalah dalam diskusi",
    "Saya lebih memilih bekerja independent",
    "Saya tidak terlalu mudah influenced oleh opinion orang lain",
    "Saya lebih objective daripada emotional dalam decision making",
    "Saya tidak terlalu suka dengan social obligation",
    "Saya lebih pragmatic dalam approach problem solving",
    "Saya tidak terlalu concern dengan people pleasing",
    "Saya lebih straightforward dalam memberikan feedback",
    "Saya tidak terlalu mudah sympathize dengan drama orang lain",
    "Saya lebih focus pada result daripada process relationship",
    "Saya tidak terlalu suka dengan consensus building yang lama",
    "Saya lebih assertive dalam memperjuangkan pendapat",
    "Saya tidak terlalu emotional dalam respond conflict",
    "Saya lebih realistis daripada idealis dalam expectation"
]

neuroticism_data = [
    # Label 1 (tinggi neuroticism)
    "Saya sering merasa khawatir tentang masa depan",
    "Saya mudah stress saat menghadapi deadline",
    "Saya kadang overthinking tentang hal-hal kecil",
    "Saya sensitive terhadap kritik dan feedback negatif",
    "Saya sering merasa anxious dalam situasi baru",
    "Saya mudah mood swing dan emotional",
    "Saya kadang insecure tentang kemampuan diri",
    "Saya sering merasa overwhelmed dengan tanggung jawab",
    "Saya mudah panik saat ada masalah mendadak",
    "Saya kadang pessimistic tentang outcome suatu hal",
    "Saya sensitive terhadap perubahan lingkungan",
    "Saya sering merasa nervous sebelum presentasi",
    "Saya mudah frustrated saat sesuatu tidak sesuai rencana",
    "Saya kadang doubt terhadap keputusan yang sudah dibuat",
    "Saya sering worrying tentang pendapat orang lain",
    "Saya mudah terpengaruh oleh negative vibes",
    "Saya kadang merasa tidak confident dengan performance",
    "Saya sering restless dan sulit relax",
    "Saya mudah triggered oleh situasi yang unpredictable",
    "Saya kadang merasa helpless saat menghadapi challenge",
    # Label 0 (rendah neuroticism)
    "Saya tetap tenang dalam situasi yang menekan",
    "Saya mudah bounce back dari kegagalan",
    "Saya tidak terlalu worry tentang hal-hal di luar kontrol",
    "Saya stable secara emosional dalam berbagai situasi",
    "Saya confident dengan kemampuan untuk handle masalah",
    "Saya tidak mudah stressed oleh deadline atau pressure",
    "Saya relaxed dan easy-going dalam berinteraksi",
    "Saya optimistic tentang outcome dan hasil",
    "Saya tidak terlalu sensitive terhadap kritik",
    "Saya calm dan composed saat menghadapi konflik",
    "Saya tidak mudah anxious dalam situasi baru",
    "Saya resilient dan adaptable terhadap perubahan",
    "Saya tidak overthinking tentang decision making",
    "Saya secure dan confident dengan diri sendiri",
    "Saya tidak mudah moody atau emotional",
    "Saya peaceful dan content dengan current situation",
    "Saya tidak mudah frustrated dengan obstacle",
    "Saya steady dan consistent dalam performance",
    "Saya tidak terlalu concern dengan judgment orang lain",
    "Saya balanced dalam manage expectation dan reality"
]

# Buat dataset seimbang
data = []
id_counter = 1

# Openness
for i, text in enumerate(openness_data):
    label = 1 if i < 20 else 0
    data.append([id_counter, f"User_O_{i+1}", text, label, 0, 0, 0, 0])
    id_counter += 1

# Conscientiousness
for i, text in enumerate(conscientiousness_data):
    label = 1 if i < 20 else 0
    data.append([id_counter, f"User_C_{i+1}", text, 0, label, 0, 0, 0])
    id_counter += 1

# Extraversion
for i, text in enumerate(extraversion_data):
    label = 1 if i < 20 else 0
    data.append([id_counter, f"User_E_{i+1}", text, 0, 0, label, 0, 0])
    id_counter += 1

# Agreeableness
for i, text in enumerate(agreeableness_data):
    label = 1 if i < 20 else 0
    data.append([id_counter, f"User_A_{i+1}", text, 0, 0, 0, label, 0])
    id_counter += 1

# Neuroticism
for i, text in enumerate(neuroticism_data):
    label = 1 if i < 20 else 0
    data.append([id_counter, f"User_N_{i+1}", text, 0, 0, 0, 0, label])
    id_counter += 1

# Shuffle data untuk randomize
random.shuffle(data)

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "id", "name", "text", "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"
])

# Save to CSV
df.to_csv("../data/train.csv", index=False, encoding="utf-8")
print(f"train.csv berhasil dibuat dengan {len(data)} baris data bervariasi dan seimbang.")
print(f"Setiap trait memiliki 20 label 1 dan 20 label 0.")