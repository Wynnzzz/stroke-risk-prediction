# Laporan Proyek Machine Learning - Berwyn Izzut Taghyir

## Domain Proyek

Stroke merupakan salah satu penyakit kardiovaskular yang paling mematikan dan penyebab utama kecacatan jangka panjang di dunia. Menurut laporan dari World Health Organization (2021), sekitar 15 juta orang mengalami stroke setiap tahunnya, dan hampir 5 juta di antaranya meninggal dunia, sementara 5 juta lainnya hidup dengan disabilitas permanen. Stroke dapat terjadi secara tiba-tiba, dan sering kali didahului oleh gejala-gejala awal yang diabaikan. Oleh karena itu, deteksi dini terhadap risiko stroke sangat penting dalam upaya pencegahan dan mitigasi dampak fatal yang ditimbulkannya.

Permasalahan dalam deteksi risiko stroke secara tradisional terletak pada ketergantungan terhadap penilaian klinis manual yang subjektif dan tidak selalu tepat waktu, terutama di wilayah dengan keterbatasan tenaga medis. Dalam konteks ini, pendekatan Predictive Analytics dengan memanfaatkan Machine Learning (ML) dapat menjadi solusi strategis untuk memprediksi kemungkinan seseorang mengalami stroke berdasarkan data gejala yang dimiliki.

Dataset Stroke Risk Prediction Based on Symptoms telah dikembangkan untuk mendukung penelitian ini. Dataset ini berisi berbagai gejala klinis yang secara medis telah dikaitkan dengan peningkatan risiko stroke, seperti nyeri dada, tekanan darah tinggi, detak jantung tidak teratur, hingga gangguan tidur. Selain itu, dataset ini dirancang secara seimbang (balanced), dengan 50% data mewakili individu berisiko dan 50% tidak berisiko, yang meningkatkan validitas dalam pelatihan model klasifikasi biner. Informasi dalam dataset disusun berdasarkan sumber medis terpercaya seperti:

- Harrison’s Principles of Internal Medicine (20th ed.)

- The Stroke Book (2nd ed., Cambridge Medicine)

- Laporan resmi dari American Stroke Association (ASA), Mayo Clinic, dan World Health Organization (WHO)

Dalam penelitian ini, model pembelajaran mesin seperti Logistic Regression dan Random Forest akan digunakan untuk membangun sistem prediksi biner mengenai apakah seseorang berisiko mengalami stroke. Untuk meningkatkan akurasi dan efisiensi, dilakukan pula hyperparameter tuning serta feature selection menggunakan teknik SelectKBest. Model akan dievaluasi menggunakan metrik seperti akurasi, presisi, recall, dan F1-score. Selain itu, model yang lebih kompleks seperti XGBoost akan digunakan sebagai pembanding untuk mengetahui apakah pendekatan kompleks dapat memberikan performa yang lebih unggul.

Masalah prediksi risiko stroke perlu diselesaikan karena dapat menjadi alat bantu yang signifikan dalam mendukung pengambilan keputusan medis secara cepat dan akurat. Implementasi sistem ini di fasilitas kesehatan, terutama di daerah dengan akses terbatas terhadap spesialis neurologi, dapat membantu dalam penyaringan awal pasien berisiko tinggi serta mempercepat intervensi medis yang krusial.

Dengan memanfaatkan kekuatan machine learning, penelitian ini diharapkan tidak hanya menghasilkan model prediktif yang akurat, tetapi juga dapat memberikan wawasan penting mengenai gejala mana yang paling berkontribusi terhadap risiko stroke, serta mendukung pengembangan sistem Explainable AI (XAI) dalam dunia medis.

**Referensi**

- Jameson, J. L., Fauci, A. S., Kasper, D. L., Hauser, S. L., Longo, D. L., & Loscalzo, J. (2018). *Harrison's Principles of Internal Medicine* (20th ed.). McGraw-Hill Education.

- Rashid, M., Shah, A. A., & Abbas, S. (2021). Stroke prediction using machine learning algorithms: A comparative study. *Biomedical Journal of Scientific & Technical Research*, 34(1), 26750–26757. https://doi.org/10.26717/BJSTR.2021.34.005537

- Sarwar, S., Dent, A., & Faust, K. (2022). Comparative analysis of machine learning models for cardiovascular risk prediction. *Journal of Biomedical Informatics*, 128, 104036. https://doi.org/10.1016/j.jbi.2022.104036

- World Health Organization. (2020). *Digital Health Guidelines: Recommendations on Digital Interventions for Health System Strengthening*. https://www.who.int/publications/i/item/9789241550505

- World Health Organization. (2021). *Global Health Estimates 2021: Deaths by Cause, Age, Sex, by Country and by Region, 2000–2019*. https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates

## Business Understanding

Stroke merupakan salah satu penyebab utama kematian dan kecacatan di seluruh dunia. Deteksi dini terhadap risiko stroke memiliki peran penting dalam mencegah komplikasi serius dan menyelamatkan nyawa pasien. Dalam konteks ini, dibutuhkan pendekatan prediktif berbasis data untuk mengidentifikasi individu yang berisiko mengalami stroke berdasarkan gejala yang mereka alami.

### Problem Statements

1. Bagaimana cara mengidentifikasi individu yang berisiko mengalami stroke berdasarkan gejala dan faktor demografis?
2. Apakah model machine learning dapat digunakan untuk memprediksi risiko stroke secara akurat?
3. Bagaimana cara mengoptimalkan model prediksi agar akurasi dan reliabilitasnya meningkat untuk digunakan dalam dunia medis?

### Goals

1. Mengembangkan model klasifikasi untuk mendeteksi risiko stroke berdasarkan fitur gejala dan usia pada setiap individu.
2. Menguji efektivitas model machine learning seperti Logistic Regression, Random Forest, dan XGBoost dalam prediksi risiko stroke.
3. Melakukan optimasi model dengan teknik hyperparameter tuning dan pemilihan fitur agar menghasilkan model yang lebih akurat, efisien, dan dapat diterapkan secara nyata di bidang kesehatan.

### Solution Statements

1. Menggunakan beberapa algoritma machine learning seperti Logistic Regression sebagai baseline model yang sederhana, Random Forest sebagai model ensambel yang mampu menangani kompleksitas data, serta XGBoost sebagai model yang lebih kompleks dengan kemampuan generalisasi tinggi.
2. Menerapkan teknik hyperparameter tuning seperti RandomizedSearchCV atau GridSearchCV untuk menemukan kombinasi parameter terbaik, serta mengevaluasi performa model menggunakan metrik seperti accuracy, precision, recall, dan F1-score untuk memastikan hasil yang optimal.
3. Melakukan feature selection dengan menggunakan metode seperti SelectKBest atau analisis feature importance untuk menyederhanakan model, meningkatkan interpretabilitas, dan menjaga performa prediksi agar tetap tinggi.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah *Stroke Risk Prediction Dataset Based on Symptoms* yang tersedia secara publik melalui platform Kaggle. Dataset ini dirancang untuk mendukung pengembangan model machine learning dan deep learning dalam bidang prediksi risiko stroke, baik untuk klasifikasi biner (berisiko/tidak) maupun regresi (persentase risiko stroke).

Dataset dapat diunduh melalui tautan berikut:  
[Stroke Risk Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/thedevastator/stroke-risk-prediction-based-on-symptoms)

Dataset ini terdiri dari fitur-fitur gejala, satu fitur demografis (usia), serta dua target yang berbeda: klasifikasi risiko stroke secara biner dan estimasi risiko stroke dalam bentuk persentase. Seluruh data telah dirancang untuk mencerminkan validitas medis berdasarkan literatur dan referensi dari organisasi kesehatan seperti WHO dan American Stroke Association. Dataset ini terdiri dari 70000 data dan juga bersifat seimbang, yaitu 50% data mewakili individu dengan risiko stroke dan 50% lainnya tidak.

### Variabel-variabel pada dataset adalah sebagai berikut:

#### Fitur Gejala Klinis (Binary: 0 = Tidak, 1 = Ya)
- **Chest Pain** : Nyeri di dada.
- **Shortness of Breath** : Sesak napas.
- **Irregular Heartbeat** : Detak jantung tidak teratur.
- **Fatigue & Weakness** : Kelelahan dan kelemahan.
- **Dizziness** : Pusing atau kehilangan keseimbangan.
- **Swelling (Edema)** : Pembengkakan pada bagian tubuh.
- **Pain in Neck/Jaw/Shoulder/Back** : Nyeri pada leher, rahang, bahu, atau punggung.
- **Excessive Sweating** : Keringat berlebihan.
- **Persistent Cough** : Batuk terus-menerus.
- **Nausea/Vomiting** : Mual atau muntah.
- **High Blood Pressure** : Tekanan darah tinggi.
- **Chest Discomfort (Activity)** : Rasa tidak nyaman di dada saat beraktivitas.
- **Cold Hands/Feet** : Tangan atau kaki terasa dingin.
- **Snoring/Sleep Apnea** : Mendengkur atau gangguan pernapasan saat tidur.
- **Anxiety/Feeling of Doom** : Kecemasan berlebihan atau perasaan akan terjadi sesuatu yang buruk.

#### Fitur Demografi
- **Age** : Usia individu (numerik), salah satu faktor risiko utama untuk stroke.

#### Fitur Target
- **At Risk (Binary)** : Kelas target biner (1 = berisiko stroke, 0 = tidak berisiko).
- **Stroke Risk (%)** : Perkiraan kemungkinan stroke dalam bentuk persentase (0-100).

### Eksplorasi Awal dan Visualisasi Data

Untuk memahami distribusi dan hubungan antar fitur, dilakukan beberapa teknik exploratory data analysis (EDA), antara lain:

1. **Deskripsi Statistik Awal**  
   Menggunakan `df.describe()` untuk memahami nilai rata-rata, standar deviasi, nilai minimum dan maksimum dari masing-masing fitur.

2. **Pengecekan Nilai Kosong dan Duplikat**  
   - Nilai kosong diperiksa dengan `df.isnull().sum()` dan tidak ditemukan missing values.
   - Ditemukan sebanyak 1021 data duplikat menggunakan `df.duplicated().sum()` yang kemudian dihapus.

3. **Visualisasi Outlier dan Distribusi**  
   - Visualisasi **boxplot** untuk mendeteksi outlier pada setiap fitur numerik.
   - Visualisasi **histogram + KDE** untuk melihat bentuk distribusi masing-masing fitur.

4. **Analisis Korelasi Antar Fitur**  
   - Korelasi antara fitur dihitung dan divisualisasikan dengan **heatmap**, membantu dalam mengidentifikasi keterkaitan antar gejala dan usia dengan risiko stroke.

Tahapan EDA ini penting untuk memahami struktur dan karakteristik data yang digunakan, serta untuk menentukan preprocessing dan pemilihan algoritma yang tepat pada tahap selanjutnya.

     
## Data Preparation

Tahapan data preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pemodelan bersih, relevan, dan sesuai untuk diterapkan pada algoritma machine learning. Berikut adalah urutan tahapan yang dilakukan dalam proses data preparation:

1. **Menghapus Data Duplikat**  
   - Ditemukan sebanyak 1.021 data duplikat menggunakan fungsi `df.duplicated().sum()`.
   - Duplikat ini dihapus dengan `df.drop_duplicates(inplace=True)` untuk menghindari bias pada model yang disebabkan oleh pengulangan data yang sama.

2. **Pengecekan dan Penanganan Missing Values**  
   - Dilakukan pengecekan missing values dengan `df.isnull().sum()`.
   - Hasilnya, tidak ditemukan nilai kosong pada dataset, sehingga tidak perlu dilakukan imputasi.

3. **Eksplorasi Outlier dan Distribusi Data**  
   - Visualisasi outlier dilakukan dengan boxplot untuk setiap fitur menggunakan `sns.boxplot`.
   - Distribusi data dianalisis menggunakan histogram dengan `sns.histplot`, dilengkapi dengan `kde=True` untuk melihat pola sebaran data.
   - Tahapan ini membantu mengenali fitur yang memiliki distribusi tidak normal atau outlier ekstrem yang mungkin memengaruhi performa model.

4. **Feature Selection (Pemilihan Fitur)**  
   - Fitur target `Stroke Risk (%)` dihapus karena fokus prediksi hanya pada klasifikasi biner (`At Risk (Binary)`).
   - Fitur input yang digunakan terdiri dari 15 gejala dan 1 fitur usia (total 16 fitur).

5. **Normalisasi Fitur Numerik**  
   - Karena data mengandung fitur numerik dengan skala berbeda-beda (misalnya, `Age` bernilai puluhan, sedangkan fitur lain berupa biner), dilakukan normalisasi menggunakan `StandardScaler()` dari Scikit-learn.
   - Ini penting untuk algoritma seperti Logistic Regression dan SVM yang sensitif terhadap skala fitur.

6. **Pemisahan Data Train dan Test**  
   - Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan `train_test_split()` dengan `random_state=42` untuk reprodusibilitas.
   - Ini dilakukan agar performa model bisa diukur secara objektif terhadap data yang tidak dilihat sebelumnya.

Proses data preparation ini bertujuan untuk memastikan bahwa data yang masuk ke dalam model machine learning dalam kondisi optimal dan siap untuk dilakukan pelatihan. Dengan menghilangkan duplikasi, menyamakan skala fitur, dan membagi data secara adil, maka performa dan generalisasi model dapat meningkat secara signifikan.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
