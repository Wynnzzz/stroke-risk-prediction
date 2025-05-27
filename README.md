# Laporan Proyek Machine Learning - Berwyn Izzut Taghyir

## Domain Proyek

Stroke merupakan salah satu penyakit kardiovaskular yang paling mematikan dan penyebab utama kecacatan jangka panjang di dunia. Menurut laporan dari World Health Organization (2021), sekitar 15 juta orang mengalami stroke setiap tahunnya, dan hampir 5 juta di antaranya meninggal dunia, sementara 5 juta lainnya hidup dengan disabilitas permanen. Stroke dapat terjadi secara tiba-tiba, dan sering kali didahului oleh gejala-gejala awal yang diabaikan. Oleh karena itu, deteksi dini terhadap risiko stroke sangat penting dalam upaya pencegahan dan mitigasi dampak fatal yang ditimbulkannya.

Permasalahan dalam deteksi risiko stroke secara tradisional terletak pada ketergantungan terhadap penilaian klinis manual yang subjektif dan tidak selalu tepat waktu, terutama di wilayah dengan keterbatasan tenaga medis. Dalam konteks ini, pendekatan Predictive Analytics dengan memanfaatkan Machine Learning (ML) dapat menjadi solusi strategis untuk memprediksi kemungkinan seseorang mengalami stroke berdasarkan data gejala yang dimiliki.

Dataset Stroke Risk Prediction Based on Symptoms telah dikembangkan untuk mendukung penelitian ini. Dataset ini berisi berbagai gejala klinis yang secara medis telah dikaitkan dengan peningkatan risiko stroke, seperti nyeri dada, tekanan darah tinggi, detak jantung tidak teratur, hingga gangguan tidur. Selain itu, dataset ini dirancang secara seimbang (balanced), dengan 50% data mewakili individu berisiko dan 50% tidak berisiko, yang meningkatkan validitas dalam pelatihan model klasifikasi biner. Informasi dalam dataset disusun berdasarkan sumber medis terpercaya seperti:

- Harrison’s Principles of Internal Medicine (20th ed.)

- The Stroke Book (2nd ed., Cambridge Medicine)

- Laporan resmi dari American Stroke Association (ASA), Mayo Clinic, dan World Health Organization (WHO)

Dalam penelitian ini, model pembelajaran mesin seperti Logistic Regression dan Random Forest akan digunakan untuk membangun sistem prediksi biner mengenai apakah seseorang berisiko mengalami stroke. Untuk meningkatkan akurasi dan efisiensi, dilakukan pula hyperparameter tuning untuk setiap model. Model akan dievaluasi menggunakan metrik seperti akurasi, presisi, recall, dan F1-score. Selain itu, model yang lebih kompleks seperti XGBoost akan digunakan sebagai pembanding untuk mengetahui apakah pendekatan kompleks dapat memberikan performa yang lebih unggul.

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
   
   Deskripsi statistik awal dilakukan dengan menggunakan fungsi df.describe(), yang memberikan informasi seperti nilai rata-rata, standar deviasi, nilai minimum, maksimum, serta kuartil dari setiap fitur. Langkah ini berguna untuk mendapatkan gambaran umum mengenai skala nilai dan variasi data pada masing-masing fitur, khususnya untuk fitur numerik seperti usia dan persentase risiko stroke.

2. **Pengecekan Nilai Kosong dan Duplikat**
   
   Pemeriksaan nilai kosong dilakukan menggunakan `df.isnull().sum()`, dan hasilnya menunjukkan bahwa tidak terdapat missing values pada dataset. Selain itu, ditemukan sebanyak 1.021 data duplikat dengan menggunakan `df.duplicated().sum()`.

3. **Visualisasi Outlier dan Distribusi**
   
   Visualisasi outlier dilakukan dengan menggunakan boxplot dari `seaborn` (`sns.boxplot`) pada masing-masing fitur. Karena sebagian besar fitur merupakan variabel biner, boxplot hanya menampilkan satu kotak horizontal penuh, yang menunjukkan bahwa nilainya berkisar pada 0 dan 1. Outlier hanya ditemukan pada fitur `Stroke Risk (%)`, namun fitur ini tidak digunakan dalam klasifikasi, sehingga outlier dapat diabaikan. Distribusi data dianalisis lebih lanjut menggunakan histogram (`sns.histplot`) dengan `kde=True`, yang menunjukkan bahwa variabel gejala sangat terpolarisasi (bernilai 0 atau 1), sementara distribusi usia lebih menyebar dan risiko stroke (%) menunjukkan pola mendekati distribusi normal.
   
      ![alt text](https://github.com/Wynnzzz/stroke-risk-prediction/blob/main/img/boxplot.png?raw=true)

   Distribusi data dianalisis menggunakan histogram dengan `sns.histplot`, dilengkapi dengan `kde=True` untuk melihat pola sebaran data.
   
     ![alt text](https://github.com/Wynnzzz/stroke-risk-prediction/blob/main/img/distribution.png?raw=true)
   
     Sebagian besar variabel gejala seperti nyeri dada, sesak napas, detak jantung tidak teratur, kelelahan, pusing, edema, dan lain-lain memiliki distribusi yang sangat terpolarisasi dengan dua puncak di ujung nilai 0 dan 1. Ini mengindikasikan bahwa data gejala tersebut bersifat biner atau sangat jarang berada dalam nilai antara, artinya responden biasanya hanya melaporkan ada atau tidaknya gejala tersebut secara tegas.
     Distribusi usia berbeda dengan variabel gejala, menunjukkan variasi yang lebih merata dengan beberapa fluktuasi, mencerminkan distribusi populasi yang beragam mulai dari usia muda hingga lansia. Sementara itu, distribusi risiko stroke (%) menunjukkan pola mendekati distribusi normal dengan puncak di tengah, yang berarti sebagian besar individu memiliki risiko stroke pada kisaran menengah, dengan lebih sedikit yang memiliki risiko sangat rendah atau sangat tinggi.
     Terakhir, distribusi status risiko biner (At Risk) juga menunjukkan pola yang didominasi oleh nilai 0 dan 1, yang menegaskan bahwa data ini juga bersifat kategorikal, memisahkan individu ke dalam dua kelompok risiko yang jelas: berisiko dan tidak berisiko. Secara keseluruhan, grafik distribusi ini memberikan gambaran jelas tentang karakteristik data yang mayoritas bersifat biner untuk gejala, variasi usia yang cukup luas, dan risiko stroke yang tersebar lebih kontinu.

   Tahapan ini membantu mengenali fitur yang memiliki distribusi tidak normal atau outlier ekstrem yang mungkin memengaruhi performa model.

4. **Analisis Korelasi Antar Fitur**
   
   Korelasi antar fitur dihitung dan divisualisasikan dalam bentuk heatmap. Heatmap ini membantu dalam memahami hubungan antara fitur-fitur gejala, usia, dan target prediksi risiko stroke. Dari hasil korelasi, ditemukan bahwa usia memiliki korelasi cukup tinggi dengan risiko stroke (0,73), dan risiko stroke memiliki korelasi kuat dengan status risiko biner (0,79). Sementara itu, korelasi antar gejala dan dengan risiko stroke cenderung rendah (sekitar 0,12–0,18), menunjukkan bahwa gejala tersebut bersifat relatif independen dan kontribusinya terhadap risiko stroke tidak terlalu besar dibandingkan usia.
     ![alt text](https://github.com/Wynnzzz/stroke-risk-prediction/blob/main/img/correlation.png?raw=true)
     
## Data Preparation

Tahapan data preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pemodelan bersih, relevan, dan sesuai untuk diterapkan pada algoritma machine learning. Berikut adalah urutan tahapan yang dilakukan dalam proses data preparation:

1. **Menghapus Data Duplikat**
   
   Sebanyak 1.021 data duplikat ditemukan dengan menggunakan fungsi `df.duplicated().sum()`. Duplikat ini dihapus dengan `df.drop_duplicates(inplace=True)` untuk menghindari bias akibat pengulangan data yang sama, yang dapat menyebabkan model belajar secara berlebihan pada data yang redundant.

2. **Feature Selection (Pemilihan Fitur)**  
   Fitur target `Stroke Risk (%)` dihapus karena fokus dari model adalah klasifikasi biner terhadap risiko stroke, yaitu fitur `At Risk (Binary)`. Fitur input yang digunakan terdiri dari 15 fitur gejala dan 1 fitur usia, sehingga total terdapat 16 fitur yang digunakan sebagai input model klasifikasi.

3. **Normalisasi Fitur Numerik**  
   Karena fitur `Age` memiliki skala nilai yang berbeda dibandingkan fitur-fitur lain yang bersifat biner, dilakukan proses normalisasi menggunakan `StandardScaler()` dari Scikit-learn. Normalisasi ini penting terutama untuk algoritma seperti Logistic Regression yang sensitif terhadap perbedaan skala antar fitur.

4. **Pemisahan Data Train dan Test**  
   Dataset dibagi menjadi dua bagian: 80% untuk data latih dan 20% untuk data uji. Pemisahan dilakukan menggunakan fungsi `train_test_split()` dari Scikit-learn dengan `random_state=42` untuk memastikan reprodusibilitas hasil. Pembagian ini bertujuan agar performa model dapat dievaluasi secara objektif menggunakan data yang tidak digunakan selama pelatihan.

Proses data preparation ini bertujuan untuk memastikan bahwa data yang masuk ke dalam model machine learning dalam kondisi optimal dan siap untuk dilakukan pelatihan. Dengan menghilangkan duplikasi, menyamakan skala fitur, dan membagi data secara adil, maka performa dan generalisasi model dapat meningkat secara signifikan.

# Modeling

Telah dilakukan pemodelan machine learning untuk menyelesaikan permasalahan klasifikasi risiko stroke menggunakan tiga algoritma, yaitu **Logistic Regression**, **Random Forest**, dan **XGBoost**.

Setiap model dibangun melalui tahapan:
- Pemisahan data menjadi training dan testing.
- Standarisasi data menggunakan `StandardScaler`.
- Hyperparameter tuning menggunakan Grid Search atau Randomized Search untuk mendapatkan kombinasi parameter terbaik:

| Algoritma            | Hyperparameter Terbaik                                      |
|----------------------|-------------------------------------------------------------|
| Logistic Regression  | `C=1`, `penalty='l2'`, `solver='liblinear'`                |
| Random Forest        | `n_estimators=200`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1` |
| XGBoost              | `n_estimators=200`, `learning_rate=0.2`, `max_depth=5`, `subsample=0.8` |

## Prinsip Kerja Algoritma

- **Logistic Regression**  
  Logistic Regression memodelkan probabilitas suatu kelas menggunakan **fungsi sigmoid** yang mengubah output linear menjadi rentang [0, 1]. Model ini mencari parameter (koefisien) yang memaksimalkan kemungkinan prediksi yang benar. Cocok untuk masalah klasifikasi biner dengan hubungan linier antar variabel.

- **Random Forest**  
  Random Forest adalah algoritma **ensemble learning** yang membangun banyak pohon keputusan (decision trees) secara paralel pada subset data dan fitur yang berbeda. Hasil akhir diperoleh melalui **voting mayoritas**. Hal ini membuatnya kuat terhadap overfitting dan cocok untuk data non-linear dan kompleks.

- **XGBoost (Extreme Gradient Boosting)**  
  XGBoost adalah implementasi dari teknik **gradient boosting** yang membangun model secara **sekuensial**, di mana tiap pohon baru dibentuk untuk memperbaiki kesalahan dari model sebelumnya. XGBoost efisien, mampu menangani missing value, dan dikenal menghasilkan performa tinggi dalam berbagai kompetisi data science.

## Kelebihan dan Kekurangan

| Algoritma            | Kelebihan                                                                 | Kekurangan                                                       |
|----------------------|---------------------------------------------------------------------------|------------------------------------------------------------------|
| Logistic Regression  | Cepat, sederhana, mudah diinterpretasi                                    | Tidak efektif untuk relasi non-linear                            |
| Random Forest        | Kuat terhadap overfitting, menangani data non-linear dengan baik          | Konsumsi memori tinggi, interpretasi hasil lebih kompleks        |
| XGBoost              | Akurasi tinggi, efisien, tangguh terhadap fitur kompleks & interaksi non-linear | Proses tuning kompleks dan memerlukan pemahaman mendalam         |

## Hasil dan Evaluasi

Proses improvement dilakukan melalui **hyperparameter tuning** untuk mengoptimalkan performa model sekaligus menghindari overfitting.

Berdasarkan hasil evaluasi, model **XGBoost** dipilih sebagai model terbaik karena menghasilkan metrik evaluasi sempurna:

- **Accuracy** = 1.00
- **Precision** = 1.00
- **Recall** = 1.00
- **F1-score** = 1.00

XGBoost terbukti memiliki kemampuan terbaik dalam menangani data kompleks dan fitur penting secara efisien pada kasus klasifikasi risiko stroke.

## Evaluation

Pada proyek klasifikasi risiko stroke ini, digunakan empat metrik evaluasi utama, yaitu **akurasi**, **precision**, **recall**, dan **F1-score**. Keempat metrik ini dipilih karena dapat memberikan gambaran menyeluruh terkait performa model klasifikasi, khususnya dalam konteks data yang memiliki dua kelas (binary classification), yaitu *berisiko* dan *tidak berisiko* stroke.

- **Akurasi** mengukur seberapa banyak prediksi yang benar dibandingkan dengan seluruh jumlah data. Formula:
  
  $$
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  $$

- **Precision** menunjukkan proporsi prediksi positif yang benar-benar positif. Formula:

$$
  \text{Precision} = \frac{TP}{TP + FP}
$$

- **Recall** (Sensitivity) mengukur seberapa banyak data positif yang berhasil dideteksi. Formula:

$$
  \text{Recall} = \frac{TP}{TP + FN}
$$

- **F1-score** adalah rata-rata harmonik antara precision dan recall, digunakan untuk menyeimbangkan keduanya. Formula:

$$
  \text{F1-score} = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### Hasil Evaluasi Model

### Tabel Perbandingan Metrik Evaluasi

| Model               | Akurasi | Precision | Recall | F1-Score |
|---------------------|---------|-----------|--------|----------|
| Logistic Regression | 1.00    | 1.00      | 1.00   | 1.00     |
| Random Forest       | 0.95    | 0.95      | 0.94   | 0.94     |
| XGBoost             | 1.00    | 1.00      | 1.00   | 1.00     |

Tiga model telah dibandingkan berdasarkan metrik-metrik di atas, dan hasil terbaik adalah sebagai berikut:

1. **Logistic Regression**:  
   Akurasi, precision, recall, dan F1-score sebesar **1.00**.  
   Model ini menunjukkan performa sempurna, namun karena sifatnya linear, ada potensi overfitting pada data uji jika tidak diuji lebih lanjut dengan data baru.

2. **Random Forest**:  
   Akurasi sebesar **0.95**, dengan precision dan recall yang juga tinggi (**0.94–0.97**).  
   Meskipun sangat baik, performanya masih sedikit di bawah XGBoost dan Logistic Regression.

3. **XGBoost**:  
   Memberikan hasil sempurna dengan akurasi, precision, recall, dan F1-score **1.00**.  
   Model ini juga lebih stabil dibanding Logistic Regression karena mampu menangani kompleksitas fitur dan interaksi antar variabel.

### Kesimpulan

Berdasarkan evaluasi metrik dan kestabilan hasil, **XGBoost** dipilih sebagai model terbaik untuk memprediksi risiko stroke. Hasil ini juga mendukung pernyataan solusi yang menginginkan model berkinerja tinggi dan mampu menangani data kompleks.
