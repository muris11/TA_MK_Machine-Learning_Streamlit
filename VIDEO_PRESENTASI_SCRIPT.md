# Script Video Presentasi UAS Machine Learning

Judul video: UAS_ML_Muhammad_Rifqy_Saputra_2307046_Prediksi_Kondisi_Sosial_Jawa_Barat
Durasi target: 5-10 menit

## 1. Pembukaan
Assalamualaikum, perkenalkan saya Muhammad Rifqy Saputra, NIM 2307046. Pada video ini saya mempresentasikan project UAS Machine Learning berjudul Prediksi Kondisi Sosial Provinsi Jawa Barat. Pilar Smart City yang digunakan adalah Smart Living dan Smart Governance.

## 2. Dataset
Dataset berisi indikator sosial ekonomi Jawa Barat, seperti gini ratio, tingkat pengangguran terbuka, inflasi, dan indeks pembangunan manusia. Tahun tetap digunakan sebagai konteks input aplikasi. Target regresi adalah estimasi kemiskinan, sedangkan target klasifikasi adalah prioritas intervensi sosial.

## 3. Proses Machine Learning
Pada tahap UAS saya memperbaiki preprocessing, terutama nilai numerik yang salah terbaca sebagai tanggal, scaling gini ratio dan inflasi, imputasi median, standardisasi, dan deduplikasi agar hasil evaluasi tidak 1.0000 semua. Algoritma yang dibandingkan adalah Random Forest, SVR, Polynomial Regression, SVC, dan Logistic Regression.

## 4. Hasil Evaluasi
Model regresi terbaik adalah RandomForestRegressor dengan R2 0.9584, MAE 0.0489, dan RMSE 0.0787. Model klasifikasi terbaik adalah SVC dengan accuracy 0.9007 dan F1-score 0.8957.

## 5. Demo Aplikasi
Aplikasi dibuat menggunakan Streamlit. Saya akan menunjukkan input data, proses prediksi, hasil estimasi kemiskinan, level prioritas, rekomendasi kebijakan, tampilan dataset, grafik visualisasi, login admin, dan upload CSV.

## 6. Kesimpulan
Project ini bermanfaat sebagai decision support untuk membantu pemerintah menentukan prioritas intervensi sosial berbasis data. Risiko prediksi salah tetap ada, sehingga hasil model harus divalidasi dengan data lapangan. Privasi dijaga karena dataset bersifat agregat dan bukan data pribadi individu.
