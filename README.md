# Tugas Kelompok: Linear Regression dan Logistic Regression

## Identitas Kelompok

| Nama | NIM |
|---|---|
| Isi nama anggota 1 | Isi NIM |
| Isi nama anggota 2 | Isi NIM |
| Isi nama anggota 3 | Isi NIM |

> Hapus baris anggota yang tidak digunakan. Jumlah anggota maksimal 3 orang.

## Deskripsi Tugas

Tugas ini membahas dua jenis model dalam Sains Data:

1. **Linear Regression**, untuk memprediksi nilai berupa angka.
2. **Logistic Regression**, untuk mengelompokkan data ke dalam dua kelas.

Pengerjaan dilakukan menggunakan Python di Google Colab. Setiap notebook berisi penjelasan, proses pengolahan data, grafik, pembuatan model, dan evaluasi hasil.

---

## File dalam Repository

- `A_linear_regression.ipynb` berisi analisis Linear Regression menggunakan dataset California Housing Prices.
- `B_logistic_regression.ipynb` berisi analisis Logistic Regression menggunakan dataset Breast Cancer Wisconsin.
- `README.md` berisi penjelasan tugas, sumber data, cara menjalankan notebook, dan ringkasan hasil.

Dataset tidak perlu disimpan di folder `data` karena kedua notebook mengambil data langsung dari Kaggle menggunakan `kagglehub`.

---

# Bagian A — Linear Regression

## A1. Dataset California Housing Prices

Dataset untuk bagian A berasal dari Kaggle:

[California Housing Prices — Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

Dataset ini berisi informasi tentang beberapa wilayah di California, Amerika Serikat. Satu baris data mewakili satu wilayah. Dataset memiliki sekitar **20.640 baris dan 10 kolom**.

Kolom-kolom yang digunakan adalah:

- `longitude`: posisi wilayah dari arah barat ke timur.
- `latitude`: posisi wilayah dari arah selatan ke utara.
- `housing_median_age`: usia tengah rumah di wilayah tersebut.
- `total_rooms`: jumlah seluruh ruangan di wilayah tersebut.
- `total_bedrooms`: jumlah seluruh kamar tidur di wilayah tersebut.
- `population`: jumlah penduduk di wilayah tersebut.
- `households`: jumlah rumah tangga di wilayah tersebut.
- `median_income`: pendapatan tengah penduduk. Angkanya menggunakan satuan puluhan ribu dolar.
- `ocean_proximity`: keterangan jarak wilayah dari laut. Kolom ini berupa kategori atau teks.
- `median_house_value`: nilai tengah rumah di wilayah tersebut.

Target yang diprediksi adalah `median_house_value`. Kolom lainnya digunakan sebagai fitur untuk membantu model memperkirakan nilai rumah.

## Proses pada Bagian A

Pada notebook A dilakukan beberapa langkah berikut:

1. Mengambil dataset langsung dari Kaggle menggunakan `kagglehub`.
2. Melihat ukuran data, jenis kolom, statistik deskriptif, data kosong, dan data duplikat.
3. Membuat tiga visualisasi:
   - histogram `median_house_value`;
   - heatmap korelasi antar kolom angka;
   - scatter plot `median_income` terhadap `median_house_value`.
4. Menghapus data duplikat.
5. Mengisi data kosong dengan nilai median.
6. Menangani outlier pada fitur angka menggunakan batas IQR.
7. Mengubah kolom `ocean_proximity` dari teks menjadi angka dengan one-hot encoding.
8. Melakukan scaling pada fitur angka.
9. Memilih 8 fitur terbaik dengan `SelectKBest`.
10. Membagi data menjadi 80% data latih dan 20% data uji dengan `random_state=42`.
11. Membuat model `LinearRegression`.
12. Menghitung MAE, MSE, RMSE, dan R-squared pada data latih dan data uji.
13. Membuat grafik nilai aktual dan nilai prediksi serta residual plot.

## Ringkasan hasil Bagian A

Berdasarkan output notebook yang sudah dijalankan:

```text
R-squared data latih: 0.6063
R-squared data uji  : 0.5841
```

Nilai R-squared pada data latih dan data uji cukup dekat. Ini menunjukkan bahwa perbedaan hasil pada kedua data tidak terlalu jauh. Dari hasil tersebut, tidak terlihat overfitting yang besar.

Model masih memiliki kesalahan prediksi. Hal ini wajar karena harga rumah tidak hanya dipengaruhi oleh kolom-kolom yang ada di dataset. Kondisi lingkungan, fasilitas, dan faktor lain juga bisa memengaruhi harga rumah.

---

# Bagian B — Logistic Regression

## B1. Dataset Breast Cancer Wisconsin

Dataset untuk bagian B berasal dari Kaggle:

[Breast Cancer Wisconsin Diagnostic — Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

Dataset ini berisi hasil pengukuran sel dari pemeriksaan payudara. Satu baris data mewakili satu sampel pemeriksaan. Dataset memiliki sekitar **569 baris dan 33 kolom** sebelum kolom yang tidak diperlukan dibuang.

Target pada dataset ini adalah `diagnosis`, dengan dua kelas:

- `M` atau 1: Malignant, yaitu ganas.
- `B` atau 0: Benign, yaitu jinak.

Beberapa fitur yang digunakan antara lain:

- `radius_mean`: rata-rata ukuran jari-jari sel.
- `texture_mean`: rata-rata tekstur sel.
- `perimeter_mean`: rata-rata keliling sel.
- `area_mean`: rata-rata luas sel.
- `smoothness_mean`: rata-rata kehalusan sel.
- `compactness_mean`: rata-rata kepadatan bentuk sel.
- `concavity_mean`: rata-rata cekungan pada sel.
- `concave points_mean`: rata-rata titik cekung pada sel.
- `symmetry_mean`: rata-rata kesimetrisan sel.
- `fractal_dimension_mean`: ukuran kerumitan bentuk sel.

Selain fitur tersebut, terdapat fitur dengan akhiran `_se` dan `_worst`. Fitur `_se` berkaitan dengan kesalahan standar pengukuran, sedangkan `_worst` berisi nilai pengukuran yang paling tinggi atau paling buruk.

Kolom `id` tidak digunakan karena hanya nomor identitas sampel. Kolom `Unnamed: 32` juga dibuang karena tidak berisi informasi yang diperlukan.

## Proses pada Bagian B

Pada notebook B dilakukan beberapa langkah berikut:

1. Mengambil dataset langsung dari Kaggle menggunakan `kagglehub`.
2. Membuang kolom `id` dan `Unnamed: 32`.
3. Mengubah target `diagnosis` dari huruf menjadi angka:
   - `M` menjadi 1;
   - `B` menjadi 0.
4. Melihat statistik deskriptif, data kosong, dan data duplikat.
5. Membuat tiga visualisasi:
   - jumlah data pada setiap kelas;
   - box plot `radius_mean` pada kelas jinak dan ganas;
   - heatmap korelasi.
6. Menghapus data duplikat.
7. Mengisi data kosong dengan median jika ada.
8. Melakukan scaling pada fitur angka.
9. Menggunakan `class_weight='balanced'` karena jumlah kelas jinak dan ganas tidak sama.
10. Memilih 10 fitur terbaik menggunakan `SelectKBest` dan `f_classif`.
11. Membagi data menjadi 80% data latih dan 20% data uji.
12. Menggunakan `stratify=y` agar perbandingan kelas pada data latih dan data uji tetap mirip.
13. Membuat model `LogisticRegression`.
14. Menghitung accuracy, precision, recall, F1-score, dan AUC.
15. Membuat confusion matrix dan ROC curve.

## Ringkasan hasil Bagian B

Berdasarkan output notebook yang sudah dijalankan:

```text
Accuracy : 0.9561
Precision: 0.9302
Recall   : 0.9524
F1-score : 0.9412
AUC      : 0.9974
```

Hasil tersebut menunjukkan bahwa model dapat membedakan kelas jinak dan ganas dengan cukup baik pada data uji. Recall juga perlu diperhatikan karena menunjukkan seberapa banyak sampel ganas yang berhasil ditemukan oleh model.

Pada kasus seperti ini, false negative perlu diperhatikan. False negative terjadi ketika sampel yang sebenarnya ganas justru diprediksi sebagai jinak. Model ini hanya digunakan untuk latihan analisis data, bukan untuk menggantikan pemeriksaan atau keputusan dokter.

---

## Link Google Colab

Notebook yang sudah dikerjakan dapat dibuka melalui link berikut:

- **Bagian A — Linear Regression:** [Buka notebook A di Google Colab](https://colab.research.google.com/drive/1IGpTixvt0zSN65vN-LSQXSg4lxzvsqcC)
- **Bagian B — Logistic Regression:** [Buka notebook B di Google Colab](https://colab.research.google.com/drive/15WyOGR0TmpzddLVujqXD4ds9APXG4Oc3)

---

## Cara Menjalankan Notebook

1. Buka salah satu link Google Colab di atas.
2. Jalankan cell dari atas ke bawah.
3. Saat cell `kagglehub` dijalankan, ikuti permintaan login atau izin Kaggle jika muncul.
4. Tunggu sampai dataset selesai diambil.
5. Lanjutkan menjalankan cell berikutnya sampai bagian terakhir.
6. Pastikan semua grafik dan hasil angka sudah muncul.
7. Jika ada perubahan pada notebook, simpan kembali notebook tersebut.

Kedua notebook mengambil data langsung dari Kaggle, sehingga tidak perlu meng-upload file CSV secara manual ke Google Colab.

---

## Pembagian Tugas Anggota

Pembagian tugas dibuat berdasarkan bagian pekerjaan supaya setiap anggota memiliki tanggung jawab yang jelas. Anggota 3 adalah saya, sehingga saya mengerjakan seluruh bagian Logistic Regression dan menyusun README.

### Anggota 1 — Dataset dan preprocessing Linear Regression

Tugas Anggota 1:

- mencari dan memeriksa dataset California Housing Prices dari Kaggle;
- memastikan dataset dapat diambil menggunakan `kagglehub`;
- menjelaskan asal dataset, jumlah baris, jumlah kolom, fitur, dan target;
- memeriksa tipe data, data kosong, dan data duplikat;
- menghapus data duplikat;
- menangani data kosong dengan median atau cara yang sesuai;
- menangani outlier pada fitur angka menggunakan aturan IQR;
- memisahkan fitur dan target pada Linear Regression;
- membantu menyiapkan pipeline preprocessing;
- menjelaskan alasan setiap langkah preprocessing.

Hasil pekerjaan Anggota 1 digunakan pada bagian A1 dan A3 notebook `A_linear_regression.ipynb`.

### Anggota 2 — Eksplorasi, visualisasi, dan evaluasi Linear Regression

Tugas Anggota 2:

- membuat statistik deskriptif dataset Linear Regression;
- membuat histogram distribusi target;
- membuat heatmap korelasi antar fitur angka;
- membuat scatter plot fitur terhadap target;
- menulis temuan sederhana dari setiap visualisasi;
- membantu memilih fitur berdasarkan korelasi atau `SelectKBest`;
- membagi data menjadi data latih dan data uji;
- membuat dan menjalankan model `LinearRegression`;
- menampilkan intercept dan koefisien;
- menghitung MAE, MSE, RMSE, dan R-squared;
- membuat grafik nilai aktual dan nilai prediksi;
- membuat residual plot;
- membantu membahas overfitting, underfitting, fitur yang berpengaruh, dan keterbatasan model.

Hasil pekerjaan Anggota 2 digunakan pada bagian A2, A4, A5, A6, A7, dan A8 notebook `A_linear_regression.ipynb`.

### Anggota 3 — Logistic Regression, evaluasi, dan dokumentasi

Tugas Anggota 3 adalah bagian saya:

- mengambil dataset Breast Cancer Wisconsin dari Kaggle menggunakan `kagglehub`;
- menjelaskan asal dataset, jumlah baris, jumlah kolom, fitur utama, dan target;
- menjelaskan arti kelas `B` sebagai jinak dan `M` sebagai ganas;
- membuang kolom `id` dan `Unnamed: 32` karena tidak diperlukan;
- mengubah target `diagnosis` dari huruf menjadi angka, yaitu `B=0` dan `M=1`;
- memeriksa statistik deskriptif, data kosong, data duplikat, dan jumlah setiap kelas;
- membuat visualisasi distribusi kelas, box plot, dan heatmap korelasi;
- menulis temuan dari setiap visualisasi;
- menghapus duplikat dan menangani data kosong;
- melakukan scaling pada fitur angka;
- menangani jumlah kelas yang tidak seimbang dengan `class_weight='balanced'`;
- memilih 10 fitur terbaik menggunakan `SelectKBest` dan `f_classif`;
- membagi data dengan perbandingan 80:20 menggunakan `train_test_split`;
- menggunakan `stratify=y` agar perbandingan kelas pada data latih dan data uji tetap mirip;
- membuat model `LogisticRegression`;
- menampilkan dan menjelaskan koefisien serta intercept;
- menghitung accuracy, precision, recall, F1-score, dan AUC;
- membuat confusion matrix dan ROC curve;
- menjelaskan false positive dan false negative;
- membahas fitur yang berpengaruh dan keterbatasan model;
- menyusun README dan merapikan dokumentasi repository;
- memeriksa kembali notebook agar dapat dijalankan dari awal sampai akhir.

Hasil pekerjaan Anggota 3 digunakan pada bagian B1 sampai B8 notebook `B_logistic_regression.ipynb` serta pada file `README.md`.

### Kerja sama dan pemeriksaan akhir

Walaupun tugas dibagi seperti di atas, semua anggota tetap saling membantu saat ada error atau bagian yang belum dipahami. Sebelum dikumpulkan, kami memeriksa bersama bahwa:

- kedua notebook dapat dijalankan dari awal sampai akhir;
- grafik dan hasil evaluasi sudah muncul;
- penjelasan di setiap bagian sesuai dengan kode;
- nama dan NIM anggota sudah dicantumkan;
- repository sudah berisi file yang diperlukan;
- link repository yang sama dikumpulkan oleh setiap anggota.

---

## Pengumpulan

Repository GitHub dibuat dalam keadaan **public**. Isi repository minimal terdiri dari:

```text
README.md
A_linear_regression.ipynb
B_logistic_regression.ipynb
```

Karena data diambil langsung dari Kaggle, folder `data` tidak wajib dimasukkan ke repository.

Link repository GitHub dikumpulkan melalui Google Classroom. Setiap anggota kelompok wajib mengumpulkan link repository GitHub yang sama di akun Classroom masing-masing.
