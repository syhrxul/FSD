# Tugas Kelompok: Linear Regression dan Logistic Regression

## Identitas Kelompok

| Nama | NIM |
|---|---|
| Sa'adina Lutfia Rahmat| 25523033 |
| Shalwa Ikara Putri | 25523004 |
| Syahrul Imtikhan Ahmad | 25523064 |

## Deskripsi Tugas

1. **Linear Regression**, untuk memprediksi nilai berupa angka.
2. **Logistic Regression**, untuk mengelompokkan data ke dalam dua kelas.

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

- Anggota 1: mengambil dataset dan mengerjakan preprocessing Linear Regression.
- Anggota 2: mengerjakan visualisasi dan evaluasi Linear Regression.
- Anggota 3: mengerjakan Logistic Regression, evaluasi, dan README.

---

