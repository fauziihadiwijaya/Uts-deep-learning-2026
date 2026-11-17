# Klasifikasi Multimodal Cacat Produk Pabrik (Multimodal Defect Classification)

Dokumentasi ini merangkum alur eksperimen, perancangan arsitektur, serta analisis penyelesaian **Task 1 – Multimodal Classification: Product Defects**. Proyek ini bertujuan mengklasifikasikan kualitas komponen pabrik menggunakan dua modalitas sensor secara bersamaan: **citra visual (kamera)** dan **sinyal tekanan (press sensor)**.

Seluruh eksperimen dijalankan menggunakan lingkungan **Google Colab dengan akselerator GPU NVIDIA T4**.

---

## 1. Gambaran Masalah & Dataset

Sistem inspeksi otomatis pada lini produksi mengamati setiap unit komponen dan memetakannya ke dalam **7 kelas kondisi**:
1. `normal`
2. `gores` (scratch)
3. `noda` (stain)
4. `retak` (crack)
5. `gelembung` (bubble)
6. `las_dingin` (cold weld)
7. `kontaminasi` (contamination)

Tantangan utama pada dataset ini adalah sifat cacat yang asimetris antar-modalitas: beberapa cacat hanya terlihat secara visual, beberapa hanya terdeteksi melalui anomali kurva tekanan, dan sebagian lainnya tampak samar pada kedua modalitas. Metrik evaluasi utama yang digunakan adalah **Macro-F1** untuk memastikan performa merata di seluruh kelas.

---

## 2. Alur Pengerjaan & Metodologi (Pipeline)

Eksperimen disusun secara bertahap mengikuti alur analisis data hingga evaluasi stabilitas model:

### Tahap A: Exploratory Data Analysis (EDA)
Sebelum membangun model, dilakukan inspeksi mendalam terhadap karakteristik data:
* **Visualisasi Sampel:** Menampilkan contoh pasangan gambar dan plot deret waktu (*time-series*) sinyal tekanan untuk ketujuh kelas guna memahami ciri fisik tiap cacat.
* **Distribusi Kelas:** Memeriksa ketidakseimbangan jumlah sampel (*class imbalance*) pada data latih sebagai dasar penentuan *class weights* atau strategi *sampling*.
* **Analisis Data Hilang (*Missing Sensor*):** Mengidentifikasi persentase rekaman sinyal sensor yang kosong/rusak (`NaN` atau nol) agar model tidak bergantung secara mutlak pada ketersediaan sensor.
* **Pergeseran Domain (*Train–Test Split Shift*):** Membandingkan distribusi lini produksi (*production line*) antara data latih (*train*) dan data uji (*test*) untuk mendeteksi potensi *domain shift*.

### Tahap B: Perancangan Arsitektur & Strategi Fusi
Model dirancang menggunakan pendekatan *dual-encoder* yang disesuaikan dengan karakteristik masing-masing data:
* **Image Encoder:** Menggunakan arsitektur CNN (seperti *ResNet* ringan atau *MobileNet*) untuk mengekstrak fitur spasial dan tekstur permukaan komponen.
* **Signal Encoder:** Menggunakan arsitektur *1D-CNN* atau *Bi-GRU* untuk menangkap pola temporal dan lonjakan tekanan selama proses *pressing*.
* **Perbandingan Strategi Fusi (*Fusion*):**
  1. **Early / Concatenation Fusion:** Menggabungkan vektor fitur dari *image encoder* dan *signal encoder* sebelum masuk ke *classifier head*.
  2. **Cross-Attention / Gated Fusion:** Memungkinkan fitur gambar dan fitur sinyal saling memberi bobot perhatian (*attention*) sehingga model dapat memprioritaskan modalitas yang paling informatif pada setiap sampel.

### Tahap C: Studi Ablasi Modalitas (Unimodal vs Multimodal)
Untuk membuktikan kontribusi masing-masing sensor, dilakukan pelatihan terpisah:
* **Image-Only Model:** Melatih cabang visual saja untuk melihat kelas mana yang mudah dikenali secara optik (misalnya `gores` dan `noda`).
* **Sensor-Only Model:** Melatih cabang sinyal saja untuk mengidentifikasi cacat internal/mekanis (misalnya `las_dingin`).
* **Analisis Per-Kelas:** Membandingkan skor F1 per kelas antara model unimodal dan multimodal untuk memetakan ketergantungan modalitas tiap jenis cacat.

### Tahap D: Penanganan Label Tidak Sempurna (*Noisy Labels*)
Karena label data latih berasal dari inspeksi manual yang rentan human error:
* **Deteksi Anomali Label:** Menggunakan prediksi *out-of-fold* (*confident learning* sederhana) untuk menginspeksi sampel dengan nilai *loss* tertinggi yang berpotensi salah label (*mislabels*).
* **Teknik Mitigasi:** Menerapkan **Label Smoothing** (atau *Symmetric Cross-Entropy*) pada fungsi *loss* untuk mencegah model terlalu percaya diri (*overconfident*) terhadap label yang keliru.

### Tahap E: Penanganan Domain Shift Antar Lini Produksi
* **Evaluasi Per-Lini:** Mengukur skor Macro-F1 secara terpisah pada setiap lini produksi di set validasi untuk melihat penurunan performa pada lini yang berbeda karakteristiknya.
* **Teknik Adaptasi:** Menerapkan **Per-Line Normalization** (normalisasi sinyal berdasarkan statistik masing-masing lini produksi) serta **Data Augmentation** (rotasi/flip ringan pada citra dan *jittering* pada sinyal) agar representasi fitur lebih invarian terhadap perbedaan mesin.

### Tahap F: Ketahanan Terhadap Hilangnya Modalitas (*Missing Modality*)
Untuk menjamin model tetap memberikan prediksi masuk akal saat sensor tekanan gagal merekam:
* **Modality Dropout:** Selama fase pelatihan, vektor fitur dari *signal encoder* secara acak di-nol-kan (*zero-masking*) dengan probabilitas tertentu. Hal ini melatih *classifier* untuk tetap mampu mengambil keputusan berbasis fitur visual semata ketika data sensor tidak tersedia saat inferensi.

### Tahap G: Validasi Pelatihan & Stabilitas Model
* **Sanity Check (Overfit 1 Batch):** Menguji arsitektur pada satu *batch* kecil untuk memastikan *loss* dapat turun mendekati nol dan gradien mengalir dengan benar.
* **Kurva Pembelajaran (*Learning Curves*):** Memantau grafik *train vs validation loss* dan *Macro-F1* untuk mendeteksi *overfitting* dan menentukan *early stopping*.
* **Evaluasi Multi-Seed:** Melatih konfigurasi model final menggunakan **minimal 3 random seed berbeda** (misalnya `seed = 42, 100, 2024`) untuk melaporkan rata-rata dan standar deviasi skor Macro-F1 secara objektif.

---

## 3. Cara Menjalankan di Google Colab (GPU T4)

1. Buka file notebook (`.ipynb`) di Google Colab.
2. Pastikan akselerator perangkat keras telah diatur ke GPU:
   * Pilih menu **Runtime** > **Change runtime type** > **T4 GPU**.
3. Sesuaikan *path* direktori dataset (misalnya pada Google Drive atau direktori lokal `/content/`).
4. Jalankan sel secara berurutan dari bagian *Setup & EDA* hingga *Multi-Seed Final Evaluation*.

---

## 4. Struktur Direktori Output

Setelah notebook dijalankan sepenuhnya, artefak berikut akan dihasilkan:
* `checkpoints/` : Bobot model terbaik (`.pt`) untuk setiap eksperimen dan *seed*.
* `figures/` : Grafik distribusi data, kurva *training/validation*, dan *confusion matrix*.
* `submission.csv` : Hasil prediksi final pada data uji yang telah mencakup penanganan *missing sensor*.
