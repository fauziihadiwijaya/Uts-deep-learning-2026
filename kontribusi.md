# Kontribusi Anggota
## Anggota dan peran

| Anggota | Peran utama |
| --- | --- |
| Mohamad Faisal Bashir (Anggota 1) | PIC Task 1 (Classification), image representation pada Task 3 |
| Mohammad Fauzi Hadiwijaya (Anggota 2) | PIC Task 2 (Regression), spectrum representation pada Task 3 |

## Rincian kontribusi

### Mohamad Faisal Bashir (Anggota 1)

**Task 1 — Multimodal Classification: Product Defects**
- EDA dataset klasifikasi: distribusi kelas, contoh citra dan sinyal per kelas, sensor yang hilang, dan perbedaan lini produksi antara train dan test.
- Preprocessing gambar dan sensor, termasuk normalisasi global dan per lini serta mask untuk sensor yang hilang.
- Membuat image encoder (CNN) dan sensor encoder (1D-CNN).
- Menguji tiga strategi fusion (concat, late, dan cross-attention), model image-only dan sensor-only, teknik label noise, domain shift, dan missing modality.
- Training dan evaluasi model: sanity check, learning curve, dan tiga seed.
- Menghasilkan `prediksi_klasifikasi.csv`.

**Task 3 — Multimodal Clustering: Material Types**
- Image representation dan image encoder (SimCLR): eksplorasi distorsi citra, augmentasi, training, evaluasi representasi, dan ekspor image embedding.

### Mohammad Fauzi Hadiwijaya (Anggota 2)

**Task 2 — Time-Series Regression: Remaining Useful Life**
- EDA data histori mesin.
- Preprocessing dan normalisasi.
- Membuat label RUL dengan batas 150 siklus.
- Membuat sliding window.
- Membandingkan LSTM/GRU dengan model lainnya.
- Menghasilkan `prediksi_rul.csv`.

**Task 3 — Multimodal Clustering: Material Types**
- Spectrum representation dan spectrum encoder (1D-CNN autoencoder): EDA spektrum, preprocessing, training, evaluasi rekonstruksi, dan ekspor spectrum embedding.

### Dikerjakan bersama

**Task 3 — Multimodal Clustering (notebook gabungan)**
- Eksplorasi distorsi spektrum dan koreksi efek mikroskop.
- Feature fusion image dan spectrum embedding.
- Menentukan jumlah cluster.
- Perbandingan metode clustering (KMeans, GMM, DEC).
- Evaluasi dan visualisasi (UMAP, analisis cluster yang sulit dipisahkan).
- Menghasilkan `cluster.csv`.

**Dokumen kelompok**
- `laporan.pdf`, `kode/README.md`, dan `kontribusi.md`.

## Bukti kontribusi di GitHub

Kelompok memakai sistem kolaborator: kedua anggota memiliki akses tulis ke repository utama dan mengunggah langsung ke branch `main`, sehingga tidak ada pull request. Kontribusi setiap anggota dapat dilihat dari riwayat commit:

| Anggota | Riwayat commit |
| --- | --- |
| Mohamad Faisal Bashir | [commits oleh mohamadfaisalbashir](https://github.com/fauziihadiwijaya/Uts-deep-learning-2026/commits/main?author=mohamadfaisalbashir) |
| Mohammad Fauzi Hadiwijaya | [commits oleh fauziihadiwijaya](https://github.com/fauziihadiwijaya/Uts-deep-learning-2026/commits/main?author=fauziihadiwijaya) |

## Penggunaan AI

| Alat | Bagian yang dibantu |
| --- | --- |
| Claude (Anthropic) | Menyusun kode notebook Task 1, Task 2, dan Task 3 (image representation, spectrum representation, dan notebook gabungan) |
| Claude (Anthropic) | Penulisan penjelasan cell dan README dilakukan dengan bantuan AI (AI-assisted) |

## Persetujuan

Isi dokumen ini telah disetujui oleh seluruh anggota kelompok.

- [x] Mohamad Faisal Bashir
- [x] Mohammad Fauzi Hadiwijaya
