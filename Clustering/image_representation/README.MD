# Task 3: Image Representation & Encoder

Folder ini berisi cabang **image** dari Task 3 (Multimodal Clustering: Material Types). Isinya adalah image encoder yang dilatih tanpa label dengan **SimCLR** (contrastive learning) untuk menghasilkan embedding citra mikrostruktur. Embedding ini kemudian digabung dengan spectrum embedding (folder `spectrum_representation`) di notebook gabungan, untuk fusion, clustering, dan pembuatan `cluster.csv`.

## Isi folder

| File | Keterangan |
| --- | --- |
| `image_representation_encoder.ipynb` | Notebook utama, lengkap dengan output hasil run |
| `simclr_e100_b512.pt` | Checkpoint SimCLR lengkap (encoder + projection head + riwayat training). Dimuat otomatis oleh notebook sehingga training tidak perlu diulang |
| `image_encoder_state_dict.pt` | Bobot final image encoder (backbone saja, 4.86 MB) |
| `image_encoder_config.json` | Hyperparameter, augmentasi, seed, versi torch, dan hasil evaluasi representasi |
| `README.md` | Dokumen ini |

File embedding (`image_embeddings.csv.gz` 26.8 MB dan `image_embeddings.npy` 36.9 MB) tidak disimpan di repository karena dapat dibuat ulang dari bobot dalam hitungan detik. Keduanya dihasilkan otomatis di Bagian 6 notebook.

## Data

Notebook memakai dua file dari data UTS (lihat `kamus_data.md`):

| File | Isi yang dipakai |
| --- | --- |
| `clustering.npz` | Array `gambar`: uint8 (36000, 64, 64), citra mikrostruktur. Array `spektrum` tidak dipakai di notebook ini |
| `clustering_meta.csv` | Kolom `indeks`, `id_sampel`, `mikroskop` (M1: 21.636 sampel, M2: 14.364 sampel) |

Data tidak disertakan di repository.

## Cara menjalankan

1. Buka notebook di Google Colab, lalu pilih **Runtime → Change runtime type → T4 GPU**.
2. Taruh `clustering.npz` dan `clustering_meta.csv` di Google Drive pada folder `MyDrive/UTS_DL` (nilai default `DATA_DIR`).
3. Agar hasil sama persis tanpa training ulang, salin `simclr_e100_b512.pt` dari folder ini ke `MyDrive/UTS_DL/hasil_tugas3/`.
4. Jalankan **Runtime → Run all**.

Jika `simclr_e100_b512.pt` tidak ditemukan, notebook melatih ulang SimCLR selama 100 epoch (sekitar 17 menit di T4). Semua output disimpan ke `MyDrive/UTS_DL/hasil_tugas3/`. Untuk uji cepat bahwa kode berjalan, ubah `SMOKE = True` (400 sampel, 1 epoch; angkanya tidak bermakna).

## Metode

**Preprocessing.** Setiap citra distandardisasi dengan rata-rata dan simpangan bakunya sendiri. Langkah ini menghapus perbedaan brightness dan contrast linear, termasuk yang berasal dari perbedaan mikroskop.

**Augmentasi.** SimCLR membuat encoder tidak peka terhadap apa pun yang diubah oleh augmentasi, sehingga augmentasi dipilih sesuai distorsi yang ingin diabaikan:

| Augmentasi | Distorsi yang dinetralkan |
| --- | --- |
| Rotasi bebas 0–360° dan flip | Orientasi sampel di bawah mikroskop |
| Zoom 0.75–1.33× dan geser kecil | Perbedaan perbesaran (skala) |
| Gamma 0.7–1.4 | Respons kecerahan non-linear antar mikroskop |
| Gaussian blur sigma 0–1.2 (50% sampel) | Perbedaan fokus dan ketajaman |
| Gaussian noise std 0–0.04 | Noise kamera |

**Arsitektur.** CNN bergaya ResNet ringan (1.21 juta parameter), sama dengan image encoder Task 1: satu konvolusi pada resolusi penuh, tiga blok residual (64 → 32 → 16 → 8), lalu global average pooling menjadi embedding 256 dimensi. Projection head MLP (256 → 256 → 128) hanya dipakai saat training.

**Training.** Loss NT-Xent dengan suhu 0.2, batch 512, AdamW (lr 1e-3, weight decay 1e-4), warmup 5% lalu cosine decay, mixed precision, 100 epoch. Data dibagi 90/10 (stratifikasi mikroskop) untuk memantau loss validasi.

**Ekstraksi embedding.** Embedding diambil dari backbone, dirata-rata atas 8 transformasi dihedral (4 rotasi 90° × flip) sehingga invarian terhadap rotasi dan flip.

## Hasil

**Eksplorasi distorsi citra (rata-rata per mikroskop):**

| Mikroskop | Brightness | Contrast | Sharpness | Skala tekstur (px) |
| --- | --- | --- | --- | --- |
| M1 | 0.501 | 0.077 | 0.020 | 7.02 |
| M2 | 0.573 | 0.078 | 0.035 | 5.49 |

Citra M2 lebih terang, lebih tajam, dan tekstur terukurnya lebih kecil dibanding M1. Contrast hampir sama.

**Sanity check (overfit satu batch, 128 citra):** loss awal 5.102 (tebakan acak ln 255 = 5.541), loss akhir 0.999, pos_acc 100%. Dengan suhu 0.2, loss NT-Xent tidak bisa mencapai nol: batas bawah teoretisnya sekitar log(1 + 254 · e⁻⁵) ≈ 1.0, sehingga 0.999 berarti batch sudah dihafal sempurna.

**Training SimCLR (100 epoch, 992 detik di T4):**

| Epoch | Train loss | Train pos_acc | Val loss | Val pos_acc |
| --- | --- | --- | --- | --- |
| 1 | 5.747 | 0.026 | 5.321 | 0.062 |
| 50 | 3.105 | 0.849 | 3.137 | 0.845 |
| 100 | 2.986 | 0.897 | 2.992 | 0.892 |

Loss validasi turun bersama loss training dengan selisih kecil, jadi tidak ada tanda overfitting.

**Evaluasi representasi tanpa label (1.000 citra untuk invariansi, 10.000 untuk kebocoran mikroskop):**

| Representasi | Dim | pos_cos | neg_cos | Gap | Retrieval top-1 | Kebocoran mikroskop |
| --- | --- | --- | --- | --- | --- | --- |
| PCA piksel (baseline) | 64 | 0.035 | -0.010 | 0.044 | 0.019 | 0.498 |
| SimCLR (deep) | 256 | 0.873 | 0.002 | 0.871 | 0.663 | 0.867 |

- **Invariansi:** SimCLR jauh lebih tahan terhadap rotasi, zoom, gamma, blur, dan noise. Citra yang sama setelah didistorsi tetap mirip (cosine 0.873), dan 66.3% versi terdistorsi bisa menemukan citra aslinya di antara 1.000 kandidat. Bandingkan dengan PCA piksel yang hanya 1.9%.
- **Kebocoran mikroskop:** embedding SimCLR bisa menebak mikroskop dengan balanced accuracy 0.867, sedangkan PCA piksel 0.498 (setara tebakan acak). Embedding SimCLR masih membawa informasi mikroskop, kemungkinan dari perbedaan ketajaman dan skala tekstur yang tidak sepenuhnya dinetralkan oleh augmentasi. Ada dua kemungkinan penjelasan yang belum bisa dibedakan: (1) ini efek alat (batch effect), atau (2) proporsi jenis material memang berbeda antara M1 dan M2. Kemungkinan pertama perlu diuji di notebook gabungan lewat Cramér's V antara cluster dan mikroskop.

**Reproduksi:** embedding 256 sampel pertama dihitung ulang dari `image_encoder_state_dict.pt` dengan selisih absolut maksimum 1.22 × 10⁻⁴ terhadap embedding tersimpan. Selisih kecil ini berasal dari pembulatan mixed precision (float16) di GPU, bukan dari perbedaan model.

## Output untuk notebook gabungan

Bagian 6 notebook menyimpan artefak di `MyDrive/UTS_DL/hasil_tugas3/image_encoder_artifacts/` beserta zip-nya. Formatnya disamakan dengan artefak spectrum encoder:

| Spectrum encoder | Image encoder |
| --- | --- |
| `spectrum_embeddings.csv` | `image_embeddings.csv.gz` |
| `spectrum_embeddings.npy` | `image_embeddings.npy` |
| `spectrum_sample_ids.npy` | `image_sample_ids.npy` |
| `spectrum_autoencoder_state_dict.pt` | `image_encoder_state_dict.pt` |
| `spectrum_encoder_config.json` | `image_encoder_config.json` |

Kolom CSV: `indeks`, `id_sampel`, `mikroskop`, `image_emb_000` sampai `image_emb_255`. Penggabungan dengan spectrum embedding memakai `id_sampel` sebagai kunci, bukan urutan baris.

## Reproducibility

| Item | Nilai |
| --- | --- |
| Seed | 42 (sama dengan notebook spectrum encoder) |
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| NumPy | 2.4.6 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| GPU | Tesla T4 (Google Colab) |

cuDNN diset deterministic. Hasil training ulang bisa berbeda sangat kecil karena operasi atomik GPU, sehingga angka di laporan diambil dari `simclr_e100_b512.pt`.

## Batasan

1. Loss kontrastif yang rendah dan retrieval yang tinggi menunjukkan embedding tahan terhadap distorsi, bukan jaminan bahwa cluster material yang terbentuk akan baik.
2. Mikroskop hanya dipakai untuk stratifikasi split dan uji kebocoran, bukan sebagai label training.
3. Embedding masih membawa informasi mikroskop (balanced accuracy 0.867). Dampaknya terhadap clustering harus dicek di notebook gabungan.
4. Notebook ini belum membuat klaim tentang jumlah cluster atau kualitas cluster; keduanya diuji di notebook gabungan setelah digabung dengan spectrum embedding.
