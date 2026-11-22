# Task 3: Multimodal Clustering — Material Types

Folder ini berisi Task 3: mengelompokkan 36.000 sampel material tanpa label berdasarkan citra mikrostruktur dan spektrum komposisi. Pekerjaannya dibagi menjadi dua cabang representasi dan satu notebook gabungan:

| Bagian | Lokasi | Isi |
| --- | --- | --- |
| Image representation | `image_representation/` | Image encoder SimCLR, eksplorasi distorsi citra, ekspor image embedding |
| Spectrum representation | `spectrum_representation/` | 1D-CNN autoencoder, EDA spektrum, ekspor spectrum embedding |
| Gabungan | `Multimodal_Clustering_Material_Types.ipynb` | Distorsi spektrum, koreksi mikroskop, fusion, jumlah cluster, KMeans/GMM/DEC, analisis, `cluster.csv` |

README ini menjelaskan notebook gabungan. Detail setiap cabang ada di README masing-masing folder.

## Isi folder

| File | Keterangan |
| --- | --- |
| `Multimodal_Clustering_Material_Types.ipynb` | Notebook gabungan, lengkap dengan output hasil run |
| `ringkasan_clustering.json` | K_FINAL, k terbaik per kriteria, metrik ketiga metode, hasil satu modalitas, koreksi mikroskop, dan seed |
| `pemilihan_k.csv` | Inertia, silhouette, Calinski-Harabasz, Davies-Bouldin, BIC, dan stabilitas untuk k = 2 sampai 20 |
| `image_representation/` | Notebook, bobot, dan README image encoder |
| `spectrum_representation/` | Notebook, bobot, dan artefak spectrum encoder |
| `README.md` | Dokumen ini |

File prediksi `cluster.csv` disimpan di root repository sesuai format pengumpulan.

## Data dan input

| File | Sumber | Isi yang dipakai |
| --- | --- | --- |
| `clustering.npz` | Data UTS | `gambar` (36000, 64, 64) dan `spektrum` (36000, 128), hanya untuk eksplorasi dan menafsirkan cluster |
| `clustering_meta.csv` | Data UTS | `indeks`, `id_sampel`, `mikroskop` |
| `image_encoder_artifacts.zip` | Notebook image | `image_embeddings.csv.gz`, 36000 × 256 |
| `spectrum_encoder_artifacts.zip` | Notebook spectrum | `spectrum_embeddings.csv`, 36000 × 32 |

Kedua embedding digabung berdasarkan `id_sampel`, bukan urutan baris, dengan pemeriksaan bahwa setiap sampel punya kedua embedding dan kolom mikroskop di ketiga file sama.

## Cara menjalankan

1. Jalankan notebook di `image_representation/`. Hasilnya `MyDrive/UTS_DL/hasil_tugas3/image_encoder_artifacts.zip`.
2. Jalankan notebook di `spectrum_representation/` dengan output diarahkan ke `MyDrive/UTS_DL/`, atau taruh `spectrum_encoder_artifacts.zip` di folder tersebut.
3. Pastikan `clustering.npz` dan `clustering_meta.csv` ada di `MyDrive/UTS_DL/` (nilai default `DATA_DIR`).
4. Buka `Multimodal_Clustering_Material_Types.ipynb` di Google Colab, pilih **Runtime → Change runtime type → T4 GPU**, lalu **Runtime → Run all**.

Notebook mencari artefak secara otomatis di `MyDrive/UTS_DL/` dan `MyDrive/UTS_DL/hasil_tugas3/`, lalu mengekstrak zip-nya sendiri. Satu kali run memakan sekitar 8 menit di T4 (pemilihan k 257 detik, DEC 67 detik, UMAP 88 detik). Untuk uji cepat bahwa kode berjalan, ubah `SMOKE = True` (400 sampel; angkanya tidak bermakna).

## Metode

**Eksplorasi distorsi spektrum.** Shift diukur dari posisi puncak tertinggi, baseline dari offset dan kemiringan ujung spektrum, dan spike dari titik yang menyimpang lebih dari 6 kali simpangan robust median filter lokal sekaligus lebih dari 10% rentang spektrum. Distorsi citra diperiksa di notebook image.

**Koreksi efek mikroskop.** Setiap embedding diuji apakah bisa menebak mikroskop (balanced accuracy regresi logistik, 5-fold). Spektrum dipakai sebagai pembanding karena tidak diukur dengan mikroskop. Jika spektrum tidak bisa menebak mikroskop tetapi image embedding bisa, informasi itu dianggap efek alat, dan image embedding distandardisasi per mikroskop.

**Fusion.** Setiap modalitas distandardisasi, direduksi dengan PCA menjadi 16 komponen, lalu diskala agar total variansnya sama dengan 1, sehingga kedua modalitas berpengaruh setara terhadap jarak. Keduanya kemudian digabung menjadi ruang fused 32 dimensi (bobot 1:1).

**Jumlah cluster.** KMeans untuk k = 2 sampai 20 dinilai dengan empat kriteria yang masing-masing memberi satu suara: silhouette, Calinski-Harabasz, Davies-Bouldin, dan BIC dari Gaussian Mixture. Elbow dan stabilitas antar seed ditampilkan sebagai pendukung. Jika suara seri, dipakai pilihan silhouette.

**Clustering.** Tiga metode dijalankan dengan tiga seed masing-masing:

| Metode | Keterangan |
| --- | --- |
| KMeans | n_init 20 |
| Gaussian Mixture | Kovarians penuh, reg_covar 1e-4 |
| DEC (Xie dkk., 2016) | Autoencoder MLP 32 → 256 → 128 → 10, inisialisasi KMeans di ruang laten, lalu optimisasi KL(P‖Q) dengan soft assignment Student-t sampai kurang dari 0.1% sampel berpindah cluster |

Metode final dipilih dari rata-rata peringkat silhouette, Davies-Bouldin, dan stabilitas.

**Analisis satu modalitas.** KMeans dijalankan di ruang image saja dan spectrum saja lalu dibandingkan dengan hasil final (ARI). Selain itu dihitung kNN purity per cluster di setiap ruang, yaitu persentase 10 tetangga terdekat yang berada di cluster yang sama.

**Analisis cluster.** UMAP pada 6.000 sampel, percampuran kNN antar pasangan cluster, silhouette per cluster, proporsi mikroskop per cluster, galeri citra, dan rata-rata spektrum per cluster.

## Hasil

**Distorsi spektrum:**

| Mikroskop | Posisi puncak (mean) | Baseline offset | Baseline slope (std) | Spike per sampel | Sampel dengan spike |
| --- | --- | --- | --- | --- | --- |
| M1 | 51.1 | 0.31 | 0.399 | 0.995 | 58.8% |
| M2 | 50.8 | 0.31 | 0.400 | 1.007 | 59.0% |

Posisi puncak berkumpul di beberapa posisi diskret, baseline offset tersebar merata dari 0 sampai 0.6, dan sekitar 59% sampel punya spike. Distribusi M1 dan M2 identik, sesuai harapan karena spektrum tidak diukur dengan mikroskop.

**Koreksi mikroskop:**

| Embedding | Kebocoran awal | Dikoreksi | Kebocoran akhir |
| --- | --- | --- | --- |
| Image | 0.8670 | Ya | 0.4980 |
| Spectrum | 0.4991 | Tidak | 0.4991 |

PCA 16 komponen mempertahankan 88.1% varians image dan 86.7% varians spectrum.

**Jumlah cluster:**

| Kriteria | k terbaik |
| --- | --- |
| Silhouette | 6 (0.251) |
| Calinski-Harabasz | 4 |
| Davies-Bouldin | 5 (1.577; k=6: 1.605) |
| BIC (GMM) | 15 |

Setiap kriteria memberi suara berbeda, sehingga aturan seri memilih **k = 6** dari silhouette. Pilihan ini didukung oleh silhouette yang turun tajam setelah k=6 (0.207 di k=7, 0.175 di k=8), Davies-Bouldin yang terendah di k=5–6, dan UMAP yang menunjukkan enam pulau terpisah. BIC terus turun sampai k=15 karena GMM kovarians penuh menutup setiap cluster yang tidak berbentuk Gaussian dengan beberapa komponen.

**Perbandingan metode (k = 6):**

| Metode | Silhouette | Davies-Bouldin | Stabilitas | Cramér's V mikroskop |
| --- | --- | --- | --- | --- |
| KMeans | 0.2514 | 1.6047 | 1.0000 | 0.0096 |
| GMM | 0.2514 | 1.6048 | 0.8586 | 0.0096 |
| DEC | 0.2514 | 1.6056 | 0.9990 | 0.0096 |

ARI antar metode 0.999–1.000. DEC berhenti setelah dua epoch karena hanya 0.12% lalu 0.07% sampel berpindah cluster: cluster di ruang fused sudah terpisah jelas sehingga hampir tidak ada yang perlu dipertajam. Metode final adalah **KMeans**. Cramér's V 0.0096 menunjukkan cluster tidak mengikuti mikroskop.

**Apa yang hilang jika hanya satu modalitas:**

| Ruang | ARI vs cluster final | Silhouette cluster final di ruang ini |
| --- | --- | --- |
| Image saja | 0.9995 | 0.378 |
| Spectrum saja | 0.4589 | 0.078 |
| Fused | 1.0000 | 0.251 |

| Cluster | Purity image | Purity spectrum | Purity fused |
| --- | --- | --- | --- |
| c0 | 1.000 | 0.981 | 0.999 |
| c1 | 0.999 | 0.554 | 0.997 |
| c2 | 1.000 | 0.425 | 1.000 |
| c3 | 1.000 | 0.980 | 1.000 |
| c4 | 1.000 | 0.481 | 0.991 |
| c5 | 1.000 | 0.484 | 0.997 |

Image saja sudah memisahkan keenam cluster. Spectrum saja menggabungkan c1 dengan c2 dan c4 dengan c5, karena kedua pasangan itu memiliki spektrum yang praktis identik dan hanya berbeda mikrostruktur. Di data ini spektrum tidak menambah pemisahan baru. Perannya adalah konfirmasi untuk c0 dan c3, serta pembanding yang membuktikan bahwa informasi mikroskop di image embedding adalah efek alat.

**Isi cluster:**

| Cluster | Ukuran | Mikrostruktur | Silhouette | Proporsi M2 |
| --- | --- | --- | --- | --- |
| c0 | 10.045 | Jaringan butir halus | 0.252 | 0.400 |
| c1 | 4.977 | Sel atau busa dengan dinding gelap | 0.233 | 0.402 |
| c2 | 3.656 | Bintik gelap di atas matriks terang (dua fase) | 0.341 | 0.407 |
| c3 | 8.685 | Lamela diagonal sangat halus | 0.262 | 0.391 |
| c4 | 4.339 | Lamela lurus kasar | 0.204 | 0.399 |
| c5 | 4.298 | Lamela bergelombang | 0.216 | 0.400 |

**Cluster yang sulit dipisahkan:** pasangan tersulit adalah c4 vs c5, dengan percampuran kNN 0.47% dan silhouette terendah. Spektrum keduanya identik, dan citranya sama-sama berlamela, hanya berbeda lurus atau bergelombang. Augmentasi rotasi bebas membuat orientasi tidak berpengaruh, sehingga pembedanya hanya kelengkungan lamela. Pasangan berikutnya c0 vs c1 (0.13%), sama-sama bertekstur jaringan atau sel yang halus.

Percampuran kNN di semua pasangan di bawah 0.5% meskipun silhouette hanya sekitar 0.25. Silhouette membandingkan jarak rata-rata ke seluruh anggota cluster, sehingga nilainya sedang karena cluster cukup lebar. Percampuran kNN hanya melihat tetangga terdekat, sehingga nilainya kecil karena di antara cluster ada celah kosong. Artinya pemisahan antar cluster bersih meskipun tiap cluster tidak padat.

**Reproduksi:** menjalankan ulang KMeans dengan seed pertama menghasilkan ARI 1.0000 terhadap `cluster.csv`.

## Output

`cluster.csv` berisi 36.000 baris dengan kolom `id_sampel` dan `cluster` (integer 0–5), sesuai urutan `clustering_meta.csv`.

```
id_sampel,cluster
S966119,4
S315633,3
S805122,0
```

| Cluster | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Jumlah sampel | 10.045 | 4.977 | 3.656 | 8.685 | 4.339 | 4.298 |

## Reproducibility

| Item | Nilai |
| --- | --- |
| Seed | 42 (fusion dan subsampel), 0 / 1 / 2 (seed clustering) |
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| NumPy | 2.4.6 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| umap-learn | 0.5.12 |
| GPU | Tesla T4 (Google Colab) |

KMeans dan GMM deterministik untuk seed yang sama. Bobot DEC disimpan sebagai cache di `hasil_tugas3/`, sehingga run berikutnya memuat hasil yang sama.

## Batasan

1. Label material disembunyikan, sehingga jumlah cluster dan metode final hanya dipilih dengan kriteria internal. Skor ARI terhadap label asli baru diketahui dari penilaian.
2. Keempat kriteria jumlah cluster tidak sepakat. k = 6 dipilih lewat aturan seri silhouette, didukung oleh bentuk kurva dan UMAP, bukan oleh kesepakatan kriteria.
3. Augmentasi rotasi dan zoom pada image encoder membuat orientasi dan skala diabaikan. Jika ada dua material yang hanya berbeda ukuran butir atau orientasi, keduanya bisa tergabung. Kandidatnya c0 dan c3, yang ukurannya sekitar dua kali cluster lain.
4. Spectrum embedding dilatih tanpa koreksi baseline, spike, atau augmentasi, sehingga ikut menyimpan distorsi tersebut. Ini menjelaskan sebagian lemahnya pemisahan di ruang spectrum saja.
5. Koreksi mikroskop mengandalkan asumsi bahwa spektrum tidak dipengaruhi mikroskop. Asumsi ini didukung oleh distribusi spektrum M1 dan M2 yang identik.
