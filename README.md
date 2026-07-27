# PredictKCMc — HELP-DKT dengan Encoder Modular (LSTM / Transformer)

Model *knowledge tracing* untuk mahasiswa pemrograman: dari urutan submission kode seorang mahasiswa, model memprediksi apakah percobaan (*attempt*) berikutnya akan benar, **sekaligus** melaporkan tingkat penguasaan per konsep pemrograman (variabel, string, list, dsb). Ditujukan untuk peneliti/pengembang *learning analytics* yang ingin mereplikasi atau melanjutkan eksperimen HELP-DKT dengan encoder yang bisa ditukar.

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2Bcu126-ee4c2c)
![Task](https://img.shields.io/badge/task-Knowledge%20Tracing-success)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

> Implementasi ini mengadaptasi [HELP-DKT (liangyubuaa)](https://github.com/liangyubuaa/HELP-DKT). Perbedaan utama: RNN + decoder yang menyatu di model official dipecah menjadi komponen modular, sehingga encoder bisa diganti (`LSTM` ↔ `TransformerCausal`) tanpa menyentuh *training loop*.

---

## 1. Core Tech Stack & Architecture Overview

### Tech Stack

| Komponen | Teknologi | Versi terpasang |
|---|---|---|
| Bahasa | Python | 3.12 (Windows) |
| Deep learning | PyTorch (build CUDA 12.6) | `2.12.1+cu126` |
| Metrik & evaluasi | scikit-learn | `1.7.1` |
| Komputasi numerik | NumPy / SciPy | `2.3.1` / `1.15.2` |
| Analisis & visualisasi | pandas, Matplotlib | `2.3.1` / `3.10.3` |
| Notebook analisis | Jupyter Lab | `4.3.5` |

Tidak ada database, API server, maupun Docker — proyek ini murni *offline batch pipeline* berbasis file CSV.

### Alur Data (Pipeline)

```
┌──────────────────────────────┐
│ Program_Vector_Embeddings.CSV│  9.119 submission, embedding 10-dim per submission
│ problemQmatrix.CSV           │  9 soal x 10 knowledge concept (0/1)
│ P-matrix-out.CSV             │  Q-matrix personal per attempt
└──────────────┬───────────────┘
               │  GetData.py  →  processData.py      (opsional: -regenData True)
               ▼
        train.CSV / test.CSV                          468 / 120 mahasiswa
               │
               │  data.load_data()
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ RunModel.RunEpoch()  — merakit tensor per batch                  │
│   input_data [B,T,20]  QmatrixInput [B,T,T]                      │
│   AbilityMask / Sub / Prod  [B,T,10]                             │
└──────────────┬───────────────────────────────────────────────────┘
               ▼
      ┌────────────────────────── HelpDKT_Model ──────────────────────────┐
      │                                                                   │
      │   ENCODER  (pilih salah satu, API-nya identik)                    │
      │   ┌─────────────────────┐   atau   ┌─────────────────────────┐    │
      │   │ LSTMEncoder         │          │ Transformer_Encoder_    │    │
      │   │ 3 layer, h=200      │          │ Causal (8 head, mask)   │    │
      │   │ input [B,T,20]      │          │ input [B,T,30]          │    │
      │   └──────────┬──────────┘          └───────────┬─────────────┘    │
      │              └────────────┬────────────────────┘                  │
      │                           ▼  h_t  [B,T,200]                       │
      │   DECODER  PredictKC (decoder/Decoder.py)                         │
      │     Eq.7  s_t = sigmoid(W·h_t)          → ability  [B,T,10]       │
      │     Eq.8  y_t = Π sigmoid(α·(s_t−θ))    → p(benar) [B,T,1]        │
      └───────────────────────────┬───────────────────────────────────────┘
                                  ▼
        BCELoss + Adam  ─────►  model.pth  ─────►  inference.py
                                  │                     │
                                  ▼                     ▼
                          result.txt (AUC/F1)   inference_*.CSV + kc_trajectory.png
```

**Inti model dalam satu kalimat:** LSTM/Transformer merangkum riwayat pengerjaan menjadi *hidden state*, decoder mengubahnya menjadi probabilitas penguasaan tiap konsep, lalu mengalikan probabilitas konsep-konsep yang dipakai soal — jadi satu konsep yang lemah sudah cukup menurunkan prediksi keberhasilan.

**Perbedaan penting kedua encoder** (lihat `RunModel.py`, blok "RAKIT INPUT FINAL"):

| | LSTM (jalur official) | TransformerCausal |
|---|---|---|
| Input model | `matmul(QmatrixInput^T, input_data)` → `[B,T,20]`, sumbu waktu diagregasi jadi sumbu konsep | `concat(input_data, P-matrix)` → `[B,T,30]`, urutan waktu dipertahankan |
| Hidden state | *stateful* `(h, c)`, dibawa antar batch | *stateless*, `None` |
| Learning rate | `0.05`, eps `0.1`, grad clip `20` | `5e-4`, eps `1e-8`, grad clip `1.0`, warmup `200` step |

Hyperparameter Transformer sengaja dipisah: nilai official yang di-*tuning* untuk LSTM terlalu agresif untuk *self-attention* dan membuat training cepat jenuh.

---

## 2. Prerequisites & Environment Setup

### System Requirements

| Kebutuhan | Spesifikasi |
|---|---|
| OS | Windows 10/11 (teruji). Linux/macOS bisa jalan, tapi lihat catatan *path* di bawah |
| Python | **3.12** ¹ (minimal 3.11 — `numpy==2.3.1` tidak mendukung 3.10 ke bawah) |
| PyTorch | 2.12.x. Build `+cu126` untuk GPU, atau build CPU biasa |
| RAM | Minimum 4 GB, disarankan 8 GB |
| Disk | ~3 GB (PyTorch + CUDA runtime), ~15 MB untuk data & checkpoint |

¹ Versi Python disimpulkan dari artefak `__pycache__/*.cpython-312.pyc` dan batas minimum `numpy==2.3.1`; `pip freeze` tidak mencatat versi interpreter. Ada juga sisa `.cpython-310.pyc` dari eksekusi lama — abaikan, karena `numpy 2.3.1` tidak bisa jalan di 3.10. Konfirmasi dengan `python -V` dan perbarui baris ini bila perlu.

### GPU (Opsional)

Training berjalan **penuh di CPU** jika GPU tidak tersedia — `RunModel.py` memilih device otomatis dan mencetak `RUNS ON CPU!!!` / `RUNS ON CUDA!!!` saat start. Dataset kecil (588 mahasiswa, ~9 ribu attempt), jadi CPU masih wajar.

- Butuh GPU: **NVIDIA driver yang mendukung CUDA 12.6** (paket `torch==2.12.1+cu126` sudah membawa runtime CUDA-nya, CUDA Toolkit terpisah tidak perlu diinstall).
- VRAM: ~2 GB sudah cukup (`hidden_size=200`, `batch_size=32`).

### External Dependencies

Tidak ada. Tanpa Docker, tanpa database, tanpa service eksternal.

### Catatan Path (penting untuk non-Windows)

`GetData.py` dan `processData.py` menyusun path dengan pola `dirPath + '/PREDICTKCMC/data/ModelInput/...'` yang mengandalkan nama folder induk tertentu dan **case-insensitive** ala Windows. Dua script ini hanya dipakai saat `-regenData True`; jika `train.CSV`/`test.CSV` sudah ada (kondisi default repo ini), keduanya tidak pernah dijalankan. `RunModel.py` dan `inference.py` sendiri sudah memakai `os.path.join` yang portabel.

---

## 3. Quick Start / Installation Guide

```powershell
# 1. Masuk ke folder proyek
cd "D:\Student Model\PredictKCMc"

# 2. Buat & aktifkan virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependensi (GPU / CUDA 12.6)
pip install -r requirements.txt `
  --index-url https://download.pytorch.org/whl/cu126 `
  --extra-index-url https://pypi.org/simple

# 4. Cek instalasi
python -c "import torch, sklearn; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

> **Kenapa butuh dua index?** `requirements.txt` memuat `torch==2.12.1+cu126`. Sufiks `+cu126` adalah *local version* yang hanya tersedia di indeks PyTorch, bukan di PyPI — tanpa `--index-url` di atas, `pip install -r requirements.txt` akan gagal dengan `No matching distribution found for torch==2.12.1+cu126`.

**Alternatif CPU-only** (tanpa GPU): hapus tiga baris `torch`, `torchaudio`, `torchvision` dari `requirements.txt`, lalu:

```powershell
pip install -r requirements.txt
pip install torch==2.12.1
```

**Linux/macOS:** ganti langkah 2 dengan `python3 -m venv .venv && source .venv/bin/activate`, dan tulis perintah langkah 3 dalam satu baris (ganti backtick PowerShell dengan `\` atau hapus saja pemisah barisnya).

> `requirements.txt` adalah hasil `pip freeze` dari environment penuh, jadi memuat 189 paket — sebagian besar tidak dipakai proyek ini (Jupyter, Streamlit, Flask, FastAPI, dll). Yang benar-benar dibutuhkan untuk training & inference hanya: **torch, scikit-learn, numpy, scipy**. Tambahan **pandas + matplotlib** hanya untuk visualisasi trajektori. Kalau ingin environment ramping, install lima paket itu saja.

### Verifikasi cepat (tanpa training, < 30 detik)

```powershell
python Test_equivalence.py             # model modular == model official, bit-per-bit
python decoder\Test_decoder.py         # decoder: shape, rentang probabilitas, masking
python Encoder\Test_encoder_clean.py   # encoder LSTM & Transformer: shape output
python _verify_no_code_change.py       # bukti komentar tidak mengubah code (AST + bytecode)
```

---

## 4. Configuration & Environment Variables

Proyek ini **tidak memakai file `.env`** dan tidak membaca satu pun variabel lingkungan. Seluruh konfigurasi lewat argumen CLI `argparse` di `RunModel.py` / `inference.py`.

> **Peringatan keamanan:** tidak ada API key, kredensial, atau *secret* apa pun di repo ini — dan memang jangan pernah menambahkannya ke README, `requirements.txt`, atau file CSV. Kalau nanti pipeline diperluas (misalnya kirim hasil ke dashboard), simpan kredensial di `.env` dan tambahkan `.env` ke `.gitignore`.

### Argumen Utama `RunModel.py`

| Argumen | Default | Wajib? | Fungsi |
|---|---|---|---|
| `-encoder` | `LSTM` | Opsional | `LSTM` (replikasi official) atau `TransformerCausal` |
| `-epochs` | `500` | Opsional | Jumlah epoch training |
| `-batch_size` | `32` | Opsional | Ukuran batch. Sisa mahasiswa yang tidak cukup satu batch penuh **dibuang** |
| `-hidden_size` | `200` | Opsional | Dimensi hidden state. Harus habis dibagi `-num_heads` untuk Transformer |
| `-hidden_layer_num` | `3` | Opsional | Jumlah layer encoder |
| `-Qmatrix_size` | `10` | Opsional | Jumlah *knowledge concept* |
| `-input_size` | `20` | Opsional | 10 dim "salah" + 10 dim "benar" (Eq. 4 paper) |
| `-evaluation_interval` | `5` | Opsional | Evaluasi test set tiap N epoch |
| `-taskModel` | `taskC` | Opsional | Jalur aktif proyek ini. `taskA`/`taskB` ada demi kesetaraan dengan code official |
| `-regenData` | `False` | Opsional | `True` = bangun ulang `train/test.CSV` via `GetData.py` + `processData.py` |
| `-model_path` | `data/ModelOutput/model.pth` | Opsional | Lokasi simpan checkpoint |

**Khusus `-encoder TransformerCausal`** (tidak berpengaruh sama sekali ke LSTM):

| Argumen | Default | Fungsi |
|---|---|---|
| `-num_heads` | `8` | Jumlah attention head |
| `-transformer_learning_rate` | `5e-4` | LR terpisah, jauh lebih kecil dari LR LSTM |
| `-transformer_epsilon` | `1e-8` | Epsilon Adam standar |
| `-transformer_max_grad_norm` | `1.0` | Grad clipping ketat |
| `-transformer_warmup_steps` | `200` | Linear LR warmup, `0` untuk menonaktifkan |
| `-transformerSeqInput` | `True` | `True` = skema input baru (urutan waktu dipertahankan). **Nilainya harus sama antara training dan inference**, kalau tidak `load_state_dict` gagal |

### Argumen Utama `inference.py`

| Argumen | Default | Fungsi |
|---|---|---|
| `-source` | `test` | `test` = mahasiswa asli dari `test.CSV`; `dummy` = data sintetis berpola sama |
| `-model_path` | `data/ModelOutput/model_transformer.pth` | Checkpoint yang dimuat |
| `-encoder` | `auto` | `auto` = dideteksi dari isi `state_dict` |
| `-num_test_students` | `1` | Jumlah mahasiswa (diurutkan dari yang attempt-nya terpanjang) |
| `-output_path` | `data/ModelOutput/inference_dummy_result.CSV` | Lokasi CSV hasil |
| `-seed` | `1` | Seed untuk reproduktibilitas |

⚠️ Hyperparameter arsitektur di `inference.py` (`-hidden_size`, `-hidden_layer_num`, `-Qmatrix_size`, `-transformerSeqInput`) **wajib sama persis dengan saat training**. Checkpoint `.pth` hanya berisi bobot, bukan arsitektur.

---

## 5. Usage / Basic Example

### Training

```powershell
# Replikasi jalur official (LSTM)
python RunModel.py -encoder LSTM -epochs 100 -model_path data\ModelOutput\model_LSTM.pth

# Varian Transformer dengan causal masking
python RunModel.py -encoder TransformerCausal -epochs 30 -model_path data\ModelOutput\model_transformer.pth
```

Expected output:

```
+-+-+-+-+-+-+-+-+-+
|run which model: taskC
|encoder: TransformerCausal
|epoch: 30
|learning rate: 0.0005 (eps: 1e-08  grad_norm: 1.0  warmup_steps: 200  seqInput: True )
|batch size: 32
+-+-+-+-+-+-+-+-+-+
+-+-+-+-+-+-+-+-+-+
| RUNS ON CUDA!!! |
+-+-+-+-+-+-+-+-+-+
the number of rows is 8754
The number of students is  468
...
TEST result:
        epoch:30
        encoder:TransformerCausal
        auc:0.983057077832876
        acc:0.9536231884057971
        ...
```

Hasil training tersimpan di:

- `data/ModelOutput/model*.pth` — bobot model (**epoch terakhir**, bukan epoch ber-AUC terbaik)
- `data/ModelOutput/result.txt` — ringkasan metrik terbaik (mode *append*, riwayat lama tidak terhapus)
- `data/ModelOutput/studentAbilityMess-taskC-<encoder>.CSV` — matriks ability per attempt

Hasil rujukan yang sudah ada di repo (`result.txt`): **AUC 0.983 · Acc 0.954 · F1 0.900** pada `TransformerCausal`, epoch 30.

### Inference

```powershell
# Mahasiswa dengan attempt terpanjang di test.CSV (default: 1 mahasiswa)
python inference.py -source test -model_path data\ModelOutput\model_transformer.pth

# Beberapa mahasiswa sekaligus
python inference.py -source test -num_test_students 3

# Atau pakai data dummy sintetis
python inference.py -source dummy -num_students 8 -seed 7
```

Expected output (hasil nyata yang tersimpan di `data/ModelOutput/inference_dummy_result.CSV`):

```
Data test  : 1 siswa test.CSV (id: ['46724'], attempt: [57]) dari 120 siswa  ->  ...\test.CSV
Model      : ...\model_transformer.pth  (encoder: TransformerCausal, 1,456,410 params)
Prediksi   : 56 attempt, akurasi vs label asli: 1.000
Hasil      : ...\inference_dummy_result.CSV

Contoh hasil (siswa pertama):
  attempt                next   p(benar)  ability KC terlibat
  b_362_46724_1.py       1      0.9307    ST=0.99, VA=0.99, OP=0.99, EX=0.99, IO=0.97, LI=0.47, CO=0.99, TU=0.17, DI=0.39, CD=0.32
  c_362_46724_2.py       0      0.2016    ST=0.90, VA=0.93, OP=0.94, EX=0.86, IO=0.38, LI=0.70, CO=0.93, TU=0.01, DI=0.20, CD=0.34
```

Dua catatan tentang output di atas:

- Akurasi `1.000` itu untuk **satu** mahasiswa (56 prediksi), bukan seluruh test set — jangan dibaca sebagai performa model. Angka test set penuh ada di `result.txt`.
- Label kolom tertulis "ability KC terlibat", tapi yang tercetak adalah **seluruh 10 konsep**. Nilai `ability` yang dilaporkan diambil sebelum *masking* per-konsep (lihat `decoder/Decoder.py`); yang dinolkan hanya baris *padding*, bukan konsep yang tidak dipakai soal.

Kolom CSV hasil: `student, step, attempt_file, next_file, actual_next, pred_prob, pred_label, ability_ST … ability_CD`.

Jumlah parameter terlatih: **822.810** (LSTM) dan **1.456.410** (TransformerCausal).

### Visualisasi Trajektori Konsep

```powershell
cd data\ModelOutput
python kc_trajectory.py        # menghasilkan kc_trajectory.png
```

Analisis lebih dalam tersedia di notebook `data/ModelOutput/analyze_inference.ipynb`.

---

## 6. Directory Structure

```
PredictKCMc/
├── RunModel.py                  # Training loop — titik masuk utama
├── inference.py                 # Inference dengan checkpoint tersimpan
├── HelpDKT_Model.py             # Model utama: perakit encoder + decoder
├── HELP_DKT_Model.py            # Model official (LSTM menyatu) — acuan pembanding
├── data.py                      # Pembaca train/test.CSV
├── GetData.py                   # Data mentah  → train/test.CSV   (-regenData True)
├── processData.py               # Pelabelan ulang lanjutan        (-regenData True)
├── Test_equivalence.py          # Uji: model modular == model official
├── _verify_no_code_change.py    # Bukti komentar tidak mengubah code (AST + bytecode)
├── _backup_original/            # Salinan file sebelum diberi komentar (aman dihapus)
├── requirements.txt             # pip freeze environment
│
├── Encoder/
│   ├── LSTM_Encoder.py          # Encoder LSTM/GRU (jalur official)
│   ├── Transformer_Encoder.py   # PositionalEncoding + Transformer + varian Causal
│   └── Test_encoder_clean.py    # Uji shape kedua encoder
│
├── decoder/
│   ├── Decoder.py               # PredictKC — Eq. (7)-(8) paper, inti matematis
│   └── Test_decoder.py          # Uji decoder
│
└── data/
    ├── ModelInput/              # Program_Vector_Embeddings, Q-matrix, P-matrix, train/test
    └── ModelOutput/             # Checkpoint .pth, result.txt, CSV ability, notebook, plot
```

---

## 7. Catatan untuk Pengembang Berikutnya

Seluruh file di atas sudah diberi komentar baris-per-baris beserta anotasi bentuk tensor (`[B, T, H]`). Beberapa hal yang perlu diketahui sebelum mengubah apa pun:

1. **Flag `masked` / `subQmatrix` / `multiLinearLayers` bertipe *string* `'True'`/`'False'`, bukan boolean.** Menulis `if self.masked:` akan selalu bernilai benar (string `'False'` pun *truthy*) dan diam-diam mengubah perilaku model. Selalu bandingkan dengan `== 'True'`.
2. **`torch.save` menyimpan bobot epoch terakhir**, sementara metrik yang dilaporkan adalah epoch ber-AUC terbaik. Untuk *best-checkpoint*, tambahkan `torch.save` di dalam blok `if auc > maxAuc:` pada `RunModel.main()`.
3. **"MODE PAPER" di `decoder/Decoder.py` belum diimplementasikan.** Jika `problemQmatrixSub`/`problemQmatrixProd` dikirim `None`, variabel `decoded` tidak pernah dibuat → `UnboundLocalError`. Jalur normal (`RunModel.py`, `inference.py`) selalu mengirim keduanya, jadi tidak pernah terpicu.
4. **Nilai balik `RunEpoch` yang pertama bukan RMSE** meski variabelnya bernama `rmse`; yang dikembalikan adalah rata-rata loss per batch, dan itulah yang tercetak sebagai `loss:` maupun tersimpan sebagai `maxRmse` di `result.txt`.
5. **Skema input Transformer terikat ke checkpoint.** Melatih dengan `-transformerSeqInput True` lalu inference dengan `False` (atau sebaliknya) membuat lebar `input_projection` berbeda dan `load_state_dict` gagal.
6. `_backup_original/` berisi salinan seluruh file sebelum pemberian komentar; `python _verify_no_code_change.py` membandingkan AST **dan** bytecode keduanya. Folder ini aman dihapus setelah tidak diperlukan.

---

## Referensi

- Liang, Y. dkk. (2022). [**HELP-DKT: an interpretable cognitive model of how students learn programming based on deep knowledge tracing**](https://www.nature.com/articles/s41598-022-07956-0). *Scientific Reports*, 12, 4012.
- Repositori official: <https://github.com/liangyubuaa/HELP-DKT>
- Vaswani, A. dkk. (2017). *Attention Is All You Need* — dasar `Transformer_Encoder.py` (positional encoding sinusoidal, skala `sqrt(d_model)`, FFN 4×).
#   S t u d e n t - M o d e l  
 