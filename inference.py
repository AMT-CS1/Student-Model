"""
inference.py -- inference HELP-DKT (taskC) memakai model tersimpan
(data/ModelOutput/model.pth) dengan DATA DUMMY yang polanya meniru
train.CSV/test.CSV asli.

Pola data asli yang ditiru:
    - Blok per siswa: [jumlah attempt], [indeks soal], [correctness 0/1],
      lalu 1 baris per attempt: nama file + 10 dim embedding program.
    - Nama file: {b|c}_{problemID}_{studentID}_{attemptKe}.py
      b = attempt gagal (correctness 0), c = attempt akhir yang benar (1).
      Tiap soal: 0..N attempt 'b' lalu ditutup 1 attempt 'c'.
    - Urutan soal tetap: 362, 371, 406, 417, 449, 472.
    - Embedding ~ Gaussian per-dimensi (mean/std dihitung dari data asli).
    - P-matrix (personalized Q-matrix) per attempt:
      'c' = baris Q-matrix soal penuh; 'b' = subset acak dari KC soal.

Encoder dideteksi otomatis dari state_dict (LSTM / TransformerCausal),
bisa dipaksa lewat -encoder. Jalur pemrosesan batch identik dengan
RunModel.RunEpoch (taskC, mode test) supaya perilaku model sama persis.

Jalankan:
    python inference.py                    # 1 siswa test.CSV terpanjang
    python inference.py -source dummy      # data dummy pola train/test
    python inference.py -num_students 16 -source dummy -seed 7
"""
# ============================================================================
# PETA FILE INI (komentar tambahan untuk pengembang berikutnya).
#
# inference.py = MEMAKAI model yang sudah dilatih (data/ModelOutput/*.pth),
# bukan melatihnya. Alur main():
#   1. Siapkan data  -> -source test  : ambil siswa asli dari test.CSV
#                    -> -source dummy : bangkitkan siswa sintetis
#   2. LoadModel()   -> deteksi encoder dari state_dict, bangun model, load
#   3. RunInference()-> BuildBatch() + forward(test=True) per batch
#   4. WriteResults()-> CSV: prediksi + ability per KC tiap attempt
#
# ATURAN EMAS: BuildBatch() di sini HARUS mencerminkan blok taskB/taskC di
# RunModel.RunEpoch(). Kalau salah satu berubah, yang lain wajib menyusul --
# kalau tidak, angka inference tidak sebanding dengan angka training.
#
# Hyperparameter di argparse bawah ini juga HARUS sama dengan saat training
# (hidden_size, hidden_layer_num, Qmatrix_size, transformerSeqInput), karena
# checkpoint hanya berisi BOBOT, bukan arsitektur.
#
# Notasi shape: B = batch, T = num_steps (57), D = input_size (20), K = 10
# ============================================================================
import argparse
import csv
import os
import random

import torch

from HelpDKT_Model import HelpDKT_Model
# CONCEPT_ORDER dipakai untuk memberi NAMA kolom ability di CSV hasil
# (ST, VA, OP, ...). Urutannya harus sama dengan urutan kolom Q-matrix.
from decoder.Decoder import CONCEPT_ORDER

dirPath = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Konstanta pola data asli
# ---------------------------------------------------------------------------
# Baris problemQmatrix.CSV untuk soal-soal yang muncul di train/test asli
# Q-matrix di-hardcode di sini supaya mode dummy bisa jalan tanpa file CSV.
# Isinya HARUS sama dengan baris di problemQmatrix.CSV untuk 6 soal ini.
PROBLEM_QMATRIX = {
    '362': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    '371': [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    '406': [0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    '417': [0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
    '449': [1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    '472': [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
}
PROBLEM_ORDER = ['362', '371', '406', '417', '449', '472']

# mean/std per dimensi embedding, dihitung dari train.CSV + test.CSV asli
EMB_MEAN = [-0.727, 0.104, -0.620, -0.026, -0.560,
            0.543, -0.926, 0.607, 0.608, -0.941]
EMB_STD = [0.072, 0.033, 0.067, 0.034, 0.058,
           0.058, 0.092, 0.060, 0.063, 0.097]

# Konstanta bentuk data. NUM_STEPS = T; kalau dataset berubah, sesuaikan.
QMATRIX_SIZE = 10
INPUT_SIZE = 20   # Eq.4 paper: 10 dim 'salah' + 10 dim 'benar'
NUM_STEPS = 57    # max jumlah attempt di train/test asli

parser = argparse.ArgumentParser(description='HELP-DKT inference (dummy data)')
parser.add_argument('-source', type=str, default='test',
                    choices=['test', 'dummy'],
                    help="'test' = ambil siswa test.CSV dengan attempt "
                         "terpanjang; 'dummy' = generate data dummy")
parser.add_argument('-num_test_students', type=int, default=1,
                    help='jumlah siswa test.CSV yang diambil (terpanjang dulu)')
parser.add_argument('-test_data_path', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelInput', 'test.CSV'))
parser.add_argument('-QmatrixPath', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelInput',
                                         'P-matrix-out.CSV'))
parser.add_argument('-ProblemQmatrixPath', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelInput',
                                         'problemQmatrix.CSV'))
parser.add_argument('-encoder', type=str, default='auto',
                    choices=['auto'] + list(HelpDKT_Model.ENCODER_TYPES),
                    help="'auto' = deteksi dari model.pth")
parser.add_argument('-model_path', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelOutput', 'model_transformer.pth'))
parser.add_argument('-output_path', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelOutput',
                                         'inference_dummy_result.CSV'))
parser.add_argument('-num_students', type=int, default=8,
                    help='jumlah siswa dummy')
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-num_steps', type=int, default=NUM_STEPS)
parser.add_argument('-seed', type=int, default=1)
# hyperparameter HARUS sama dengan training (default RunModel.py)
parser.add_argument('-input_size', type=int, default=INPUT_SIZE)
parser.add_argument('-hidden_size', type=int, default=200)
parser.add_argument('-hidden_layer_num', type=int, default=3)
parser.add_argument('-Qmatrix_size', type=int, default=QMATRIX_SIZE)
parser.add_argument('-num_heads', type=int, default=8)
parser.add_argument('-transformer_dropout', type=float, default=0.1)
parser.add_argument('-transformerSeqInput', type=str, default='True',
                    help="HARUS sama dengan nilai dipakai saat training "
                         "(RunModel.py -transformerSeqInput) untuk checkpoint "
                         "yang dimuat. 'True' = skema input baru (urutan "
                         "waktu dipertahankan + P-matrix per-step); 'False' "
                         "= skema lama (matmul agregat). Tidak berlaku utk "
                         "-encoder LSTM.")
parser.add_argument('-masked', type=str, default='True')
parser.add_argument('-subQmatrix', type=str, default='True')
parser.add_argument('-multiLinearLayers', type=str, default='True')
parser.add_argument('-taskModel', type=str, default='taskC')
args = parser.parse_args()

# Model & semua tensor input harus berada di device yang sama.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# 1. Generator data dummy (pola == train/test asli)
# ---------------------------------------------------------------------------
def MakeEmbedding():
    """Embedding 10-dim ~ N(mean, std) per dimensi (pola data asli)."""
    # Meniru sebaran embedding asli per dimensi -> data dummy 'terasa' wajar
    # bagi model, bukan angka acak seragam.
    return [random.gauss(EMB_MEAN[d], EMB_STD[d]) for d in range(10)]


def MakeDummyStudents(num_students):
    """
    Bangun daftar siswa berformat SAMA dengan output data.load_data:
    tuple baris CSV -> (['n'], [indeks soal], [correctness], [file, e1..e10], ...)
    Sekaligus bangun P-matrix (Qmatrix) dummy per attempt.

    Returns: (students, Qmatrix) -- Qmatrix: {namaFile: [str x10]}
    """
    students = []
    Qmatrix = {}
    for s in range(num_students):
        studentID = str(90000 + s)  # ID dummy, di luar rentang ID asli
        # Jumlah attempt per soal mengikuti pola nyata: mayoritas cepat benar,
        # sebagian kecil butuh banyak percobaan (weights di random.choices).
        # tiap siswa mengerjakan 2..6 soal pertama secara berurutan
        num_problems = random.randint(2, len(PROBLEM_ORDER))
        problem_idx, correctness, attempt_rows = [], [], []
        for p in range(num_problems):
            problemID = PROBLEM_ORDER[p]
            qrow = PROBLEM_QMATRIX[problemID]
            kc_idx = [k for k in range(QMATRIX_SIZE) if qrow[k] == 1]
            # jumlah attempt per soal: mayoritas cepat benar, ekor panjang
            tries = random.choices([1, 2, 3, 4, 5, 6],
                                   weights=[30, 25, 18, 12, 9, 6])[0]
            for t in range(1, tries + 1):
                # Attempt terakhir tiap soal SELALU 'c' (benar) -> pola sama
                # dengan data asli: 0..N kali gagal ('b') lalu ditutup benar.
                is_last = (t == tries)
                prefix = 'c' if is_last else 'b'
                fname = '%s_%s_%s_%d.py' % (prefix, problemID, studentID, t)
                problem_idx.append(str(p))
                correctness.append('1' if is_last else '0')
                attempt_rows.append(
                    [fname] + ['%.16f' % v for v in MakeEmbedding()])
                # P-matrix: 'c' = Q-row penuh; 'b' = subset acak KC soal
                    # P-matrix (personalized): attempt benar menguasai SEMUA
                    # konsep soal; attempt salah hanya sebagian (subset acak).
                if is_last:
                    prow = list(qrow)
                else:
                    hit = random.sample(kc_idx,
                                        random.randint(1, len(kc_idx) - 1))
                    prow = [1 if k in hit else 0 for k in range(QMATRIX_SIZE)]
                Qmatrix[fname] = [str(v) for v in prow]
        # Format tuple ini HARUS sama dengan output data.load_data:
        # ([n], [indeks soal], [correctness], baris attempt...)
        n = len(attempt_rows)
        if n > args.num_steps:  # jaga-jaga, pola asli max 57
            continue
        students.append(tuple([[str(n)], problem_idx, correctness]
                              + attempt_rows))
    return students, Qmatrix


def WriteDummyCSV(students, path):
    """Simpan data dummy dalam format CSV yang sama dengan train/test.CSV."""
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        for student in students:
            for row in student:
                writer.writerow(row)


def LoadTestStudents(path):
    """Baca test.CSV -> list tuple siswa (format sama dgn data.load_data,
    TANPA shuffle) supaya siswa attempt-terpanjang bisa dipilih deterministik."""
    # Pembacaan manual (bukan data.load_data) karena load_data MENG-SHUFFLE;
    # di sini urutan harus tetap agar pemilihan siswa deterministik.
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    students = []
    index = 0
    while index < len(rows) - 1:
        # Tiap blok siswa = 3 baris header + n baris attempt. Siswa dengan
        # n <= 1 dilewati (tidak ada pasangan 'attempt berikutnya').
        n = int(rows[index][0])
        if n <= 1:
            index += 3 + n
            continue
        tup = tuple(rows[index + i] for i in range(n + 3))
        students.append(tup)
        index += n + 3
    return students


def PickLongestStudents(students, k):
    """Urutkan siswa dari jumlah attempt terpanjang, ambil k pertama."""
    # Siswa dengan attempt terbanyak paling informatif untuk melihat
    # perkembangan ability dari waktu ke waktu.
    ranked = sorted(students, key=lambda s: len(s[1]), reverse=True)
    return ranked[:k]


def ReadQmatrixDict(path):
    """Baca P-matrix-out.CSV -> {namaFile: [str x10]} (identik RunModel.ReadQmatrix)."""
    Qmatrix = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            Qmatrix[row[0]] = row[1:]
    return Qmatrix


# ---------------------------------------------------------------------------
# 2. Penyusunan tensor batch -- IDENTIK RunModel.RunEpoch (taskC, test)
# ---------------------------------------------------------------------------
def GetVec(correctness, student, j):
    # Eq. (4): embedding ditaruh di paruh DEPAN kalau salah, paruh BELAKANG
    # kalau benar. Sisi lainnya diisi noise ~1e-10 (bukan nol murni),
    # persis seperti RunModel.GetVec.
    """Salinan RunModel.GetVec: embedding di 10 dim pertama jika salah,
    10 dim terakhir jika benar (Eq. 4 paper)."""
    vec = []
    if int(float(correctness[j])) == 0:
        for _ in range(1, 11):
            vec.append(float(student[3 + j][_]))
        for _ in range(10):
            vec.append(random.uniform(-1e-10, 1e-10))
    else:
        for _ in range(10):
            vec.append(random.uniform(-1e-10, 1e-10))
        for _ in range(1, 11):
            vec.append(float(student[3 + j][_]))
    return vec, student[3 + j][0]


def BuildBatch(batch_students, num_steps, Qmatrix, encoder='LSTM'):
    """Bangun seluruh tensor input persis blok taskB/taskC RunEpoch.

    encoder: 'LSTM' -> model_input dirakit PERSIS spt sebelumnya (matmul
        agregat QmatrixInput^T @ input_data), tidak berubah sama sekali.
    'TransformerCausal' -> mengikuti skema baru RunModel.py: kalau
        args.transformerSeqInput=='True', model_input = input_data mentah
        (urutan waktu asli per-attempt) digabung di dimensi fitur dengan
        Qmatrix_size kolom pertama QmatrixInput (P-matrix per-step),
        BUKAN hasil agregasi matmul yang menghancurkan sumbu waktu.
    """
    # ---- alokasi tensor, semua nol (= padding) ---------------------------
    # Beda kecil vs RunModel: di sini bsz mengikuti sisa siswa, jadi batch
    # terakhir boleh lebih kecil dari args.batch_size (tidak ada yang dibuang).
    bsz = len(batch_students)
    input_data = torch.zeros(bsz, num_steps, args.input_size).to(device)
    QmatrixInput = torch.zeros(bsz, num_steps, num_steps).to(device)
    problemQmatrixMask = torch.zeros(
        bsz, num_steps, args.Qmatrix_size, dtype=torch.uint8).to(device)
    problemQmatrixAbilityMask = torch.zeros(
        bsz, num_steps, args.Qmatrix_size).to(device)
    problemQmatrixSub = torch.zeros(
        bsz, num_steps, args.Qmatrix_size).to(device)
    problemQmatrixProd = torch.zeros(
        bsz, num_steps, args.Qmatrix_size).to(device)

    target_id, target_correctness, vecMess = [], [], []
    for i, student in enumerate(batch_students):
        problem_ids = student[1]
        correctness = student[2]
        # `-1`: attempt terakhir tidak punya label 'berikutnya'.
        for j in range(len(problem_ids) - 1):
            vec, fname = GetVec(correctness, student, j)
            input_data[i, j, :] = torch.tensor(vec, dtype=torch.float)
            vecMess.append(fname)

            # Indeks datar i*T + j -> dipakai torch.gather setelah view(-1).
            target_id.append(i * num_steps + j)
            target_correctness.append(float(correctness[j + 1]))

            # input layer Qmatrix (P-matrix + padding noise)
            tmp = []
            for _ in Qmatrix[fname]:
                tmp.append(int(_) if int(_) == 1
                           else random.uniform(-1e-10, 1e-10))
            for _ in range(num_steps - args.Qmatrix_size):
                tmp.append(random.uniform(-1e-10, 1e-10))
            QmatrixInput[i, j, :] = torch.tensor(tmp, dtype=torch.float)

            # fname = {b|c}_{problemID}_{studentID}_{attemptKe}.py
            # -> split('_')[1] = problemID. k = jumlah konsep yang dipakai soal.
            # Rumus Sub & Prod di bawah HARUS sama persis dengan RunModel.py:
            #   Sub : terlibat k/K,  tak terlibat -0.5
            #   Prod: terlibat 5/(1-k/K), tak terlibat 10
            qrow = PROBLEM_QMATRIX[fname.split('_')[1]]
            k = sum(qrow)
            problemQmatrixMask[i, j, :] = torch.tensor(qrow, dtype=torch.uint8)
            problemQmatrixAbilityMask[i, j, :] = torch.tensor(
                qrow, dtype=torch.float)
            problemQmatrixSub[i, j, :] = torch.tensor(
                [float(k / args.Qmatrix_size) if q == 1 else -0.5
                 for q in qrow], dtype=torch.float)
            problemQmatrixProd[i, j, :] = torch.tensor(
                [5 / (1 - float(k / args.Qmatrix_size)) if q == 1 else 10.0
                 for q in qrow], dtype=torch.float)

    # Rakit model_input FINAL -- IDENTIK logikanya dgn RunModel.py RunEpoch
    # (lihat komentar di sana). LSTM: tidak berubah (matmul agregat lama).
    if encoder == 'LSTM':
        model_input = torch.matmul(QmatrixInput.transpose(1, 2), input_data)
    else:  # TransformerCausal
        if args.transformerSeqInput == 'True':
            model_input = torch.cat(
                [input_data, QmatrixInput[:, :, :args.Qmatrix_size]], dim=-1)
        else:
            model_input = torch.matmul(
                QmatrixInput.transpose(1, 2), input_data)
    return (model_input, QmatrixInput, problemQmatrixMask,
            problemQmatrixAbilityMask, problemQmatrixSub, problemQmatrixProd,
            target_id, target_correctness, vecMess)


# ---------------------------------------------------------------------------
# 3. Load model (deteksi encoder otomatis dari state_dict)
# ---------------------------------------------------------------------------
def DetectEncoder(state_dict):
    # Checkpoint Transformer punya key berawalan 'base_encoder.' /
    # 'transformer...' (lihat Transformer_Encoder_Causal). Kalau tidak ada,
    # berarti checkpoint LSTM.
    for key in state_dict.keys():
        if 'base_encoder' in key or 'transformer' in key:
            return 'TransformerCausal'
    return 'LSTM'


def LoadModel(num_steps):
    # weights_only=True lebih aman (tidak meng-unpickle objek sembarang);
    # torch versi lama belum punya argumen itu -> fallback di except.
    try:
        state_dict = torch.load(args.model_path, map_location=device,
                                weights_only=True)
    except TypeError:  # torch lama tanpa weights_only
        state_dict = torch.load(args.model_path, map_location=device)

    encoder = args.encoder if args.encoder != 'auto' \
        else DetectEncoder(state_dict)
    # Bangun ulang arsitektur dulu, baru isi bobotnya. load_state_dict akan
    # ERROR kalau ada hyperparameter yang berbeda dari saat training.
    # model.eval() WAJIB: mematikan dropout agar hasil deterministik.
    model = HelpDKT_Model(encoder, args, num_skills=args.Qmatrix_size,
                          timeSteps=num_steps).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, encoder


# ---------------------------------------------------------------------------
# 4. Inference
# ---------------------------------------------------------------------------
def RunInference(model, encoder, students, Qmatrix, num_steps):
    """Forward test=True per batch; kembalikan baris hasil per attempt."""
    # Loop batch. init_hidden dipanggil TIAP batch (bukan sekali di awal
    # seperti training) -> tiap batch siswa mulai dari riwayat kosong.
    results = []
    for start in range(0, len(students), args.batch_size):
        batch = students[start:start + args.batch_size]
        (input_data, QmatrixInput, mask, abilityMask, sub, prod,
         target_id, target_corr, vecMess) = BuildBatch(
            batch, num_steps, Qmatrix, encoder=encoder)

        with torch.no_grad():
            hidden = model.init_hidden(len(batch))
            # True = mode test -> model juga mengembalikan ability [B][T][K].
            output, hidden, ability = model(
                input_data, hidden, QmatrixInput, mask, sub,
                abilityMask, prod, True)

        # [B, T, 1] -> [B*T], lalu pungut hanya posisi berlabel -> [N].
        flat = output.contiguous().view(-1)
        preds = torch.gather(
            flat, 0, torch.tensor(target_id, dtype=torch.int64).to(device))

        # c = penunjuk datar yang berjalan sinkron dengan urutan pengisian
        # di BuildBatch (karena keduanya memakai urutan i, lalu j).
        c = 0
        for i, student in enumerate(batch):
            for j in range(len(student[1]) - 1):
                fname = vecMess[c]
                p = preds[c].item()
                results.append({
                    'student': fname.split('_')[2],
                    'step': j,
                    'attempt_file': fname,
                    'next_file': student[3 + j + 1][0],
                    'actual_next': int(target_corr[c]),
                    'pred_prob': p,
                    'pred_label': 1 if p >= 0.5 else 0,
                    # ability[i][j] = daftar K nilai; yang bernilai 0 berarti
                    # konsep itu tidak dipakai soal ini (sudah dimask decoder).
                    'ability': ability[i][j],
                })
                c += 1
    return results


def WriteResults(results, encoder, path):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
    # Baris 1 = metadata, baris 2 = header kolom, sisanya data per attempt.
        writer.writerow(['encoder:', encoder, 'taskModel:', args.taskModel])
        writer.writerow(['student', 'step', 'attempt_file', 'next_file',
                         'actual_next', 'pred_prob', 'pred_label']
                        + ['ability_' + kc for kc in CONCEPT_ORDER])
        for r in results:
            writer.writerow([r['student'], r['step'], r['attempt_file'],
                             r['next_file'], r['actual_next'],
                             '%.6f' % r['pred_prob'], r['pred_label']]
                            + ['%.6f' % a for a in r['ability']])


def main():
    # Seed ganda (random + torch) supaya data dummy DAN noise di GetVec
    # sama persis tiap kali dijalankan.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    num_steps = args.num_steps

    if args.source == 'test':
        # 1. ambil siswa test.CSV asli dengan attempt terpanjang
        all_students = LoadTestStudents(args.test_data_path)
        students = PickLongestStudents(all_students, args.num_test_students)
        Qmatrix = ReadQmatrixDict(args.QmatrixPath)
        lens = [len(s[1]) for s in students]
        ids = [s[3][0].split('_')[2] for s in students]
        print('Data test  : %d siswa test.CSV (id: %s, attempt: %s) dari %d siswa  ->  %s'
              % (len(students), ids, lens, len(all_students), args.test_data_path))
    else:
        # 1. data dummy dengan pola train/test asli
        students, Qmatrix = MakeDummyStudents(args.num_students)
        dummy_csv = os.path.join(dirPath, 'data', 'ModelOutput', 'dummy_data.CSV')
        WriteDummyCSV(students, dummy_csv)
        total_attempts = sum(len(s[1]) for s in students)
        print('Data dummy : %d siswa, %d attempt  ->  %s'
              % (len(students), total_attempts, dummy_csv))

    # 2. model
    model, encoder = LoadModel(num_steps)
    print('Model      : %s  (encoder: %s, %s params)'
          % (args.model_path, encoder,
             format(model.count_parameters(), ',')))

    # 3. inference
    results = RunInference(model, encoder, students, Qmatrix, num_steps)

    # 4. output
    WriteResults(results, encoder, args.output_path)
    # Akurasi kasar dengan threshold 0.5. Untuk -source dummy angka ini
    # tidak bermakna sebagai evaluasi (labelnya sintetis), hanya sanity check.
    correct = sum(1 for r in results if r['pred_label'] == r['actual_next'])
    label_desc = 'label asli' if args.source == 'test' else 'label dummy'
    print('Prediksi   : %d attempt, akurasi vs %s: %.3f'
          % (len(results), label_desc, correct / max(len(results), 1)))
    print('Hasil      : %s' % args.output_path)

    # Cetak ringkas hasil siswa PERTAMA saja; sisanya ada di file CSV.
    print('\nContoh hasil (siswa pertama):')
    first = results[0]['student']
    print('  %-22s %-6s %-9s %s' % ('attempt', 'next', 'p(benar)', 'ability KC terlibat'))
    for r in results:
        if r['student'] != first:
            break
        kcs = ', '.join('%s=%.2f' % (CONCEPT_ORDER[k], r['ability'][k])
                        for k in range(QMATRIX_SIZE) if r['ability'][k] != 0)
        print('  %-22s %-6d %-9.4f %s'
              % (r['attempt_file'], r['actual_next'], r['pred_prob'], kcs))


if __name__ == '__main__':
    main()
