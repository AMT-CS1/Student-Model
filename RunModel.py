"""
RunModel.py -- training loop HELP-DKT (adaptasi dari RunModel.py official
github.com/liangyubuaa/HELP-DKT, Code_HELP_DKT/RunModel.py).

Perbedaan dengan official (selebihnya identik):
    1. Model: HelpDKT_Model (encoder-decoder modular) menggantikan
       HELP_DKT_Model; argumen -encoder memilih 'LSTM' (== official) atau
       'TransformerCausal' tanpa mengubah training loop.
    2. Path data mengikuti struktur proyek ini: ./data/ModelInput dan
       ./data/ModelOutput relatif terhadap file ini.
    3. RunPyFile: official memanggil getDataA/B/C.py; proyek ini memakai
       GetData.py + processData.py, dan hanya dijalankan jika -regenData True
       (train.CSV/test.CSV sudah tersedia).
    4. RepackageHidden menangani hidden None (Transformer stateless).

Jalankan:  python RunModel.py            (LSTM, identik official taskC)
           python RunModel.py -encoder TransformerCausal
"""
# ============================================================================
# PETA FILE INI (komentar tambahan untuk pengembang berikutnya).
#
# RunModel.py = TRAINING LOOP. Alur eksekusinya, dari bawah ke atas:
#   __main__ -> PrintMessage() -> RunPyFile() -> main()
#   main()   -> load_data() -> HelpDKT_Model() -> Adam -> ReadQmatrix()
#               -> loop epoch: RunEpoch(train) + RunEpoch(test tiap N epoch)
#   RunEpoch -> rakit tensor per batch -> model(...) -> loss/metrik
#
# BAGIAN TERPENTING ada di RunEpoch(): 90% file ini isinya MERAKIT TENSOR
# dari data CSV mentah, bukan matematika model. Matematikanya ada di
# HelpDKT_Model.py -> Encoder/ -> decoder/Decoder.py.
#
# Notasi shape yang dipakai di komentar:
#   B = batch_size (32), T = num_steps (attempt terpanjang, mis. 57),
#   D = input_size (20), K = Qmatrix_size (10 knowledge concept)
#
# STRUKTUR SATU SISWA di train.CSV/test.CSV (hasil GetData.py):
#   student[0] = [jumlah attempt]
#   student[1] = [indeks soal per attempt]      <- dipakai sbg problem_ids
#   student[2] = [correctness 0/1 per attempt]
#   student[3+j] = [namaFile, e1..e10]          <- embedding attempt ke-j
#   namaFile = {b|c}_{problemID}_{studentID}_{attemptKe}.py
#              b = attempt salah, c = attempt benar
#   -> student[3+j][0].split('_')[1] = problemID (kunci problemQmatrix)
#   -> student[3+j][0].split('_')[2] = studentID
# ============================================================================
from typing import List

import torch
import torch.nn as nn
import argparse
import random
import datetime
import csv
import os
import sys
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

# load_data: pembaca train/test.CSV -> list tuple per siswa (lihat data.py).
# HelpDKT_Model: model utama (encoder-decoder), lihat HelpDKT_Model.py.
from data import load_data
from HelpDKT_Model import HelpDKT_Model

# Seed TETAP -> noise kecil (random.uniform(-1e-10, 1e-10)) & shuffle data
# reproducible. Ganti angka ini kalau ingin menguji stabilitas antar-seed.
random.seed(1)

# path relatif terhadap file ini (root proyek PredictKCMc)
dirPath = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# ARGPARSE. Catatan penting: banyak flag bertipe str berisi 'True'/'False'
# (BUKAN bool), mengikuti code official. Jadi pengecekannya selalu
# `== 'True'`; menulis `if args.masked:` akan SELALU benar dan salah hasil.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='HELP-DKT model')
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
parser.add_argument('-encoder', type=str,
                    default='LSTM',
                    choices=list(HelpDKT_Model.ENCODER_TYPES),
                    help="tipe encoder: 'LSTM' (official) atau 'TransformerCausal'")
parser.add_argument('-num_heads', type=int,
                    default=8,
                    help='jumlah attention head (hanya TransformerCausal)')
parser.add_argument('-transformer_dropout', type=float,
                    default=0.1,
                    help='dropout Transformer (hanya TransformerCausal)')
parser.add_argument('-transformerSeqInput', type=str,
                    default='True',
                    help="hanya TransformerCausal. 'True' = pakai skema input BARU: "
                         "urutan waktu per-attempt dipertahankan (embedding asli step j "
                         "digabung dengan P-matrix step j di dimensi fitur), TIDAK memakai "
                         "agregasi matmul(QmatrixInput^T, input) yang menghancurkan sumbu "
                         "waktu dan menyebabkan output Transformer nyaris konstan antar "
                         "step/siswa. 'False' = perilaku lama (sama dgn LSTM, utk komparasi "
                         "kompatibilitas checkpoint lama saja, TIDAK disarankan). "
                         "Tidak berpengaruh sama sekali ke -encoder LSTM.")
parser.add_argument('-regenData', type=str,
                    default='False',
                    help='True = regenerasi train/test.CSV via GetData.py + processData.py')
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
parser.add_argument('-linearWithQmatrix', type=str,
                    default='False',
                    help='whether the output linear layer with Qmatrix')
parser.add_argument('-multiLinearLayers', type=str,
                    default='True',
                    help='whether the num of output layers is more than 1 or not')
parser.add_argument('-one_hot', type=str,
                    default='True',
                    help='whether input layer combine vec and Qmatrix based on one-hot')
parser.add_argument('-inputConnectQmatrix', type=str,
                    default='False',
                    help='whether input vec conncet with Qmatrix')
parser.add_argument('-input_size', type=int,
                    default=20,
                    help='set input size for model and RunEpoch')
parser.add_argument('-inputQmatrixType', type=str,
                    default='P_Qmatrix',
                    help='input vec concrete or multiply with which Qmatrix type')
parser.add_argument('-inputMulQmatix', type=str,
                    default='True',
                    help='whether the input vec matmul with the Qmatrix')
parser.add_argument('-taskModel', type=str,
                    default='taskC',
                    help='run which task model')
parser.add_argument('-QmatrixType', type=str,
                    default='P_Qmatrix',
                    help='use which Qmatrix type: personalized or original Qmatrix')
parser.add_argument('-masked', type=str,
                    default='True',
                    help='personalized or original Qmatrix')
parser.add_argument('-set2zero', type=str,
                    default='True',
                    help='whether set some of the output of the first linear layer to zero')
parser.add_argument('-subQmatrix', type=str,
                    default='True',
                    help='whether the output of the first linear layer sub the Qmatrix')
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
parser.add_argument('-tryNum', type=int,
                    default=3,
                    help='the threshold for taskA')
parser.add_argument('-epochs', type=int,
                    default=500,
                    help='Number of epochs to train')
parser.add_argument('-learning_rate', type=float,
                    default=0.05,
                    help='Learning rate')
parser.add_argument('-batch_size', type=int,
                    default=32,
                    help='Batch size for training')
parser.add_argument('-Qmatrix_size', type=int,
                    default=10,
                    help='the size of one Qmatrix')
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
parser.add_argument('-epsilon', type=float, default=0.1,
                    help='Epsilon value for Adam Optimizer (khusus -encoder LSTM, '
                         'nilai official/paper -- TIDAK dipakai utk TransformerCausal)')
parser.add_argument('-l2_lambda', type=float, default=0.3,
                    help='Lambda for l2 loss')
parser.add_argument('-max_grad_norm', type=float, default=20,
                    help='Clip gradients to this norm (khusus -encoder LSTM)')
# --- hyperparameter optimizer KHUSUS TransformerCausal, TIDAK memengaruhi
# LSTM sama sekali. lr=0.05/eps=0.1/grad_norm=20 di atas adalah nilai
# official paper yang di-tuning utk LSTM+Adam; angka itu JAUH terlalu besar
# utk melatih self-attention dari nol (praktik umum Transformer: lr~1e-4-5e-4,
# eps=1e-8, grad clip ~1.0). Dgn lr setinggi itu, training Transformer bisa
# cepat jenuh (weight/attention/sigmoid saturasi) dlm beberapa epoch pertama
# sehingga model berhenti sensitif thd input -- gejalanya PERSIS "ability
# beku" yg tetap muncul walau skema input sudah diperbaiki.
parser.add_argument('-transformer_learning_rate', type=float, default=5e-4,
                    help='Learning rate KHUSUS TransformerCausal (default jauh '
                         'lebih kecil drpd -learning_rate LSTM)')
parser.add_argument('-transformer_epsilon', type=float, default=1e-8,
                    help='Epsilon Adam KHUSUS TransformerCausal (standar Adam, '
                         'bukan 0.1 spt LSTM)')
parser.add_argument('-transformer_max_grad_norm', type=float, default=1.0,
                    help='Grad clip norm KHUSUS TransformerCausal (standar '
                         'training Transformer, jauh lebih ketat drpd LSTM)')
parser.add_argument('-transformer_warmup_steps', type=int, default=200,
                    help='Jumlah step linear LR warmup di awal training '
                         'TransformerCausal (0 = nonaktif). Warmup mencegah '
                         'attention jenuh di awal saat bobot masih acak.')
parser.add_argument('-keep_prob', type=float, default=0.6,
                    help='Keep probability for dropout')
parser.add_argument('-hidden_layer_num', type=int, default=3,
                    help='The number of hidden layers')
parser.add_argument('-hidden_size', type=int, default=200,
                    help='The number of hidden nodes')
parser.add_argument('-evaluation_interval', type=int, default=5,
                    help='Evalutaion and print result every x epochs')
parser.add_argument('-output_size', type=int, default=6,
                    help='model linear layer output size')
parser.add_argument('-num_problems', type=list,
                    default=[10], help='num of problems in original data')
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
parser.add_argument('-train_data_path', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelInput', 'train.CSV'),
                    help='Path to the training dataset')
parser.add_argument('-test_data_path', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelInput', 'test.CSV'),
                    help='Path to the testing dataset')
parser.add_argument('-QmatrixPath', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelInput', 'P-matrix-out.CSV'),
                    help='Path to the personalized Q-matrix')
parser.add_argument('-ProblemQmatrixPath', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelInput', 'problemQmatrix.CSV'),
                    help='Path to the problemQmatrix')
parser.add_argument('-model_output', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelOutput', 'result.txt'),
                    help='Path to the result file')
parser.add_argument('-id2problems', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelInput', 'ID2problem.CSV'),
                    help='Path to the id2problems dataset')
parser.add_argument('-model_path', type=str,
                    default=os.path.join(dirPath, 'data', 'ModelOutput', 'model.pth'),
                    help='Path to save the trained model')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# STATE GLOBAL
# ---------------------------------------------------------------------------

savedStdout = sys.stdout
# (disimpan tapi tidak dipakai lagi -- peninggalan code official)

# Satu device untuk seluruh proses: model & SEMUA tensor input wajib ke sini.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Diisi sekali oleh ReadQmatrix()/ReadId2Problems() lalu dibaca terus di RunEpoch:
#   Qmatrix        : {namaFile -> baris P-matrix (personalized, per ATTEMPT)}
#   problemQmatrix : {problemID -> baris Q-matrix (per SOAL)}
#   id2problems    : {studentID -> {problemID -> jumlah percobaan}} (taskA)
Qmatrix = {}
problemQmatrix = {}
id2problems = {}


def RunPyFile():
    '''
    @description: Regenerasi data input model (opsional, -regenData True).
                  Official memanggil getDataA/B/C.py; proyek ini memakai
                  GetData.py lalu processData.py.
    '''
    # Default -regenData False -> langsung return, karena train/test.CSV
    # biasanya sudah ada. Set True hanya kalau data mentah berubah.
    if args.regenData != 'True':
        return
    os.system('"%s" "%s" -QmatrixType %s -Qmatrix_size %s' % (
        sys.executable, os.path.join(dirPath, 'GetData.py'),
        args.QmatrixType, args.Qmatrix_size))
    os.system('"%s" "%s"' % (
        sys.executable, os.path.join(dirPath, 'processData.py')))
    return


def RepackageHidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history.
    None (Transformer, stateless) dikembalikan apa adanya."""
    # Transformer stateless -> hidden None, biarkan apa adanya.
    if h is None:
        return None
    # detach() memutus tali ke graph batch SEBELUMNYA. Tanpa ini, backward()
    # akan mencoba menelusuri seluruh riwayat batch dan error/meledak memori.
    # Nilai hidden-nya sendiri TIDAK berubah (tetap dibawa antar batch).
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
    # LSTM -> hidden berupa tuple (h, c): rekursif ke tiap elemen.
        return tuple(RepackageHidden(v) for v in h)


def RunEpoch(m, optimizer, students, batch_size, num_steps, num_skills, training=True, epoch=1, scheduler=None):
    """Runs the model on the given data.

    scheduler: opsional, hanya dipakai kalau args.encoder=='TransformerCausal'
    (LR warmup). None utk LSTM -- tidak ada perubahan perilaku sama sekali.
    """
    # --- akumulator lintas batch --------------------------------------
    # actual_labels / pred_labels dikumpulkan SATU EPOCH penuh, baru dihitung
    # AUC/akurasi di akhir -> metrik level epoch, bukan level batch.
    total_loss = 0
    index = 0
    actual_labels = []
    pred_labels = []
    testSetVesMess = []  # save TEST set's message
    # Hidden diinisialisasi SEKALI di awal epoch, lalu dibawa antar batch
    # (LSTM: stateful). Transformer mengembalikan None -> tidak berpengaruh.
    hidden = m.init_hidden(batch_size)
    count = 0
    ability = []
    predsAll = []
    # LOOP BATCH. Perhatikan `+ batch_size <` : sisa siswa yang tidak cukup
    # satu batch penuh DIBUANG (perilaku official, batch selalu tepat B).
    while (index + batch_size < len(students)):
        # target_id  : posisi datar (i*T + j) dari prediksi yang punya label
        # target_correctness: label benar/salah attempt BERIKUTNYA (j+1)
        target_id: List[int] = []
        target_correctness = []
        # ---- alokasi tensor batch, semua di-nol-kan dulu ----------------
        # Timestep yang tidak terpakai (siswa dgn attempt < T) tetap 0 =
        # padding. Padding inilah yang nanti disaring lewat AbilityMask.
        # input_data [B, T, D] -- embedding program tiap attempt (Eq. 4)
        input_data = torch.FloatTensor(
            batch_size, num_steps, args.input_size).to(device)
        input_data = input_data.zero_()

        # QmatrixInput [B, T, T] -- P-matrix per attempt + padding noise.
        # Lebarnya T (bukan K) karena LSTM memakainya untuk matmul agregat:
        # matmul(QmatrixInput^T [B,T,T], input_data [B,T,D]) -> [B,T,D].
        QmatrixInput = torch.FloatTensor(
            batch_size, num_steps, num_steps).to(device)
        QmatrixInput = QmatrixInput.zero_()

        # problemQmatrixMask [B, T, K] -- 0/1 Q-matrix soal. CATATAN: tensor
        # ini diteruskan ke model tapi TIDAK dipakai decoder (lihat Decoder.py).
        problemQmatrixMask = torch.ByteTensor(
            batch_size, num_steps, args.Qmatrix_size
        ).to(device)
        problemQmatrixMask = problemQmatrixMask.zero_()

        # problemQmatrixAbilityMask [B, T, K] -- 1 = konsep dipakai soal ini.
        # INI yang menentukan konsep mana yang di-mask & baris mana yang
        # dianggap padding saat pelaporan ability.
        problemQmatrixAbilityMask = torch.FloatTensor(
            batch_size, num_steps, args.Qmatrix_size
        ).to(device)
        problemQmatrixAbilityMask = problemQmatrixAbilityMask.zero_()

        # problemQmatrixSub [B, T, K] -- ambang kesulitan efektif (theta versi
        # official): dikurangkan dari ability di decoder.
        problemQmatrixSub = torch.FloatTensor(
            batch_size, num_steps, args.Qmatrix_size
        ).to(device)
        problemQmatrixSub = problemQmatrixSub.zero_()

        # problemQmatrixProd [B, T, K] -- faktor skala (alpha versi official):
        # dikalikan sebelum sigmoid di decoder.
        problemQmatrixProd = torch.FloatTensor(
            batch_size, num_steps, args.Qmatrix_size
        ).to(device)
        problemQmatrixProd = problemQmatrixProd.zero_()

        # vecMess: nama file tiap attempt, dipakai memberi label baris CSV hasil.
        vecMess = []
        # ---- isi tensor, satu siswa (i) per iterasi ---------------------
        for i in range(batch_size):
            student = students[index + i]
            # problem_ids = student[1] -> panjangnya = jumlah attempt siswa ini
            problem_ids = student[1]
            correctness = student[2]

            # ============ taskA: 'apakah soal berikutnya selesai dlm <= tryNum
            # percobaan' -- TIDAK dipakai proyek ini (default taskC). Blok ini
            # dipertahankan demi parity dengan code official; boleh dilewati
            # saat membaca. Langsung ke cabang elif taskB/taskC di bawah. =====
            if args.taskModel == 'taskA':
                for j in range(len(problem_ids)):
                    vec, tmp = GetVec(correctness, student, j)
                    input_data[i, j, :] = torch.tensor(
                        vec, dtype=torch.float64).to(device)
                    vecMess.append(tmp)
                    testSetVesMess.append(tmp)

                    tmp = []  # input layer Qmatrix
                    for _ in Qmatrix[student[3+j][0]]:
                        if int(_) == 1:
                            tmp.append(int(_))
                        else:
                            tmp.append(random.uniform(-1e-10, 1e-10))
                    for _ in range(num_steps-args.Qmatrix_size):
                        tmp.append(random.uniform(-1e-10, 1e-10))
                    QmatrixInput[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64).to(device)

                    tmp = []  # masked
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmp.append(0)  # set 0
                        else:
                            tmp.append(1)
                    problemQmatrixMask[i, j, :] = torch.tensor(
                        tmp, dtype=torch.uint8
                    ).to(device)

                    tmp = []  # problemQmatrixAbilityMask
                    for _ in Qmatrix[student[3+j][0]]:
                        if int(_) == 1:
                            tmp.append(1)  # torch.mul()
                        else:
                            tmp.append(0)
                    problemQmatrixAbilityMask[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64
                    ).to(device)

                    tmp = []  # sub
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmp.append(1)  # set 1 as original Qmatrix data
                        else:
                            tmp.append(0)
                    problemQmatrixSub[i, j, :] = torch.tensor(
                        tmp, dtype=torch.uint8
                    ).to(device)

                    tmp = []  # prod
                    tmpInt = 0
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmpInt += 1
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmp.append(5/(1-float(tmpInt/args.Qmatrix_size)))
                        else:
                            tmp.append((1-float(tmpInt/args.Qmatrix_size)))
                    problemQmatrixProd[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64
                    ).to(device)

                tmp = GetNextProblem(
                    id2problems[student[3][0].split('_')[2]], problem_ids[0])
                target_correctness.append(tmp)
                actual_labels.append(tmp)
            # ================= JALUR AKTIF (taskC) ========================
            # Prediksi: 'apakah attempt BERIKUTNYA akan benar?'
            elif args.taskModel == 'taskB' or args.taskModel == 'taskC':
                # `-1` karena attempt TERAKHIR tidak punya 'berikutnya' untuk
                # dijadikan label.
                for j in range(len(problem_ids)-1):
                    # vec [D=20] embedding attempt j; tmp = nama file attempt j.
                    vec, tmp = GetVec(correctness, student, j)
                    input_data[i, j, :] = torch.tensor(
                        vec, dtype=torch.float64).to(device)
                    vecMess.append(tmp)
                    testSetVesMess.append(tmp)

                    # Indeks DATAR setelah output di-view(-1): baris i kolom j
                    # -> i*T + j. Dipakai torch.gather di bawah untuk memungut
                    # hanya timestep yang punya label (buang padding).
                    # Label = correctness attempt j+1 (bukan j) -> next-step.
                    target_id.append(i * num_steps + j + 0)
                    target_correctness.append(
                        float(correctness[j+1]))
                    actual_labels.append(int(float(correctness[j+1])))

                    # P-matrix attempt ini: 1 tetap 1, 0 diganti noise ~1e-10.
                    # Noise (bukan 0 murni) adalah trik official agar hasil
                    # matmul tidak persis nol; besarnya dapat diabaikan.
                    # Sisa kolom (T - K) diisi noise sebagai padding.
                    tmp = []  # input layer Qmatrix
                    for _ in Qmatrix[student[3+j][0]]:
                        if int(_) == 1:
                            tmp.append(int(_))
                        else:
                            tmp.append(random.uniform(-1e-10, 1e-10))
                    for _ in range(num_steps-args.Qmatrix_size):
                        tmp.append(random.uniform(-1e-10, 1e-10))
                    QmatrixInput[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64).to(device)

                    # Salinan langsung baris Q-matrix soal (0/1).
                    tmp = []  # masked
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmp.append(1)
                        else:
                            tmp.append(0)
                    problemQmatrixMask[i, j, :] = torch.tensor(
                        tmp, dtype=torch.uint8
                    ).to(device)

                    # Sama isinya dengan mask di atas, tapi INI yang benar-benar
                    # dipakai decoder untuk memilih konsep & menandai padding.
                    tmp = []  # problemQmatrixAbilityMask
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            # ability & problemQmatrixAbilityMask
                            tmp.append(1)
                        else:
                            tmp.append(0)
                    problemQmatrixAbilityMask[i, j, :] = torch.tensor(
                        tmp, dtype=torch.uint8
                    ).to(device)

                    # tmpInt = k = jumlah konsep yang dipakai soal ini.
                    #   konsep terlibat     -> k/K   (ambang; makin banyak
                    #                          konsep makin tinggi ambangnya)
                    #   konsep tak terlibat -> -0.5  (dikurangi -0.5 = ditambah
                    #                          0.5 -> sigmoid ~0.993, praktis
                    #                          netral terhadap torch.prod)
                    tmp = []  # sub
                    tmpInt = 0
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmpInt += 1
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            # set 1 as original Qmatrix
                            tmp.append(float(tmpInt/args.Qmatrix_size))
                        else:
                            tmp.append(-0.5)
                    problemQmatrixSub[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64
                    ).to(device)

                    # Faktor skala sebelum sigmoid:
                    #   konsep terlibat     -> 5/(1 - k/K)
                    #   konsep tak terlibat -> 10  (0.5 * 10 = 5 -> sigmoid
                    #                          ~0.993, netral di perkalian)
                    tmp = []  # prod
                    tmpInt = 0
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmpInt += 1
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmp.append(5/(1-float(tmpInt/args.Qmatrix_size)))
                        else:
                            tmp.append(float(10))

                    problemQmatrixProd[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64
                    ).to(device)

        # Geser jendela ke siswa batch berikutnya.
        index += batch_size
        count += 1

        # Pindahkan list Python -> tensor di device untuk gather & loss.
        # target_id2 [N] int64, target_correctness [N] float, N = jumlah
        # attempt berlabel di batch ini (<= B*T karena ada padding).
        target_id2 = torch.tensor(
            target_id, dtype=torch.int64).to(device)
        target_correctness = torch.tensor(
            target_correctness, dtype=torch.float).to(device)

        
        # ---- RAKIT INPUT FINAL untuk model ------------------------------
        # Dua skema berbeda, INILAH perbedaan utama LSTM vs Transformer:
        #
        # LSTM (official): matmul(QmatrixInput^T, input_data) -> [B, T, D].
        #   Ini AGREGASI: baris ke-c hasilnya = jumlah embedding semua attempt
        #   yang memakai konsep c. Sumbu waktu berubah jadi 'sumbu konsep'.
        #
        # TransformerCausal (skema baru, default): concat di dimensi fitur
        #   -> [B, T, D+K]. Urutan waktu per-attempt DIPERTAHANKAN, karena
        #   self-attention + positional encoding butuh sumbu waktu asli.
        #   Memakai skema agregat di Transformer membuat output nyaris
        #   konstan antar step/siswa (gejala 'ability beku').
        if args.encoder == 'LSTM':
            if args.inputMulQmatix == 'True':
                model_input = torch.matmul(
                    QmatrixInput.transpose(1, 2), input_data)
            else:
                model_input = input_data
        else:  # TransformerCausal
            if args.transformerSeqInput == 'True':
                model_input = torch.cat(
                    [input_data, QmatrixInput[:, :, :args.Qmatrix_size]],
                    dim=-1)
            elif args.inputMulQmatix == 'True':
                # fallback skema lama (utk load/komparasi checkpoint lama)
                model_input = torch.matmul(
                    QmatrixInput.transpose(1, 2), input_data)
            else:
                model_input = input_data

        # ================= MODE TRAINING ==================================
        if training:
            m.train()
            # m.train() -> dropout AKTIF; hidden di-detach dari batch lalu;
            # zero_grad() menghapus gradien batch sebelumnya (PyTorch
            # mengakumulasi gradien secara default).
            hidden = RepackageHidden(hidden)
            optimizer.zero_grad()
            # FORWARD. Argumen terakhir False = mode training -> model
            # mengembalikan (decoded [B,T,1], hidden) tanpa ability.
            output, hidden = m(model_input, hidden,
                               QmatrixInput, problemQmatrixMask, problemQmatrixSub, problemQmatrixAbilityMask, problemQmatrixProd, False)

            # Get predictions
            # [B, T, 1] -> [B*T] datar. contiguous() diperlukan agar view()
            # legal setelah operasi yang mengubah stride.
            output = output.contiguous().view(-1)
            if args.taskModel == 'taskA':
                logitsPred = output
            elif args.taskModel == 'taskB' or args.taskModel == 'taskC':
                # Pungut hanya posisi berlabel -> [N]. Padding otomatis
                # tersingkir karena indeksnya tidak ada di target_id2.
                logitsPred = torch.gather(output, 0, target_id2).to(device)

            # preds
            # multiLinearLayers 'True' (jalur HELP-DKT penuh): decoder SUDAH
            # mengeluarkan probabilitas (hasil sigmoid + prod), jadi TIDAK
            # disigmoid lagi dan loss-nya BCELoss.
            # 'False' (DKT biasa): output masih logit -> perlu sigmoid dan
            # BCEWithLogitsLoss. Salah pasang di sini = sigmoid ganda.
            if args.multiLinearLayers == 'False':
                preds = torch.sigmoid(logitsPred).to(device)
            else:
                preds = logitsPred.to(device)

            for p in preds:
                pred_labels.append(p.item())

            if args.multiLinearLayers == 'False':
                criterion = nn.BCEWithLogitsLoss().to(device)
            else:
                criterion = nn.BCELoss().to(device)
            loss = criterion(
                logitsPred, target_correctness)
            # Hitung gradien seluruh parameter (encoder + decoder).
            loss.backward()

            # grad clip: LSTM pakai -max_grad_norm lama (20, longgar, sesuai
            # official). TransformerCausal pakai -transformer_max_grad_norm
            # (default 1.0, standar training Transformer) -- self-attention
            # jauh lebih rentan meledak/jenuh drpd LSTM dgn clip selonggar itu.
            grad_norm = (args.max_grad_norm if args.encoder == 'LSTM'
                        else args.transformer_max_grad_norm)
            torch.nn.utils.clip_grad_norm_(m.parameters(), grad_norm)

            # Update bobot, setelah gradien di-clip di atas.
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # LR warmup, khusus TransformerCausal

            # .item() melepas tensor dari graph -> tidak menahan memori.
            total_loss += loss.item()
        else:
            # ================= MODE EVALUASI/TEST =========================
            # no_grad(): tidak membangun graph -> hemat memori & lebih cepat.
            # m.eval(): dropout DIMATIKAN (wajib, kalau tidak hasil acak).
            with torch.no_grad():
                m.eval()
                # Argumen terakhir True = mode test -> model mengembalikan
                # 3 nilai; `tmp` = ability per KC (list [B][T][K]).
                output, hidden, tmp = m(model_input, hidden,
                                        QmatrixInput, problemQmatrixMask, problemQmatrixSub, problemQmatrixAbilityMask, problemQmatrixProd, True)
                # Tempelkan nama file ke tiap baris ability/prediksi supaya
                # hasil CSV bisa ditelusuri ke attempt aslinya.
                tmp_predsAll, tmp_ability = addVecMess(
                    output.tolist(), tmp, vecMess)
                predsAll += tmp_predsAll
                ability += tmp_ability

                output = output.contiguous().view(-1)
                if args.taskModel == 'taskA':
                    logitsPred = output
                elif args.taskModel == 'taskB' or args.taskModel == 'taskC':
                    logitsPred = torch.gather(output, 0, target_id2).to(device)

                # preds
                if args.multiLinearLayers == 'False':
                    preds = torch.sigmoid(logitsPred).to(device)
                else:
                    preds = logitsPred.to(device)
                for p in preds:
                    pred_labels.append(p.item())

                if args.multiLinearLayers == 'False':
                    criterion = nn.BCEWithLogitsLoss().to(device)
                else:
                    criterion = nn.BCELoss().to(device)
                loss = criterion(
                    logitsPred, target_correctness)
                total_loss += loss.item()

                hidden = RepackageHidden(hidden)

    # ---- METRIK SATU EPOCH ----------------------------------------------
    # CATATAN PENAMAAN: variabel ini bernama `rmse`, tapi yang dikembalikan
    # fungsi (dan dicetak sebagai 'loss') adalah total_loss/jumlah_batch.
    # Nilai rmse di baris ini sebenarnya TIDAK ikut di-return.
    rmse = sqrt(mean_squared_error(
        actual_labels, pred_labels))
    # AUC = metrik utama paper: seberapa baik model MERANGKING benar vs salah,
    # tidak bergantung pemilihan threshold.
    fpr, tpr, thresholds = metrics.roc_curve(
        actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # Threshold 0.5 -> label biner, untuk accuracy/precision/recall/F1.
    all_pred = []
    for _ in pred_labels:
        if _ >= 0.5:
            all_pred.append(1.0)
        else:
            all_pred.append(0.0)

    prec, rec, f1, _ = precision_recall_fscore_support(
        actual_labels, all_pred, average="binary")
    accuracy = metrics.accuracy_score(actual_labels, all_pred)

    # Tulis ability ke CSV hanya sesekali (epoch ke-99, 199, ...) supaya
    # file tidak membengkak. Filter kedua ada di writeAbilityPredsMess.
    if training == False and epoch % 100 == 99:
        writeAbilityPredsMess(ability, predsAll, epoch)

    # Nilai pertama = rata-rata loss per batch (dicetak sebagai 'loss').
    return total_loss/(len(students)/batch_size), auc, accuracy, prec, rec, f1


def writeAbilityPredsMess(ability, predsAll, epoch):
    '''
    @description: Write ability messages
    '''
    # Filter ganda: hanya taskC DAN hanya di dekat epoch terakhir. Jadi file
    # ability yang tersimpan = kondisi model SETELAH training selesai.
    if args.taskModel != 'taskC' or epoch < args.epochs - args.evaluation_interval:
        return

    resultParas = args.taskModel + '-' + args.encoder
    # Nama file dibedakan per encoder -> hasil LSTM & Transformer tidak
    # saling menimpa. Mode 'a+' = APPEND, file lama TIDAK dihapus; hapus
    # manual kalau ingin mulai bersih.

    path = os.path.join(dirPath, 'data', 'ModelOutput',
                        'studentAbilityMess-' + str(resultParas) + '.CSV')
    with open(path, 'a+', encoding='utf-8-sig', newline='') as fileOutput:
        writer = csv.writer(fileOutput)
        writer.writerow(
            ['epoch:', epoch, 'taskModel:', args.taskModel, '\n\n'])
        writer.writerow(ability)


def addVecMess(output, ability, vecMess):
    # Menempelkan nama file attempt ke tiap baris ability & prediksi.
    #
    # ability [B][T][K] & output [B][T][1] datang dari model dalam urutan
    # batch x timestep, sedangkan vecMess adalah daftar DATAR nama file
    # (hanya attempt yang benar-benar ada, tanpa padding). Karena itu
    # counter `count` hanya maju saat baris tidak kosong -- begitulah kedua
    # urutan tetap sinkron.
    #
    # `any(ability[i][j])` bernilai False untuk timestep padding, karena
    # Decoder.py sudah menolkan baris padding lewat tensor tmp.
    #
    # Return: (preds, result) -- keduanya list bersarang [B][T][1+K] berisi
    # [namaFile, nilai...], siap ditulis ke CSV.
    # taskA/taskB tidak menghasilkan ability per konsep -> tidak ada yang
    # perlu dilaporkan.
    if args.taskModel == 'taskA' or args.taskModel == 'taskB':
        return [], []
    result = []
    count = 0
    for i in range(len(ability)):
        result.append([])
        for j in range(len(ability[0])):
            result[i].append([])
            if any(ability[i][j]):
                result[i][j].append(vecMess[count])
                for _ in ability[i][j]:
                    result[i][j].append(_)
                count += 1

    preds = []
    count = 0
    for i in range(len(ability)):
        preds.append([])
        for j in range(len(ability[0])):
            preds[i].append([])
            if any(ability[i][j]):
                preds[i][j].append(vecMess[count])
                for _ in output[i][j]:
                    preds[i][j].append(_)
                count += 1

    return preds, result


def GetVec(correctness, student, j):
    '''
    @description: Get vec as model's input
    @return: vec
    '''
    # CATATAN: `tmp` dihitung di sini (P-matrix / Q-matrix attempt), tapi
    # TIDAK dipakai lagi -- sisa dari code official. Yang di-return hanya
    # `vec` dan nama file.
    tmp = []
    if args.inputQmatrixType == 'P_Qmatrix':
        for _ in Qmatrix[student[3+j][0]]:
            if int(_) == 1:
                tmp.append(int(_))
            else:
                tmp.append(random.uniform(-1e-10, 1e-10))
    else:
        for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
            if int(_) == 1:
                tmp.append(int(_))
            else:
                tmp.append(random.uniform(-1e-10, 1e-10))

    # === Eq. (4) paper: POSISI menyandikan benar/salah ===================
    # Panjang vec = 20. Embedding 10-dim ditaruh di:
    #   dim 0..9   kalau attempt ini SALAH  (sisanya noise ~0)
    #   dim 10..19 kalau attempt ini BENAR  (sisanya noise ~0)
    # Jadi model tahu 'apa yang ditulis siswa' DAN 'hasilnya benar/salah'
    # dari satu vektor yang sama.
    vec = []
    if (int(float(correctness[j])) == 0):
        for _ in range(1, 11):
            vec.append(float(student[3 + j][_]))
        for _ in range(10):
            vec.append(random.uniform(-1e-10, 1e-10))
    else:
        for _ in range(10):
            vec.append(random.uniform(-1e-10, 1e-10))
        for _ in range(1, 11):
            vec.append(float(student[3 + j][_]))

    # student[3+j][0] = nama file attempt ke-j.
    return vec, student[3+j][0]


def TakeEle(ele):
    # key sort: ubah string ID soal jadi int supaya '10' > '9' (bukan '10' < '9').
    return int(ele)


def GetNextProblem(problems, problemNum):
    '''
    @description: Get problem situation
    @return: 1: tryNum of next problem <= args.tryNum
    '''
    # Cari soal pertama yang ID-nya lebih besar dari soal sekarang, lalu cek
    # apakah siswa menyelesaikannya dalam <= args.tryNum percobaan.
    # (Khusus taskA; mengembalikan None kalau tidak ada soal berikutnya.)
    tmp = []
    for i in problems.keys():
        tmp.append(i)
    tmp.sort(key=TakeEle)
    for i in tmp:
        if int(i) > int(problemNum):
            if int(problems[i]) <= args.tryNum:
                return 1
            else:
                return 0


def ReadQmatrix():
    '''
    @description: Get Qmatrix from args.QmatrixPath & args.ProblemQmatrixPath
    '''
    global Qmatrix
    # P-matrix-out.CSV: kolom 0 = nama file attempt, kolom 1.. = 0/1 per KC.
    # 'utf-8-sig' penting: membuang BOM yang ditulis Excel di awal file.
    # -> Qmatrix[namaFile] = ['1','0','1',...]
    with open(args.QmatrixPath, 'r', encoding='utf-8-sig') as fileInput:
        reader = csv.reader(fileInput)
        for i in reader:
            tmp = []
            for _ in i[1:]:
                tmp.append(_)
            Qmatrix[i[0]] = tmp
    # problemQmatrix.CSV: kolom 0 = problemID, kolom 1.. = 0/1 per KC.
    # -> problemQmatrix[problemID] = ['1','1','0',...]
    with open(args.ProblemQmatrixPath, 'r', encoding='utf-8-sig') as fileInput:
        reader = csv.reader(fileInput)
        for i in reader:
            tmp = []
            for _ in i[1:]:
                tmp.append(_)
            problemQmatrix[i[0]] = tmp
    return


def ReadId2Problems():
    '''
    @description: Get information of Qmatrix-id
    '''
    global id2problems
    with open(args.id2problems, "r", encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # id2problems[studentID][problemID] = jumlah percobaan (taskA).
            if row[0] not in id2problems.keys():
                id2problems[row[0]] = {row[1]: row[2]}
            else:
                id2problems[row[0]][row[1]] = row[2]
    return id2problems


def main():
    # Penampung hasil TERBAIK selama training (dipilih berdasarkan AUC test).
    maxAuc = 0
    maxRmse = 0
    maxEpoch = 0
    maxAcc = 0
    maxPrec = 0
    maxRec = 0
    maxF1 = 0

    startTime = datetime.datetime.now()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    batch_size = args.batch_size
    # Baca train & test. load_data juga MENG-SHUFFLE urutan siswa.
    train_students, train_max_num_problems, train_max_skill_num = load_data(
        train_data_path)
    num_skills = train_max_skill_num
    test_students, test_max_num_problems, test_max_skill_num = load_data(
        test_data_path)
    # T = attempt terpanjang di train ATAU test -> satu ukuran tensor untuk
    # keduanya. Siswa yang lebih pendek otomatis ter-padding nol.
    num_steps = max(test_max_num_problems, train_max_num_problems)

    # HelpDKT_Model: drop-in replacement HELP_DKT_Model official;
    # args.encoder = 'LSTM' (identik official) atau 'TransformerCausal'
    model = HelpDKT_Model(
        args.encoder, args, num_skills, num_steps).to(device)

    # Optimizer: LSTM TIDAK berubah (lr/eps official/paper). TransformerCausal
    # pakai hyperparameter terpisah (-transformer_learning_rate/-epsilon,
    # jauh lebih kecil) + LR warmup linear -- lr=0.05/eps=0.1 official
    # terlalu agresif utk self-attention yg dilatih dari nol & menyebabkan
    # training cepat jenuh/collapse (ability jadi konstan meski skema input
    # sudah benar).
    scheduler = None
    if args.encoder == 'LSTM':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, eps=args.epsilon)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.transformer_learning_rate,
            eps=args.transformer_epsilon)
        if args.transformer_warmup_steps > 0:
            warmup_steps = args.transformer_warmup_steps

            def _warmup_lr_lambda(step):
                return min(1.0, float(step + 1) / float(warmup_steps))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, _warmup_lr_lambda)

    # Muat kamus Q-matrix/P-matrix SEBELUM epoch pertama (dipakai RunEpoch).
    ReadQmatrix()
    ReadId2Problems()

    # train & test model:
    # ---- LOOP UTAMA TRAINING --------------------------------------------
    # Tiap epoch: sekali RunEpoch training; tiap `evaluation_interval` epoch
    # tambah sekali RunEpoch test.
    aucList = ['auc']
    lossList = ['loss']
    for i in range(args.epochs):
        rmse, auc, accuracy, prec, rec, f1 = RunEpoch(
            model, optimizer, train_students, batch_size, num_steps, num_skills,
            epoch=i, scheduler=scheduler)
        # Testing
        # training=False -> tidak ada backward/optimizer.step, dan model
        # mengembalikan ability untuk dilaporkan.
        if ((i + 1) % args.evaluation_interval == 0):
            rmse, auc, accuracy, prec, rec, f1 = RunEpoch(model, optimizer, test_students,
                                                          batch_size, num_steps, num_skills, training=False, epoch=i)
            print("TEST result:\n\tepoch:{}\n\tmodel:{}\n\tencoder:{}\n\tauc:{}\n\tacc:{}\n\tloss:{}\n\tprec:{}\n\trec:{}\n\tf1:{}\n\t".format(
                i+1, args.taskModel, args.encoder, auc, accuracy, rmse, prec, rec, f1))

            # Simpan snapshot metrik saat AUC test mencapai rekor baru.
            # CATATAN: bobot model TIDAK ikut disimpan di titik ini --
            # torch.save di bawah menyimpan bobot epoch TERAKHIR, bukan
            # epoch terbaik. Kalau ingin best-checkpoint, tambahkan
            # torch.save di dalam blok ini.
            if auc > maxAuc:
                maxAuc = auc
                maxRmse = rmse
                maxEpoch = i+1
                maxAcc = accuracy
                maxPrec = prec
                maxRec = rec
                maxF1 = f1
            aucList.append(auc)
            lossList.append(rmse)

    # print AUC messages
    print("\n\nMAX TEST result:\n epoch:{}\nmodel:{}\nencoder:{}\nauc:{}\nacc:{}\nloss:{}\nprec:{}\nrec:{}\nf1:{}\n\n\n".format(
        maxEpoch, args.taskModel, args.encoder, maxAuc, maxAcc, maxRmse, maxPrec, maxRec, maxF1))
    # Menyimpan BOBOT saja (bukan arsitektur) -> saat load, model harus
    # dibangun ulang dengan hyperparameter yang SAMA (lihat inference.py).
    torch.save(model.state_dict(), args.model_path)
    WriteResult([aucList, lossList], ['maxAuc', 'maxRmse', 'maxEpoch', 'maxAcc', 'maxPrec',
                                      'maxRec', 'maxF1'], [maxAuc, maxRmse, maxEpoch, maxAcc, maxPrec, maxRec, maxF1])


def WriteResult(modelResult, bestResultName, bestResult):
    # Menambahkan ringkasan hasil ke result.txt (mode 'a+' = append, riwayat
    # eksperimen sebelumnya tetap tersimpan). Parameter modelResult diterima
    # tapi tidak ditulis -- sisa dari code official.
    with open(args.model_output, 'a+', encoding='utf-8-sig', newline='') as resultFile:
        tmp = "MODEL:" + args.taskModel + " ENCODER:" + args.encoder + '\n'
        resultFile.write(tmp)
        for i in range(len(bestResultName)):
            tmp = '\t' + bestResultName[i] + '_' + str(bestResult[i])
            resultFile.write(tmp)
            resultFile.write('\n')


def PrintMessage():
    '''
    @description: Print messages
    '''
    print('+-+-+-+-+-+-+-+-+-+')
    print('|run which model:', args.taskModel)
    print('|encoder:', args.encoder)
    print('|epoch:', args.epochs)
    if args.encoder == 'LSTM':
        print('|learning rate:', args.learning_rate, '(eps:', args.epsilon,
              ' grad_norm:', args.max_grad_norm, ')')
    else:
        print('|learning rate:', args.transformer_learning_rate,
              '(eps:', args.transformer_epsilon,
              ' grad_norm:', args.transformer_max_grad_norm,
              ' warmup_steps:', args.transformer_warmup_steps,
              ' seqInput:', args.transformerSeqInput, ')')
    print('|batch size:', args.batch_size)
    print('+-+-+-+-+-+-+-+-+-+')


if __name__ == "__main__":

    PrintMessage()

    if device.type == 'cuda':
        print('+-+-+-+-+-+-+-+-+-+')
        print("| RUNS ON CUDA!!! |")
        print('+-+-+-+-+-+-+-+-+-+')
    else:
        print('+-+-+-+-+-+-+-+-+-+')
        print("| RUNS ON CPU!!!  |")
        print('+-+-+-+-+-+-+-+-+-+')

    # Regenerasi data mentah kalau -regenData True (default: dilewati).
    RunPyFile()

    # Semua pekerjaan sebenarnya ada di sini.
    main()

    PrintMessage()