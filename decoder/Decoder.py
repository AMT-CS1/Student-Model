# ============================================================================
# Decoder.py -- OTAK MATEMATIS HELP-DKT (Eq. 7-8 paper).
#
# Posisi dalam pipeline:
#   RunModel.py (rakit tensor) -> Encoder (h_t) -> DECODER (file ini) -> loss
#
# Tugas decoder, dua langkah:
#   (1) hidden state h_t [B,T,H]  ->  ability per KC s_t [B,T,K]   (Eq. 7)
#   (2) ability s_t + info Q-matrix soal -> p(benar) y_t [B,T,1]   (Eq. 8)
#
# Yang penting dipahami: y_t adalah PERKALIAN (torch.prod) probabilitas
# menguasai tiap konsep. Artinya "boleh dijawab benar hanya kalau SEMUA
# konsep yang dipakai soal itu dikuasai" -- satu konsep lemah sudah cukup
# menurunkan y_t. Karena perkalian, konsep yang TIDAK dipakai soal WAJIB
# dinetralkan ke ~1, kalau tidak ikut mengecilkan hasil (lihat komentar
# mode OFFICIAL di forward()).
#
# Notasi shape: B = batch, T = seq_len, H = hidden_size, K = qmatrix_size (10)
# ============================================================================
import torch
from torch import nn

# Difficulty (theta) tiap knowledge concept, NILAI sesuai paper HELP-DKT:
# co/va/op = 0.3, st/ex/li/tu = 0.4, di/cd/io = 0.5.
#
# URUTAN KOLOM problemQmatrix.CSV TIDAK sama dengan urutan penyebutan konsep
# di paper (co,va,op,st,ex,li,tu,di,cd,io). Urutan kolom di bawah dideduksi
# dengan mencocokkan baris Q-matrix terhadap Table 1 paper via mapping README
# official (C-1=362, C-2=371, C-3=406, C-4=417, C-5=449, C-6=472):
#   kol:  1   2   3   4   5   6   7   8   9   10
#   KC :  ST  VA  OP  EX  IO  LI  CO  TU  DI  CD
# - 7 kolom PASTI: ST(1), OP(3), LI(6), CO(7), TU(8), DI(9), CD(10)
#   (contoh: kol1 hanya 1 utk C-1/C-5/C-6 = tepat pola ST; kol7 = pola CO).
# - kol 2/4/5 = {VA, EX, IO}: kol5=IO (satu-satunya yg absen di 449;
#   VA/EX muncul di semua soal), kol2=VA & kol4=EX (didukung case study
#   paper: error C-4 sorting tertinggi di kol4 = expressions). Jika VA/EX
#   ternyata tertukar, dampaknya hanya swap theta 0.3<->0.4 di kol 2 & 4.
# Verifikasi konsistensi data: semua attempt 'c_' punya P-row == Q-row dan
# tak ada P>Q pada attempt 'b_' -- P-matrix & Q-matrix se-urutan kolom.
#
# CONCEPT_ORDER dipakai juga oleh inference.py untuk memberi NAMA pada tiap
# kolom ability saat menulis CSV hasil -- jangan diubah urutannya tanpa
# mengubah data, karena nama kolom akan bergeser dan hasil salah baca.
CONCEPT_ORDER = ('ST', 'VA', 'OP', 'EX', 'IO', 'LI', 'CO', 'TU', 'DI', 'CD')
# theta per konsep, urutannya SEJAJAR dengan CONCEPT_ORDER di atas.
# Hanya terpakai di MODE PAPER (lihat forward); mode OFFICIAL memakai
# problemQmatrixSub yang dikirim RunModel.py.
CONCEPT_THETA = torch.tensor([0.4, 0.3, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.5])
ALPHA = 10.0  # paper: skala agar sigmoid(alpha*(1-0.5)) ~ 0.99


class PredictKC(nn.Module):
    """
    Decoder HELP-DKT sesuai Eq. (7)-(8) paper:

        s_t = sigmoid(W * h_t)                          (cognitive layer, Eq. 7)
        w_t = (s_t - theta) (x) q_t                     (mask memilih konsep terlibat)
        y_t = prod_j sigmoid(alpha * w_tj)              (produk HANYA atas konsep terlibat)

    Theta (difficulty per konsep) dan alpha dihitung DI DALAM decoder sesuai
    paper: theta dari CONCEPT_THETA (0.3/0.4/0.5 per konsep, label pakar),
    alpha = 10. Konsep yang terlibat dipilih lewat problemQmatrixAbilityMask
    (1 terlibat, 0 tidak); konsep tak terlibat dinetralkan ke 1 sebelum
    torch.prod -- tanpa ini mereka menyumbang sigmoid(0) = 0.5 dan merusak
    probabilitas.

    problemQmatrixSub dan problemQmatrixProd: jika KEDUANYA diberikan,
    decoder berjalan MODE OFFICIAL (semantik RunModel.py/HELP_DKT_Model.py
    taskC, lihat komentar di forward); jika None, MODE PAPER (theta/alpha).

    Flag masked/subQmatrix/multiLinearLayers berupa STRING 'True'/'False'
    mengikuti argparse RunModel.py official (bukan boolean Python).
    """

    def __init__(self, hidden_size, qmatrix_size,masked='True',subQmatrix='True',multiLinearLayers='True', concept_theta=None, alpha=ALPHA):
        """
        Inisialisasi decoder: buffer theta (difficulty per konsep) dan
        cognitive layer (Linear + Sigmoid, Eq. 7).

        Args:
            hidden_size (int): Dimensi hidden state dari encoder
            qmatrix_size (int): Jumlah knowledge concepts
            masked (str): 'True'/'False' -- apakah output cognitive layer
                di-mask dengan problemQmatrixAbilityMask (string, ikut argparse
                RunModel.py official)
            subQmatrix (str): 'True'/'False' -- apakah hasil masking dikurangi
                problemQmatrixSub
            multiLinearLayers (str): 'True'/'False' -- konfigurasi output layer;
                cabang test=True hanya mengembalikan ability matrix jika 'True'
            concept_theta (array-like, optional): Difficulty per konsep sepanjang
                qmatrix_size; None = pakai CONCEPT_THETA default paper (10 KC)
            alpha (float): Faktor skala sigmoid pada Eq. (8)

        Raises:
            ValueError: Jika panjang concept_theta tidak cocok dengan qmatrix_size.
        """
        super(PredictKC, self).__init__()

        # Simpan konfigurasi. INGAT: masked/subQmatrix/multiLinearLayers adalah
        # STRING 'True'/'False', bukan bool -- jadi pengecekannya harus
        # `== 'True'`. Menulis `if self.masked:` akan SELALU benar (string
        # 'False' pun truthy) dan diam-diam mengubah perilaku model.
        self.qmatrix_size = qmatrix_size
        self.alpha = alpha
        self.masked = masked
        self.subQmatrix = subQmatrix
        self.multiLinearLayers=multiLinearLayers

        # theta: difficulty per konsep (Eq. 8), dilabel pakar di paper.
        if concept_theta is None:
            # Default hanya valid untuk 10 KC (panjang CONCEPT_THETA).
            # Kalau dataset lain punya jumlah KC berbeda -> wajib kirim
            # concept_theta sendiri.
            if qmatrix_size != CONCEPT_THETA.numel():
                raise ValueError(
                    f"qmatrix_size={qmatrix_size} tidak cocok dengan theta default "
                    f"paper ({CONCEPT_THETA.numel()} KC). Berikan concept_theta "
                    f"eksplisit sepanjang {qmatrix_size}."
                )
            concept_theta = CONCEPT_THETA
        else:
            # as_tensor menerima list/ndarray/tensor; dipaksa float32 agar
            # tipenya seragam dengan bobot model.
            concept_theta = torch.as_tensor(concept_theta, dtype=torch.float32)
            if concept_theta.numel() != qmatrix_size:
                raise ValueError(
                    f"concept_theta punya {concept_theta.numel()} elemen, "
                    f"harus sama dengan qmatrix_size={qmatrix_size}"
                )
        # buffer: ikut pindah device bersama model, bukan trainable parameter
        # .clone() supaya tidak berbagi memori dengan konstanta global
        # CONCEPT_THETA (kalau di-share, satu model bisa merusak model lain).
        self.register_buffer('theta', concept_theta.clone())

        # COGNITIVE LAYER (Eq. 7): h_t [B,T,H] -> s_t [B,T,K] dalam (0,1).
        # Inilah satu-satunya bobot terlatih di decoder. Output-nya yang
        # kemudian dilaporkan sebagai "student ability per konsep".
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_size, qmatrix_size),
            nn.Sigmoid()
        )
        # Sigmoid murni tanpa bobot. Dibungkus nn.Sequential semata-mata agar
        # struktur & nama state_dict-nya identik dengan HELP_DKT_Model official
        # (supaya checkpoint lama tetap bisa dimuat).
        self.decoder2 = nn.Sequential(
            nn.Sigmoid()
        )

        self.init_weights()

    def forward(self, encoded_output, Qmatrix, problemQmatrix, problemQmatrixSub,
        problemQmatrixAbilityMask, problemQmatrixProd,test=False, **kwargs):
        # CATATAN: parameter `Qmatrix` dan `problemQmatrix` diterima demi
        # kompatibilitas signature dengan official, tapi TIDAK dipakai di
        # badan fungsi ini. Yang benar-benar menentukan hasil adalah
        # AbilityMask, Sub, dan Prod. **kwargs menyerap argumen ekstra.

        # === Eq. (7) === h_t -> ability mentah per konsep.
        #   encoded_output [B, T, H] -> decoded1 [B, T, K], tiap nilai di (0,1)
        decoded1 = self.decoder1(encoded_output)
        # inisiasi saja

        # `ability` menyimpan REFERENSI ke ability SEBELUM di-mask/dikurangi.
        # Inilah angka yang dilaporkan ke CSV hasil (interpretabilitas paper).
        # Aman karena baris-baris di bawah memakai torch.mul/torch.sub yang
        # menghasilkan tensor BARU (bukan operasi in-place), sehingga `ability`
        # tidak ikut berubah.
        ability = decoded1

        # masking: pilih konsep yang terlibat (identik official)
        # AbilityMask [B,T,K] berisi 0/1 dari baris Q-matrix soal:
        # konsep yang tidak dipakai soal -> dinolkan dulu di sini.
        if self.masked == 'True':
            decoded1 = torch.mul(decoded1, problemQmatrixAbilityMask)

        # Dua mode, dipilih otomatis dari argumen yang diberikan:
        #   OFFICIAL (Sub & Prod != None) -- persis HELP_DKT_Model.py taskC:
        #       y = prod sigmoid((s*q - Sub) * Prod)
        #       Sub : terlibat k/10 (k = jml konsep soal), tak terlibat -0.5
        #       Prod: terlibat 5/(1-k/10),                tak terlibat 10
        #       Konsep tak terlibat: sigmoid((0-(-0.5))*10) ~ 0.993 (~netral)
        #       Catatan: INI yang menghasilkan angka paper, meski teks paper
        #       menyebut theta pakar 0.3/0.4/0.5 -- kode official tidak
        #       pernah memakai theta pakar.
        #   PAPER (Sub/Prod None) -- Eq. (7)-(8) literal:
        #       y = prod sigmoid(alpha*(s - theta)) atas konsep terlibat,
        #       theta pakar per KC, konsep tak terlibat dinetralkan ke 1.
        #
        # PERINGATAN untuk pengembang berikutnya: cabang MODE PAPER BELUM
        # diimplementasikan di code ini -- kalau Sub/Prod dikirim None,
        # variabel `decoded` tidak pernah dibuat dan baris `return decoded`
        # akan melempar UnboundLocalError. Jalur yang dipakai RunModel.py &
        # inference.py selalu mengirim Sub & Prod (mode OFFICIAL), jadi tidak
        # terjadi di pemakaian normal. Kalau ingin mode PAPER, tambahkan
        # cabang `else:` di sini memakai self.theta dan self.alpha.
        if problemQmatrixSub is not None and problemQmatrixProd is not None:
            # Kurangi ambang kesulitan: nilai positif = "di atas ambang".
            # Untuk konsep tak terlibat (nilainya sudah 0) hasilnya +0.5.
            if self.subQmatrix == 'True':
                decoded1 = torch.sub(decoded1, problemQmatrixSub)
            # Kalikan skala (alpha efektif) lalu sigmoid -> probabilitas
            # menguasai tiap konsep. probs [B, T, K].
            probs = self.decoder2(torch.mul(decoded1, problemQmatrixProd))
            # === Eq. (8) === perkalian sepanjang sumbu konsep (dim=2).
            # keepdim=True menjaga bentuk [B, T, 1] supaya cocok dengan
            # target_correctness setelah .view(-1) di RunModel.py.
            decoded = torch.prod(probs, dim=2, keepdim=True)

        if test == False:
            # ini kalau Training
            # decoded akan digunakan untuk loss
            # Hanya prediksi yang dikembalikan -> nn.BCELoss di RunModel.py.
            return decoded
        else:
            # ini kalau test true

            # buat matriks all 1
            # tmp [B, T, K] dipakai sebagai penyaring baris untuk pelaporan.
            tmp = torch.ones_like(problemQmatrixAbilityMask)

            # Mengecek konsep yang tidak digunakan
            # Jika suatu timestep
            # tidak memiliki konsep
            # supaya nanti tidak divisualisasikan.
            #
            # Timestep tanpa satu pun konsep = PADDING (siswa punya attempt
            # lebih sedikit dari T). Barisnya dinolkan supaya `any(...)` di
            # RunModel.addVecMess melewatinya dan tidak ikut ditulis ke CSV.
            # Loop Python bersarang ini lambat, tapi hanya jalan saat evaluasi
            # (bukan saat training) -- kalau perlu dipercepat, versi vektor:
            # tmp = problemQmatrixAbilityMask.amax(dim=2, keepdim=True)
            #         .eq(1).float().expand_as(tmp)
            for i in range(problemQmatrixAbilityMask.size()[0]):     # loop batch
                for j in range(problemQmatrixAbilityMask.size()[1]): # loop timestep
                    if 1 not in problemQmatrixAbilityMask[i, j, :]:
                        tmp[i, j, :] = 0
            if self.multiLinearLayers == 'True':
                # apa yang direturn
                # decoded
                # hidden layer
                # student ability yang
                # dikalikan keperluan/ ability yang ada
                # dan dikembalikan ke dalam list
                #
                # ability.mul(tmp) -> [B,T,K] dengan baris padding = 0.
                # .tolist() memindahkan ke Python list (lepas dari GPU/graph)
                # karena hasilnya hanya untuk ditulis ke CSV.
                # CATATAN: kalau multiLinearLayers != 'True', fungsi ini jatuh
                # ke akhir tanpa return -> None, dan HelpDKT_Model.forward akan
                # gagal saat membongkar (decoded, ability). Jalur taskC selalu
                # memakai 'True', jadi aman di pemakaian normal.
                return decoded, ability.mul(tmp).tolist()

    def init_weights(self):
        """Inisialisasi bobot cognitive layer: weight uniform (-0.05, 0.05)
        sesuai paper, bias nol."""
        # Rentang kecil & simetris -> sigmoid mulai dekat 0.5 (netral), tidak
        # langsung jenuh di 0/1 yang membuat gradien mati sejak epoch pertama.
        # decoder2 tidak diinisialisasi karena memang tidak punya bobot.
        for name, param in self.decoder1.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.05, 0.05)
            elif 'bias' in name:
                nn.init.zeros_(param)