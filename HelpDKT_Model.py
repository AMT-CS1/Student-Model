"""
Model gabungan Encoder + Decoder untuk HELP-DKT (PredictKC).

API dibuat IDENTIK dengan HELP_DKT_Model.py official (LSTM-only) supaya
drop-in replacement di training loop (RunModel.py):

    - __init__(rnn_type, args, num_skills, timeSteps, dropout=0.6,
      tie_weights=False)               -> signature sama persis
    - forward(input, hidden, Qmatrix, problemQmatrixMask, problemQmatrixSub,
      problemQmatrixAbilityMask, problemQmatrixProd, test=False)
        train -> (decoded, hidden)
        test  -> (decoded, hidden, ability_list)
    - init_hidden(bsz)                 -> LSTM: (h0, c0); Transformer: None

Perbedaan dengan official hanya di DALAM: RNN + decoder inline diganti
komponen modular (Encoder/LSTM_Encoder.py atau Transformer_Encoder_Causal
dan decoder/Decoder.py PredictKC). rnn_type='LSTM' memakai LSTMEncoder
(perilaku == official taskC); 'TransformerCausal' menukar encoder saja
tanpa mengubah training loop.
"""
# ============================================================================
# CATATAN UNTUK PENGEMBANG BERIKUTNYA (komentar tambahan, code TIDAK diubah)
# ============================================================================
# Notasi shape yang dipakai di seluruh komentar file ini:
#   B = batch size          (args.batch_size, default 32)
#   T = timeSteps/num_steps (jumlah attempt maksimum satu siswa, mis. 57)
#   D = args.input_size     (default 20 = 10 dim 'salah' + 10 dim 'benar')
#   H = args.hidden_size    (default 200)
#   K = args.Qmatrix_size   (default 10 = jumlah knowledge concept / KC)
#
# Alur besar satu forward pass:
#   input [B,T,D_efektif] --(encoder)--> output [B,T,H] --(decoder)--> [B,T,1]
#   di mana [B,T,1] = probabilitas siswa BENAR pada attempt berikutnya.
#
# File ini sengaja "tipis": semua logika matematis paper ada di
# decoder/Decoder.py (Eq. 7-8), semua logika sequence ada di Encoder/.
# File ini hanya PERAKIT + penjaga kompatibilitas API dengan code official.
# ============================================================================
import torch
from torch import nn

# Dua pilihan encoder. Keduanya WAJIB punya API sama: forward(input, hidden)
# -> (output, hidden) dan init_hidden(bsz). Itulah yang membuat encoder bisa
# ditukar tanpa menyentuh RunModel.py sedikit pun.
from Encoder.LSTM_Encoder import LSTMEncoder
from Encoder.Transformer_Encoder import Transformer_Encoder_Causal
# Decoder tunggal (dipakai kedua encoder): mengubah hidden state -> ability
# per KC -> probabilitas benar. Lihat Decoder.py untuk rumus Eq. (7)-(8).
from decoder.Decoder import PredictKC


class HelpDKT_Model(nn.Module):
    """HELP-DKT model (encoder-decoder), API kompatibel HELP_DKT_Model."""

    # Daftar putih tipe encoder. Dipakai juga oleh RunModel.py & inference.py
    # sebagai `choices` argparse, jadi menambah encoder baru cukup:
    # (1) tambahkan namanya di tuple ini, (2) tambahkan cabang di __init__.
    ENCODER_TYPES = ('LSTM', 'TransformerCausal')

    # Signature SAMA dengan official: (rnn_type, args, num_skills, timeSteps,
    # dropout=0.6, tie_weights=False). num_skills & tie_weights diterima demi
    # kompatibilitas walau (seperti di official) tidak dipakai.
    def __init__(self, rnn_type, args, num_skills, timeSteps, dropout=0.6,
                 tie_weights=False):
        # Wajib dipanggil lebih dulu supaya nn.Module bisa mendaftarkan
        # submodule/parameter (self.encoder, self.decoder) di bawahnya.
        super(HelpDKT_Model, self).__init__()

        # initialize dropout (parity dengan official; di official pun tidak
        # dipakai di forward -- dropout efektif hanya antar-layer nn.LSTM)
        # Jangan dihapus: atribut ini ada di state_dict lama (checkpoint).
        self.drop = nn.Dropout(dropout)

        # Validasi awal: gagal cepat dengan pesan jelas kalau -encoder salah
        # ketik, daripada error aneh jauh di dalam getattr(nn, ...).
        if rnn_type not in self.ENCODER_TYPES:
            raise ValueError(
                f"rnn_type '{rnn_type}' tidak dikenal, "
                f"pilih salah satu dari {self.ENCODER_TYPES}"
            )

        # ------------------------------------------------------------------
        # Encoder: dipilih lewat rnn_type, API forward/init_hidden sama
        # ------------------------------------------------------------------
        if rnn_type == 'LSTM':
            # Jalur OFFICIAL. input_size = args.input_size (20) karena
            # RunModel.py mengirim hasil matmul(QmatrixInput^T, input_data)
            # yang lebarnya tetap D=20.
            # Output: [B, T, H]; hidden: tuple (h, c) masing-masing
            # [num_layers, B, H].
            self.encoder = LSTMEncoder(
                rnn_type=rnn_type,           # 'LSTM' (LSTMEncoder juga dukung 'GRU')
                input_size=args.input_size,  # D = 20
                hidden_size=args.hidden_size,      # H = 200
                num_layers=args.hidden_layer_num,  # 3 layer bertumpuk
                dropout=dropout,             # dropout ANTAR layer LSTM
            )
        else:  # TransformerCausal
            # Skema input BARU (lihat RunModel.py/inference.py): kalau
            # transformerSeqInput=='True' (default), input yang dikirim ke
            # encoder ini adalah embedding per-step berurutan waktu DIGABUNG
            # (concat di dimensi fitur) dengan P-matrix per-step
            # (Qmatrix_size kolom), bukan hasil matmul agregat spt LSTM --
            # jadi input_size Transformer LEBIH BESAR drpd args.input_size.
            # LSTM di atas sama sekali tidak tersentuh oleh cabang ini.
            #
            # getattr(..., 'True') dipakai (bukan args.transformerSeqInput
            # langsung) supaya model tetap bisa dibuat dari objek args yang
            # tidak punya flag ini -- mis. script lama atau unit test.
            transformer_seq_input = getattr(
                args, 'transformerSeqInput', 'True') == 'True'
            # Lebar input yang HARUS cocok dengan tensor rakitan RunModel.py:
            #   seqInput True  -> D + K   (20 + 10 = 30)  [concat]
            #   seqInput False -> D       (20)            [matmul agregat lama]
            # Kalau angka ini salah, load_state_dict checkpoint akan gagal di
            # input_projection.weight karena shape-nya beda.
            transformer_input_size = (
                args.input_size + args.Qmatrix_size
                if transformer_seq_input else args.input_size)
            self.encoder = Transformer_Encoder_Causal(
                input_size=transformer_input_size,   # 30 (default) atau 20
                hidden_size=args.hidden_size,        # H = 200 (d_model)
                num_layers=args.hidden_layer_num,    # jumlah blok transformer
                # num_heads: H harus habis dibagi angka ini (200 % 8 == 0 OK)
                num_heads=getattr(args, 'num_heads', 8),
                # dropout Transformer dipisah dari dropout LSTM (0.1 vs 0.6)
                dropout=getattr(args, 'transformer_dropout', 0.1),
                # tabel positional encoding di-precompute sepanjang ini; ambil
                # minimal 1000 agar aman kalau T bertambah di dataset lain.
                max_seq_len=max(timeSteps, 1000),
            )

        # ------------------------------------------------------------------
        # Decoder: PredictKC (Eq. 7-8 paper). Dengan Sub & Prod dari data
        # (bukan None) jalurnya identik official taskC.
        # ------------------------------------------------------------------
        # Decoder SAMA untuk kedua encoder -> perbandingan LSTM vs Transformer
        # jadi adil (yang berbeda hanya cara meng-encode urutan waktu).
        self.decoder = PredictKC(
            hidden_size=args.hidden_size,    # H, harus == output encoder
            qmatrix_size=args.Qmatrix_size,  # K, jumlah KC yang diprediksi
            masked=args.masked,              # str 'True'/'False' (ikut argparse)
            subQmatrix=args.subQmatrix,      # str 'True'/'False'
            multiLinearLayers=args.multiLinearLayers,  # str 'True'/'False'
        )

        # Atribut default persis official (dipakai training loop / logging)
        # Disimpan sebagai atribut biasa (bukan parameter) -> tidak ikut
        # ter-training, tapi berguna untuk logging & pengecekan konfigurasi.
        self.rnn_type = rnn_type                    # 'LSTM' / 'TransformerCausal'
        self.nhid = args.hidden_size                # H
        self.nlayers = args.hidden_layer_num        # jumlah layer encoder
        self.timeSteps = timeSteps                  # T
        self.multiLinearLayers = args.multiLinearLayers
        self.masked = args.masked
        self.QmatrixSize = args.Qmatrix_size        # K
        self.subQmatrix = args.subQmatrix
        # parity atribut official (dipakai training loop / logging official;
        # model modular ini setara jalur taskC)
        # getattr dengan default -> aman dipanggil dari script yang args-nya
        # tidak selengkap RunModel.py (mis. inference.py atau smoke test).
        self.taskModel = getattr(args, 'taskModel', 'taskC')
        self.linearWithQmatrix = getattr(args, 'linearWithQmatrix', 'False')

    def forward(self, input, hidden, Qmatrix, problemQmatrixMask,
                problemQmatrixSub, problemQmatrixAbilityMask,
                problemQmatrixProd, test=False):
        """
        Forward pass, signature & return IDENTIK HELP_DKT_Model.forward.

        Args:
            input (Tensor): [B, T, input_size]
            hidden: hidden state awal (LSTM: tuple (h, c) hasil init_hidden /
                batch sebelumnya; Transformer: diabaikan)
            Qmatrix, problemQmatrixMask, problemQmatrixSub,
            problemQmatrixAbilityMask, problemQmatrixProd:
                diteruskan ke decoder persis semantik official taskC
            test (bool): False = training, True = evaluasi (+ ability matrix)

        Returns:
            test=False -> (decoded [B, T, 1], hidden)
            test=True  -> (decoded [B, T, 1], hidden, ability_list)
        """
        # hasil hidden state ht (sama dengan output, hidden = self.rnn(...))
        # LANGKAH 1 -- ENCODE.
        #   input  [B, T, D_efektif]  ->  output [B, T, H]
        #   LSTM             : hidden = (h, c) [nlayers, B, H], dipakai lagi
        #                      di batch berikutnya (stateful antar batch).
        #   TransformerCausal: hidden = None (stateless); urutan waktu
        #                      ditangani positional encoding + causal mask.
        output, hidden = self.encoder(input, hidden)

        # LANGKAH 2 -- DECODE.
        # Semua tensor Q/P-matrix diteruskan apa adanya; decoder yang memutuskan
        # mana yang dipakai (lihat Decoder.py: mode OFFICIAL vs mode PAPER).
        #   test=False -> result = decoded              [B, T, 1]
        #   test=True  -> result = (decoded, ability)   ability: list [B][T][K]
        # Perhatikan pemetaan nama argumen: problemQmatrixMask di sini masuk
        # ke parameter bernama `problemQmatrix` milik decoder.
        result = self.decoder(
            output,                                            # [B, T, H]
            Qmatrix=Qmatrix,                                   # [B, T, T] (P-matrix + padding)
            problemQmatrix=problemQmatrixMask,                 # [B, T, K] 0/1
            problemQmatrixSub=problemQmatrixSub,               # [B, T, K] theta efektif
            problemQmatrixAbilityMask=problemQmatrixAbilityMask,  # [B, T, K] 0/1
            problemQmatrixProd=problemQmatrixProd,             # [B, T, K] skala alpha
            test=test,
        )

        # LANGKAH 3 -- kembalikan dalam BENTUK yang diharapkan RunModel.py.
        # `== False` dipertahankan apa adanya (official) walau `not test`
        # lebih idiomatis -- jangan diubah supaya diff dengan official minimal.
        if test == False:
            # training: decoded dipakai untuk loss
            # RunModel.py akan .view(-1) lalu torch.gather posisi target.
            return result, hidden
        else:
            # test: ability matrix ikut keluar (posisi return sama official:
            # decoded, hidden, ability)
            # Decoder mengembalikan tuple, dibongkar di sini supaya urutan
            # return-nya (decoded, hidden, ability) sama persis official.
            decoded, ability = result
            return decoded, hidden, ability

    # initialize hidden state menjadi 0 (delegasi ke encoder)
    def init_hidden(self, bsz):
        """LSTM -> (h0, c0) [nlayers, bsz, nhid]; Transformer -> None."""
        # Sengaja mendelegasi ke encoder: model utama tidak perlu tahu
        # encoder-nya stateful atau tidak. Dipanggil sekali di awal RunEpoch
        # (dan setiap batch di inference.py).
        return self.encoder.init_hidden(bsz)

    def get_config(self):
        """Return konfigurasi model untuk logging/debugging."""
        # Berguna saat membandingkan hasil eksperimen: simpan dict ini bersama
        # checkpoint agar jelas model dilatih dengan konfigurasi apa.
        return {
            'rnn_type': self.rnn_type,
            'hidden_size': self.nhid,
            'num_layers': self.nlayers,
            'qmatrix_size': self.QmatrixSize,
            'masked': self.masked,
            'subQmatrix': self.subQmatrix,
            'multiLinearLayers': self.multiLinearLayers,
        }

    def count_parameters(self):
        """Hitung total trainable parameters (encoder + decoder)."""
        # requires_grad=True saja yang dihitung -> buffer seperti `theta` dan
        # `pe` (positional encoding) TIDAK ikut, karena bukan bobot terlatih.
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# if __name__ == '__main__':
#     # ------------------------------------------------------------------
#     # Smoke test deterministik: args gaya argparse RunModel.py official,
#     # cek signature __init__/forward/init_hidden identik dengan
#     # HELP_DKT_Model dan shape output benar untuk kedua encoder.
#     # ------------------------------------------------------------------
#     from types import SimpleNamespace

#     n = 10                       # jumlah KC (Qmatrix_size paper)
#     args = SimpleNamespace(
#         input_size=2 * n,        # Eq. 4: setengah benar + setengah salah
#         hidden_size=200,
#         hidden_layer_num=3,
#         Qmatrix_size=n,
#         masked='True',
#         subQmatrix='True',
#         multiLinearLayers='True',
#     )
#     batch, timeSteps = 2, 5

#     # mask konsep terlibat: kolom 0-2
#     mask = torch.zeros(batch, timeSteps, n)
#     mask[:, :, 0:3] = 1.0

#     for enc in HelpDKT_Model.ENCODER_TYPES:
#         torch.manual_seed(42)
#         model = HelpDKT_Model(enc, args, num_skills=n, timeSteps=timeSteps)
#         model.eval()

#         # Lebar input HARUS cocok dgn encoder yang dibangun: LSTM tetap
#         # args.input_size (20); TransformerCausal (skema baru) = input_size
#         # + Qmatrix_size (30) karena P-matrix per-step ikut digabung --
#         # lihat perhitungan transformer_input_size di __init__ di atas.
#         in_width = model.encoder.base_encoder.input_projection.in_features \
#             if enc != 'LSTM' else args.input_size
#         x = torch.zeros(batch, timeSteps, in_width)
#         x[:, :, 0:3] = 0.5       # bobot konsep ST/VA/OP di setengah 'benar'

#         with torch.no_grad():
#             hidden = model.init_hidden(batch)

#             # train mode: (decoded, hidden) -- persis official
#             out, hidden = model(x, hidden, Qmatrix=mask,
#                                 problemQmatrixMask=mask,
#                                 problemQmatrixSub=None,
#                                 problemQmatrixAbilityMask=mask,
#                                 problemQmatrixProd=None, test=False)

#             # test mode: (decoded, hidden, ability) -- persis official
#             out_t, hidden_t, ability = model(x, model.init_hidden(batch),
#                                              Qmatrix=mask,
#                                              problemQmatrixMask=mask,
#                                              problemQmatrixSub=None,
#                                              problemQmatrixAbilityMask=mask,
#                                              problemQmatrixProd=None,
#                                              test=True)

#         assert out.shape == (batch, timeSteps, 1), out.shape
#         assert (out >= 0).all() and (out <= 1).all()
#         assert torch.tensor(ability).shape == (batch, timeSteps, n)
#         if enc == 'LSTM':
#             assert isinstance(hidden, tuple) and \
#                 hidden[0].shape == (args.hidden_layer_num, batch,
#                                     args.hidden_size)
#         else:
#             assert hidden is None

#         print(f"=== {enc}: OK | params={model.count_parameters():,} | "
#               f"y[0] = {[round(v, 4) for v in out[0, :, 0].tolist()]}")
