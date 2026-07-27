# ============================================================================
# LSTM_Encoder.py -- ENCODER default HELP-DKT (jalur identik paper/official).
#
# Perannya dalam pipeline:
#   RunModel.py merakit tensor input  ->  ENCODER (file ini)  ->  Decoder.py
#
# Yang dikerjakan file ini HANYA satu: mengubah urutan attempt seorang siswa
# menjadi hidden state h_t yang merangkum "riwayat belajar sampai attempt t".
# Semua rumus paper (ability per KC, probabilitas benar) ada di Decoder.py.
#
# Notasi shape yang dipakai di komentar:
#   B = batch size, T = seq_len (jumlah attempt), D = input_size (20),
#   H = hidden_size (200), L = num_layers (3)
# ============================================================================
import torch
from torch import nn

class LSTMEncoder(nn.Module):
    """
    Encoder berbasis RNN (LSTM/GRU) untuk HELP-DKT.

    Meng-encode sequence attempt mahasiswa [batch, seq_len, input_size]
    menjadi hidden representation [batch, seq_len, hidden_size] yang
    kemudian dikonsumsi decoder (PredictKC/PredictMc).
    """

    def __init__(self, rnn_type, input_size, hidden_size = 200, num_layers= 3, dropout=0.6):
        """
            Inisialisasi RNN encoder (LSTM atau GRU) + dropout layer.

            Args:
                rnn_type (str): 'LSTM' atau 'GRU'
                input_size (int): Dimensi input feature
                hidden_size (int): Dimensi hidden state
                num_layers (int): Jumlah layer RNN
                dropout (float): Dropout rate antar layer
        """
        # Registrasi nn.Module (wajib sebelum menambah submodule/parameter).
        super(LSTMEncoder, self).__init__()
        
        # Disimpan untuk dipakai lagi di init_hidden() & get_config().
        # rnn_type: 'LSTM' -> hidden = (h, c); 'GRU' -> hidden = h saja.
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        # H = 200
        self.num_layers = num_layers
        # L = 3
        
        # Inisialisasi RNN sesuai tipe
        # getattr(nn, 'LSTM') == nn.LSTM. Trik ini membuat tipe RNN bisa
        # dipilih lewat string dari argparse tanpa if/else berlapis.
        self.rnn = getattr(nn, rnn_type)(
            # input size sesuai input
            # D -- HARUS sama dengan lebar tensor yang dikirim RunModel.py,
            # kalau tidak PyTorch langsung error saat forward.
            input_size=input_size,
            # Hidden Size 200
            # H -- sekaligus jadi lebar input decoder (PredictKC.hidden_size).
            hidden_size=hidden_size,
            # layer 3
            # L -- LSTM bertumpuk: output layer ke-n jadi input layer ke-(n+1).
            num_layers=num_layers,
            # args untuk batch pertama
            # batch_first=True -> tensor berbentuk [B, T, ...], bukan [T, B, ...].
            # Konvensi ini dipakai konsisten di SELURUH proyek.
            batch_first=True,
            # drop out
            # PyTorch hanya menerapkan dropout ANTAR layer, jadi kalau cuma
            # 1 layer nilainya WAJIB 0 (kalau tidak, PyTorch memberi warning
            # dan dropout-nya memang tidak berefek apa-apa).
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass encoder
        
        Args:
            input_seq (Tensor): Shape [batch_size, seq_len, input_size]
            hidden_state (Tensor, optional): Initial hidden state
        
        Returns:
            output (Tensor): Shape [batch_size, seq_len, hidden_size]
            hidden_state (Tensor/tuple): Final hidden state(s)
        """
        # initiate LSTM dan
        # (tanpa dropout tambahan pada output -- official HELP-DKT hanya
        # memakai dropout antar-layer bawaan nn.LSTM, tidak men-dropout
        # output terakhir sebelum decoder)
        #
        # SATU-SATUNYA baris komputasi di encoder ini:
        #   input_seq [B, T, D] + hidden_state (h, c) [L, B, H]
        #   -> output [B, T, H] = hidden state layer TERAKHIR di SETIAP t
        #   -> hidden_state (h, c) = kondisi akhir di t = T-1 saja
        # hidden_state=None berarti 'mulai dari nol' (PyTorch mengisi zeros).
        # RunModel.py sengaja MEMBAWA hidden antar batch (stateful) dan
        # men-detach-nya lewat RepackageHidden agar backprop tidak menjalar
        # ke batch sebelumnya.
        output, hidden_state = self.rnn(input_seq, hidden_state)

        # mengembalikan output dan nilai hidden statenya
        # output -> ke decoder; hidden_state -> dipakai batch berikutnya.
        return output, hidden_state

    def init_hidden(self, batch_size, device=None):
        """
        Inisialisasi hidden state awal (zeros)
        
        Args:
            batch_size (int): Ukuran batch
            device (torch.device, optional): Device placement
        
        Returns:
            hidden_state (Tensor or tuple): 
                - LSTM: tuple (h0, c0)
                - GRU: Tensor h0
        """
        # cek device kalau GPU di masukkan ke GPU untuk perhitungannya
        # Ambil satu parameter model (mis. weight_ih_l0) hanya untuk MEMBACA
        # device-nya. Cara ini otomatis benar setelah model.to('cuda').
        # Catatan: argumen `device` dari pemanggil sengaja ditimpa di baris
        # berikutnya -- device selalu mengikuti device model.
        weight = next(self.parameters())
        device = weight.device
        
        if self.rnn_type == 'LSTM':
            # initiate bobot hidden H dan information C
            # h0 = hidden state (memori jangka pendek), c0 = cell state
            # (memori jangka panjang). Keduanya [L, B, H] -- perhatikan L di
            # depan, BUKAN batch_first (aturan PyTorch untuk hidden state).
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h0, c0)
        else:  # GRU
            # GRU tidak punya cell state -> cukup satu tensor h0 [L, B, H].
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
    def get_config(self):
        """
        Return model configuration untuk logging/debugging
        
        Returns:
            dict: Model hyperparameters
        """
        # Dipakai saat mencatat eksperimen; tidak memengaruhi komputasi.
        return {
            'model_type': self.rnn_type,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
        }
    
    def count_parameters(self):
        """
        Hitung total trainable parameters
        
        Returns:
            int: Total number of trainable parameters
        """
        # numel() = jumlah elemen tensor; hanya yang requires_grad yang dilatih.
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
