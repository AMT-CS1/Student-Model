# ============================================================================
# Transformer_Encoder.py -- ENCODER alternatif (pengganti LSTM) untuk HELP-DKT.
#
# Isi file, 3 class:
#   1. PositionalEncoding         -> menyuntikkan informasi URUTAN waktu
#   2. Transformer_Encoder        -> blok transformer dasar (INTERNAL saja)
#   3. Transformer_Encoder_Causal -> pembungkus #2 + causal mask  <-- INI yang
#                                    dipakai HelpDKT_Model
#
# Kenapa butuh causal mask? Self-attention secara default melihat SELURUH
# sequence sekaligus, termasuk attempt masa depan. Untuk knowledge tracing itu
# kebocoran label (model "mengintip" jawaban berikutnya) -> AUC terlihat bagus
# tapi tidak valid. Causal mask memaksa posisi t hanya melihat step 0..t.
#
# Notasi shape: B = batch, T = seq_len, D = input_size, H = hidden_size (d_model)
# ============================================================================
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding untuk Transformer
    Menambahkan informasi posisi ke input sequence
    """
    
    def __init__(self, d_model, max_seq_len=1000, dropout=0.1):
        """
        Pre-compute matriks sinusoidal positional encoding sekali di awal
        (disimpan sebagai buffer, bukan parameter trainable).

        Args:
            d_model (int): Dimensi embedding (harus sama dengan hidden_size)
            max_seq_len (int): Panjang sequence maksimum yang didukung
            dropout (float): Dropout rate setelah penjumlahan PE
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Buat positional encoding matrix
        # pe [max_seq_len, d_model] -- dihitung SEKALI di __init__, bukan tiap
        # forward, karena isinya konstan (tidak bergantung data).
        pe = torch.zeros(max_seq_len, d_model)
        # position [max_seq_len, 1] = [[0], [1], [2], ...] -- indeks waktu.
        # unsqueeze(1) menyiapkan broadcasting terhadap div_term di bawah.
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Dimension index untuk sin/cos
        # div_term [d_model/2] = 1 / 10000^(2i/d_model), ditulis lewat exp+log
        # supaya stabil secara numerik. Efeknya: tiap pasangan dimensi punya
        # frekuensi gelombang berbeda -> posisi unik & bisa diekstrapolasi.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            -(math.log(10000.0) / d_model)
        )
        
        # Apply sin ke even indices, cos ke odd indices
        # Kolom genap pakai sin, kolom ganjil pakai cos (rumus asli paper
        # "Attention is All You Need").
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # d_model ganjil: kolom ganjil jumlahnya satu lebih sedikit, jadi
            # div_term dipotong ([:-1]) agar shape-nya cocok.
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register sebagai buffer (bukan parameter)
        # buffer = ikut pindah device & ikut tersimpan di state_dict, TAPI
        # tidak dilatih (tidak muncul di optimizer). unsqueeze(0) menambah
        # sumbu batch -> [1, max_seq_len, d_model] agar bisa di-broadcast.
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Tambahkan positional encoding ke input lalu terapkan dropout.

        Args:
            x (Tensor): Shape [batch, seq_len, d_model]

        Returns:
            Tensor: Input dengan informasi posisi, shape sama [batch, seq_len, d_model].
        """
        # Potong pe sepanjang T aktual lalu jumlahkan (broadcast di sumbu batch):
        #   x [B, T, H] + pe [1, T, H] -> [B, T, H]
        # Kalau T > max_seq_len, baris ini error -> naikkan max_seq_len.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Transformer_Encoder(nn.Module):
    """
    Base Transformer Encoder -- INTERNAL, jangan dipakai langsung.

    Class ini hanya building block untuk Transformer_Encoder_Causal
    (dipakai sebagai base_encoder + causal mask). Tanpa causal mask,
    self-attention bisa melihat attempt masa depan -- tidak valid untuk
    Task C yang memprediksi attempt t+1 dari t attempt sebelumnya.
    Gunakan Transformer_Encoder_Causal untuk semua eksperimen.

    Args:
        input_size (int): Dimensi input feature
        hidden_size (int): Dimensi embedding/hidden state (d_model)
        num_layers (int): Jumlah transformer layer
        num_heads (int): Jumlah attention heads
        ff_dim (int): Dimensi feed-forward inner layer
        dropout (float): Dropout rate
        max_seq_len (int): Maximum sequence length
    """
    
    def __init__(
        self,
        input_size,
        hidden_size = 200,
        num_layers = 3,
        num_heads = 8,
        ff_dim=None,
        dropout=0.1,
        max_seq_len=1000
    ):
        """
        Inisialisasi komponen encoder: input projection (input_size ->
        hidden_size), positional encoding, dan stack TransformerEncoderLayer.
        Deskripsi tiap argumen ada di docstring class di atas.

        Raises:
            ValueError: Jika hidden_size tidak habis dibagi num_heads.
        """
        super(Transformer_Encoder, self).__init__()

        # hidden size
        # H = d_model. Dipakai lagi untuk skala sqrt(H) di forward.
        self.hidden_size = hidden_size
        # layers untuk encoder
        self.num_layers = num_layers
        
        # Validate attention heads
        # karena hidden size harus bisa dibagi head
        # head menyimpan informasi suatu blok
        # Tiap head mendapat H/num_heads dimensi. Kalau tidak habis dibagi,
        # PyTorch error di dalam MultiheadAttention -- dicegat lebih awal di
        # sini agar pesannya jelas. (200 / 8 = 25 -> OK)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        
        # Default feed-forward dimension
        if ff_dim is None:
            # Konvensi paper asli: lebar FFN = 4x d_model (200 -> 800).
            ff_dim = hidden_size * 4
        
        # Input projection (map input_size -> hidden_size)
        # Transformer tidak punya 'input gate' seperti LSTM; fitur mentah
        # (D=30 pada skema seqInput) diproyeksikan dulu ke ruang d_model (H=200).
        # CATATAN: in_features layer inilah yang harus cocok dengan checkpoint
        # saat load_state_dict -- penyebab error paling sering kalau flag
        # -transformerSeqInput berbeda antara training dan inference.
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        # Wajib: tanpa ini transformer buta urutan (permutation invariant).
        self.pos_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout)
        
        # inisiasi Transformer encoder layers
        # SATU blok = MultiheadAttention + FFN + residual + LayerNorm.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            # H
            nhead=num_heads,
            # jumlah head attention paralel
            dim_feedforward=ff_dim,
            # lebar FFN internal (4H)
            dropout=dropout,
            batch_first=True,  # [batch, seq_len, d_model]
            activation='relu'
        )
        
        # layer yang digunakan untuk attentionnya
        # disusun dari beberapa blok encoder
        # inisiasi encoder terdiri dari beberapa layer encoder
        # nn.TransformerEncoder menyalin `encoder_layer` sebanyak num_layers
        # (deep copy -- bobot TIDAK dibagi antar layer).
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size)
            # LayerNorm akhir, setelah semua blok
        )

    def forward(self, input_seq, hidden_state=None, src_mask=None,
                src_key_padding_mask=None):
        """
        Forward pass transformer encoder

        Args:
            input_seq (Tensor): Shape [batch, seq_len, input_size]
            hidden_state (Tensor, optional): Ignored for Transformer
                (kept for API compatibility with LSTM)
            src_mask (Tensor, optional): Attention mask for masking positions
                Shape: [seq_len, seq_len], True = mask out (konvensi PyTorch)
            src_key_padding_mask (Tensor, optional): Shape [batch, seq_len],
                True = timestep padding (tidak boleh di-attend). Tanpa ini,
                varian non-causal akan ikut meng-attend timestep padding.

        Returns:
            output (Tensor): Shape [batch, seq_len, hidden_size]
            hidden_state (None): Transformer tidak return hidden state tradisional
                Return None untuk compatibility
        """
        # Project input to hidden_size
        # [batch, seq_len, input_size] -> [batch, seq_len, hidden_size]
        # Dikali sqrt(d_model) (konvensi "Attention is All You Need") agar
        # magnitude embedding sebanding dengan positional encoding dan
        # informasi input tidak tenggelam oleh PE.
        # Perhatikan: `hidden_state` sengaja TIDAK dipakai -- parameternya ada
        # semata-mata agar signature-nya sama dengan LSTMEncoder.forward.
        # [B, T, D] -> [B, T, H]
        x = self.input_projection(input_seq) * math.sqrt(self.hidden_size)

        # Add positional encoding untuk interpretasi waktu
        # (dropout sudah diterapkan di dalam PositionalEncoding dan di tiap
        # TransformerEncoderLayer -- tidak perlu dropout ekstra di sini)
        x = self.pos_encoding(x)

        # Forward through transformer
        # Attention mask handling:
        # - Jika ingin causal masking (tidak boleh attend future),
        # gunakan src_mask atau generate_square_subsequent_mask
        # di parameter forward
        # Dua jenis mask dengan peran berbeda:
        #   src_mask             [T, T] -> aturan WAKTU (sama untuk semua siswa)
        #   src_key_padding_mask [B, T] -> menandai timestep KOSONG per siswa
        # Shape tidak berubah: [B, T, H] masuk, [B, T, H] keluar.
        output = self.transformer_encoder(
            x, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )

        # Transformer tidak punya hidden state tradisional seperti LSTM
        # Return None untuk compatibility dengan init_hidden()
        # None = hidden state. Karena None, RunModel.RepackageHidden harus
        # tahan menerima None (lihat penanganannya di sana).
        return output, None
    
    def generate_square_subsequent_mask(self, seq_len, device):
        """
        Generate causal mask (triangular mask) untuk mencegah
        position t attend ke position > t

        Konvensi PyTorch untuk boolean attention mask:
        True = posisi DIBLOKIR (tidak boleh di-attend), False = boleh.
        triu dengan diagonal=1 menghasilkan True tepat di posisi masa depan,
        jadi mask ini langsung dipakai TANPA inversi. (Inversi `~mask` pada
        versi sebelumnya membuat model justru hanya bisa attend ke masa
        depan/anti-kausal dan posisi terakhir menghasilkan NaN.)

        Args:
            seq_len (int): Sequence length
            device (torch.device): Device untuk mask

        Returns:
            mask (Tensor): Shape [seq_len, seq_len]
                True = mask out (masa depan), False = attend
        """
        # torch.triu(..., diagonal=1) = segitiga ATAS tanpa diagonal utama.
        # Contoh seq_len=4 (True = diblokir):
        #   t=0: [F T T T]   -> hanya boleh melihat dirinya sendiri
        #   t=1: [F F T T]
        #   t=2: [F F F T]
        #   t=3: [F F F F]   -> boleh melihat seluruh masa lalu
        # JANGAN menambahkan `~` di sini (lihat catatan bug di docstring).
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
    
    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden state (compatibility dengan LSTM)
        
        Transformer tidak menggunakan hidden state tradisional.
        Method ini return None karena Transformer stateless.
        
        Args:
            batch_size (int): Ignored
            device (torch.device, optional): Ignored
        
        Returns:
            None: Transformer tidak perlu hidden state
        """
        return None


class Transformer_Encoder_Causal(nn.Module):
    """
    Transformer Encoder dengan Causal Masking (untuk next-token prediction)
    
    Memastikan setiap position hanya bisa attend ke position sebelumnya,
    simulasi autoregressive decoding.
    
    Berguna untuk:
    - Task C: predict next problem without lookahead bias
    - Temporal consistency dalam learning path
    """
    
    def __init__(
        self,
        input_size,
        hidden_size=200,
        num_layers=3,
        num_heads=8,
        ff_dim=None,
        dropout=0.1,
        max_seq_len=1000
    ):
        """
        Inisialisasi causal encoder. Membungkus Transformer_Encoder biasa;
        satu-satunya perbedaan ada di forward() yang selalu memasang
        causal mask otomatis.

        Args: sama persis dengan Transformer_Encoder.
        """
        super(Transformer_Encoder_Causal, self).__init__()
        # inisiasi untuk hidden size dan layer 
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # menggunakan base model saja yang sudah di buat untuk mempermudah
        # perbedaan terjadi hanya pada masking 
        # Pola komposisi (bukan pewarisan): semua bobot sebetulnya milik
        # base_encoder -> itulah sebabnya key state_dict mengandung
        # 'base_encoder.', dan inference.DetectEncoder memakainya untuk
        # menebak tipe encoder langsung dari file checkpoint.
        self.base_encoder = Transformer_Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
    
    def forward(self, input_seq, hidden_state=None, src_key_padding_mask=None):
        """
        Forward dengan automatic causal masking

        Args:
            input_seq (Tensor): Shape [batch, seq_len, input_size]
            hidden_state (Tensor, optional): Ignored
            src_key_padding_mask (Tensor, optional): Shape [batch, seq_len],
                True = timestep padding

        Returns:
            output (Tensor): Shape [batch, seq_len, hidden_size]
            hidden_state (None)
        """
        # T diambil dari data (bukan konstanta) -> mask selalu pas walau
        # batch terakhir lebih pendek.
        seq_len = input_seq.size(1)
        # Perhitungan masuk ke gpu
        # Mask HARUS dibuat di device yang sama dengan input, kalau tidak
        # PyTorch melempar error 'expected all tensors on same device'.
        device = input_seq.device

        # Generate causal mask (True = masa depan, diblokir)
        # Dibuat ulang tiap forward (murah: [T, T] boolean) supaya panjang
        # sequence yang berubah-ubah tetap tertangani.
        causal_mask = self.base_encoder.generate_square_subsequent_mask(
            seq_len, device
        )

        # Forward dengan mask
        # INILAH satu-satunya perbedaan dengan Transformer_Encoder biasa:
        # src_mask selalu diisi causal_mask, tidak pernah None.
        output, _ = self.base_encoder(
            input_seq,
            src_mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # None mengikuti kontrak API bersama LSTMEncoder.forward.
        return output, None
    
    def init_hidden(self, batch_size, device=None):
        """
        Kompatibilitas API dengan LSTM -- Transformer stateless.

        Returns:
            None: Tidak ada hidden state yang perlu diinisialisasi.
        """
        return None