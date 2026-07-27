"""
Test ekuivalensi: HelpDKT_Model (encoder-decoder modular, rnn_type='LSTM')
vs HELP_DKT_Model official (LSTM-only, taskC, multiLinearLayers='True').

Kedua model dibangun dengan seed sama -> urutan draw RNG identik
(nn.LSTM lalu Linear decoder1 lalu re-init uniform), sehingga bobot awal
harus sama bit-per-bit dan output forward harus torch.equal.

Jalankan:  python Test_equivalence.py
"""
import torch
from types import SimpleNamespace

from HELP_DKT_Model import HELP_DKT_Model      # original
from HelpDKT_Model import HelpDKT_Model        # modular (encoder + decoder)

# ---------------------------------------------------------------- konfigurasi
n = 10                                          # jumlah KC
args = SimpleNamespace(
    input_size=2 * n,
    hidden_size=200,
    hidden_layer_num=3,
    Qmatrix_size=n,
    taskModel='taskC',
    masked='True',
    subQmatrix='True',
    multiLinearLayers='True',
    linearWithQmatrix='False',
)
batch, timeSteps, seed = 4, 6, 42

# ------------------------------------------------------------------ input
torch.manual_seed(0)
x = torch.rand(batch, timeSteps, args.input_size)

# mask konsep terlibat: kolom 0-2 (k = 3 konsep)
k = 3
mask = torch.zeros(batch, timeSteps, n)
mask[:, :, 0:k] = 1.0

# Sub/Prod gaya official: terlibat k/10 & 5/(1-k/10), tak terlibat -0.5 & 10
Sub = torch.where(mask == 1, torch.tensor(k / 10), torch.tensor(-0.5))
Prod = torch.where(mask == 1, torch.tensor(5 / (1 - k / 10)), torch.tensor(10.0))

# ------------------------------------------------------------------ build
torch.manual_seed(seed)
official = HELP_DKT_Model('LSTM', args, num_skills=n, timeSteps=timeSteps)
torch.manual_seed(seed)
modular = HelpDKT_Model('LSTM', args, num_skills=n, timeSteps=timeSteps)
official.eval()
modular.eval()

# cek bobot awal identik (nama param beda prefix, bandingkan nilainya)
off_params = [p for _, p in sorted(official.named_parameters())]
mod_params = [p for _, p in sorted(modular.named_parameters())]
assert len(off_params) == len(mod_params), (
    f"jumlah parameter beda: {len(off_params)} vs {len(mod_params)}")
for po, pm in zip(off_params, mod_params):
    assert torch.equal(po, pm), "bobot awal TIDAK identik"
print(f"[OK] bobot awal identik "
      f"({sum(p.numel() for p in off_params):,} params)")

# ------------------------------------------------------------------ forward
with torch.no_grad():
    h1 = official.init_hidden(batch)
    h2 = modular.init_hidden(batch)
    assert torch.equal(h1[0], h2[0]) and torch.equal(h1[1], h2[1])

    # train mode
    y1, h1 = official(x, h1, mask, mask, Sub, mask, Prod, test=False)
    y2, h2 = modular(x, h2, mask, mask, Sub, mask, Prod, test=False)
    assert y1.shape == y2.shape == (batch, timeSteps, 1)
    assert torch.equal(y1, y2), (
        f"output train BEDA, max|diff|={(y1 - y2).abs().max().item():.3e}")
    assert torch.equal(h1[0], h2[0]) and torch.equal(h1[1], h2[1])
    print(f"[OK] train : decoded & hidden identik "
          f"(y[0] = {[round(v, 4) for v in y1[0, :, 0].tolist()]})")

    # test mode (+ ability matrix)
    y1t, _, a1 = official(x, official.init_hidden(batch),
                          mask, mask, Sub, mask, Prod, test=True)
    y2t, _, a2 = modular(x, modular.init_hidden(batch),
                         mask, mask, Sub, mask, Prod, test=True)
    assert torch.equal(y1t, y2t)
    assert a1 == a2, "ability matrix BEDA"
    print("[OK] test  : decoded & ability matrix identik")

print("\nSEMUA PASS -- HelpDKT_Model (LSTM) ekuivalen bit-per-bit "
      "dengan HELP_DKT_Model official (taskC).")
