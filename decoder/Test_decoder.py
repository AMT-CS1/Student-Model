import torch
from Decoder import PredictKC, CONCEPT_THETA


def test_predictKC():
    """Uji decoder PredictKC sesuai Eq. (7)-(8) paper: theta dan alpha
    dihitung di dalam decoder, konsep terlibat dipilih lewat ability mask.
    Verifikasi shape, rentang probabilitas, dan netralisasi konsep tak
    terlibat."""
    print("=" * 60)
    print("Testing predictKC Decoder")
    print("=" * 60)

    batch_size = 32
    seq_len = 50
    hidden_size = 200
    qmatrix_size = 10  # sesuai paper: 10 knowledge concepts

    decoder = PredictKC(hidden_size, qmatrix_size)

    encoded_output = torch.randn(batch_size, seq_len, hidden_size)

    # Q-matrix per timestep: 1 kalau konsep terlibat, 0 kalau tidak
    Qmatrix_input = torch.randint(
        0, 2, (batch_size, seq_len, qmatrix_size)).float()
    ability_mask = Qmatrix_input

    problemQmatrix = torch.randint(0, 2, (1, qmatrix_size))

    print(f"""
    qinput {Qmatrix_input},
    abilitiMask {ability_mask},
    theta {decoder.theta},
    alpha {decoder.alpha}
""")

    # Training mode -- problemQmatrixSub/Prod tidak dipakai lagi
    # (theta & alpha internal sesuai paper), None untuk kompatibilitas signature
    prediction = decoder(
        encoded_output,
        Qmatrix=Qmatrix_input,
        problemQmatrix=problemQmatrix,
        problemQmatrixSub=None,
        problemQmatrixAbilityMask=ability_mask,
        problemQmatrixProd=None,
        test=False
    )

    print(f"Train output     : {prediction.shape}")

    assert prediction.shape == (batch_size, seq_len, 1)
    assert (prediction >= 0).all() and (prediction <= 1).all(), \
        "prediksi harus probabilitas [0, 1]"

    # timestep tanpa konsep: semua elemen netral 1 -> produknya harus 1
    no_concept = ability_mask.sum(dim=2) == 0
    if no_concept.any():
        assert torch.allclose(
            prediction.squeeze(-1)[no_concept],
            torch.ones(int(no_concept.sum()))
        ), "timestep tanpa konsep terlibat harus menghasilkan produk netral 1"

    # Test mode
    prediction, ability_matrix = decoder(
        encoded_output,
        Qmatrix=Qmatrix_input,
        problemQmatrix=problemQmatrix,
        problemQmatrixSub=None,
        problemQmatrixAbilityMask=ability_mask,
        problemQmatrixProd=None,
        test=True
    )

    ability_matrix = torch.tensor(ability_matrix)

    print(f"Test output      : {prediction.shape}")
    print(f"Ability matrix   : {ability_matrix.shape}")

    assert prediction.shape == (batch_size, seq_len, 1)
    assert ability_matrix.shape == (
        batch_size,
        seq_len,
        qmatrix_size,
    )

    # timestep tanpa konsep sama sekali harus 0 semua (tidak divisualisasikan)
    if no_concept.any():
        assert (ability_matrix[no_concept] == 0).all(), \
            "timestep tanpa konsep harus 0 di ability matrix"

    print("[OK] TaskC passed\n")


def test_theta_alpha():
    """Uji numerik Eq. (8): dengan satu konsep terlibat, output harus persis
    sigmoid(alpha * (s - theta)) untuk konsep itu."""
    print("=" * 60)
    print("Testing theta & alpha (Eq. 8)")
    print("=" * 60)

    hidden_size = 16
    qmatrix_size = 10
    decoder = PredictKC(hidden_size, qmatrix_size)
    decoder.eval()

    encoded_output = torch.randn(1, 1, hidden_size)

    # hanya konsep ke-0 (theta = 0.3) yang terlibat
    mask = torch.zeros(1, 1, qmatrix_size)
    mask[0, 0, 0] = 1.0

    with torch.no_grad():
        s = decoder.decoder1(encoded_output)  # Eq. (7)
        expected = torch.sigmoid(
            decoder.alpha * (s[0, 0, 0] - CONCEPT_THETA[0]))

        prediction = decoder(
            encoded_output,
            Qmatrix=mask,
            problemQmatrix=mask[0],
            problemQmatrixSub=None,
            problemQmatrixAbilityMask=mask,
            problemQmatrixProd=None,
            test=False
        )

    assert torch.allclose(prediction[0, 0, 0], expected, atol=1e-6), \
        f"output {prediction[0, 0, 0]} != sigmoid(alpha*(s-theta)) {expected}"
    print(f"[OK] sigmoid(alpha*(s-theta)) = {expected.item():.6f} cocok\n")


def test_error_handling():
    """Uji error handling decoder: forward tanpa tensor Q-matrix harus
    TypeError (argumen wajib), dan qmatrix_size yang tidak cocok dengan theta
    default paper harus ValueError."""
    print("=" * 60)
    print("Testing Error Handling")
    print("=" * 60)

    batch_size = 8
    seq_len = 10
    hidden_size = 64
    qmatrix_size = 10  # sesuai paper: 10 knowledge concepts

    encoded_output = torch.randn(batch_size, seq_len, hidden_size)

    decoder = PredictKC(hidden_size, qmatrix_size)

    # forward tanpa argumen Q-matrix -> TypeError (missing positional args)
    try:
        decoder(encoded_output)
        raise AssertionError("[FAIL] Should raise TypeError")
    except TypeError as e:
        print(f"[OK] Caught expected error: {e}")

    # qmatrix_size != 10 tanpa concept_theta eksplisit -> ValueError
    try:
        PredictKC(hidden_size, 5)
        raise AssertionError("[FAIL] Should raise ValueError")
    except ValueError as e:
        print(f"[OK] Caught expected error: {e}")


if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("HELP-DKT Decoder Unit Test")
    print("=" * 80)

    test_predictKC()
    test_theta_alpha()
    test_error_handling()

    print("=" * 80)
    print("All decoder tests passed successfully!")
    print("=" * 80)
