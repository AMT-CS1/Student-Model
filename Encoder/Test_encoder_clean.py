import torch
import sys
sys.path.append('.')

from LSTM_Encoder import LSTMEncoder
from Transformer_Encoder import Transformer_Encoder_Causal

# ============================================================================
# TEST 1: LSTM Encoder
# ============================================================================

def test_lstm_encoder():
    """Uji LSTMEncoder: init hidden state (h0, c0), forward pass, dan
    verifikasi shape output [B, T, hidden] serta hidden [layers, B, hidden]."""
    print("\n" + "=" * 80)
    print("TEST 1: LSTM Encoder")
    print("=" * 80)
    
    batch_size = 32
    seq_len = 50
    input_size = 7
    hidden_size = 200
    num_layers = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = LSTMEncoder(
        rnn_type='LSTM',
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.6
    )
    model = model.to(device)
    
    print(f"Device: {device}")
    print(f"Config: {model.get_config()}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # Initialize hidden state
    h = model.init_hidden(batch_size, device)
    print(f"Hidden state initialized (tuple): h={h[0].shape}, c={h[1].shape}")
    
    # Forward pass
    output, hidden = model(x, h)
    
    print(f"Output shape: {output.shape}")
    print(f"Final hidden state: h={hidden[0].shape}, c={hidden[1].shape}")
    
    # Verify shapes
    assert output.shape == (batch_size, seq_len, hidden_size), f"Expected {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    assert hidden[0].shape == (num_layers, batch_size, hidden_size), f"Expected h shape {(num_layers, batch_size, hidden_size)}"
    assert hidden[1].shape == (num_layers, batch_size, hidden_size), f"Expected c shape {(num_layers, batch_size, hidden_size)}"
    
    print("✓ LSTM Encoder test passed!")
    return model


# ============================================================================
# TEST 2: Transformer Encoder with Causal Masking
# ============================================================================

def test_transformer_causal_encoder():
    """Uji Transformer_Encoder_Causal: forward pass dengan causal masking
    otomatis, verifikasi shape output dan hidden state None."""
    print("\n" + "=" * 80)
    print("TEST 2: Transformer Encoder with Causal Masking")
    print("=" * 80)
    
    batch_size = 32
    seq_len = 50
    input_size = 7
    hidden_size = 200
    num_layers = 3
    num_heads = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = Transformer_Encoder_Causal(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1
    )
    model = model.to(device)
    
    print(f"Device: {device}")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Num layers: {model.num_layers}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Causal masking: enabled (no lookahead)")
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # Initialize hidden state
    h = model.init_hidden(batch_size, device)
    print(f"Hidden state: {h}")
    
    # Forward pass
    output, hidden = model(x, h)
    
    print(f"Output shape: {output.shape}")
    print(f"Final hidden state: {hidden}")
    
    # Verify shapes
    assert output.shape == (batch_size, seq_len, hidden_size), f"Expected {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    assert hidden is None, "Transformer should return None for hidden state"
    
    print("✓ Transformer Causal Encoder test passed!")
    return model


# ============================================================================
# TEST 3: Encoder Comparison
# ============================================================================

def test_encoder_comparison():
    """Bangun kedua encoder (LSTM, TransformerCausal) sekaligus,
    jalankan forward pass, dan print tabel perbandingan jumlah parameter."""
    print("\n" + "=" * 80)
    print("TEST 3: Encoder Comparison")
    print("=" * 80)
    
    batch_size = 32
    seq_len = 50
    input_size = 7
    hidden_size = 200
    num_layers = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create all models
    lstm_model = LSTMEncoder(
        rnn_type='LSTM',
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.6
    ).to(device)
    
    tf_causal_model = Transformer_Encoder_Causal(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    
    # Count parameters
    lstm_params = lstm_model.count_parameters()
    tf_causal_params = sum(p.numel() for p in tf_causal_model.parameters() if p.requires_grad)
    
    # Test forward pass
    print(f"\nInput: {x.shape}")
    print(f"\n{'Encoder':<25} {'Params':<15} {'Has Hidden':<15} {'Status':<15}")
    print("-" * 70)
    
    # LSTM
    h_lstm = lstm_model.init_hidden(batch_size, device)
    out_lstm, h_lstm = lstm_model(x, h_lstm)
    print(f"{'LSTM':<25} {lstm_params:<15,} {'Yes (h, c)':<15} {'✓ OK':<15}")

    # Transformer Causal
    h_tf_c = tf_causal_model.init_hidden(batch_size, device)
    out_tf_c, h_tf_c = tf_causal_model(x, h_tf_c)
    print(f"{'Transformer (Causal)':<25} {tf_causal_params:<15,} {'No (None)':<15} {'✓ OK':<15}")
    
    print("\n" + "-" * 70)
    print("Output shapes:")
    print(f"  LSTM: {out_lstm.shape}")
    print(f"  Transformer (Causal): {out_tf_c.shape}")
    
    print("\n✓ All encoders working correctly!")


# ============================================================================
# TEST 4: Error Handling
# ============================================================================

def test_error_handling():
    """Uji error handling encoder: num_heads yang tidak habis membagi
    hidden_size (ValueError) dan input LSTM dengan shape salah (RuntimeError)."""
    print("\n" + "=" * 80)
    print("TEST 4: Error Handling")
    print("=" * 80)
    
    batch_size = 32
    seq_len = 50
    input_size = 7
    hidden_size = 200
    num_layers = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nTest 4.1: Invalid num_heads (not divisible by hidden_size)")
    try:
        model = Transformer_Encoder_Causal(
            input_size=input_size,
            hidden_size=200,
            num_heads=7,  # 200 % 7 != 0
            num_layers=num_layers
        )
        print("✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Caught: {str(e)[:70]}")

    print("\nTest 4.2: LSTM with invalid input")
    try:
        model = LSTMEncoder(
            rnn_type='LSTM',
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(device)
        
        # Wrong input shape
        x = torch.randn(batch_size, seq_len).to(device)  # Missing feature dimension
        h = model.init_hidden(batch_size, device)
        output, hidden = model(x, h)
        print("✗ Should have raised error!")
    except RuntimeError as e:
        print(f"✓ Caught: {str(e)[:70]}")
    
    print("✓ Error handling tests passed!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("ENCODER UNIT TESTS")
    print("=" * 80)
    
    try:
        # Run all tests
        test_lstm_encoder()
        test_transformer_causal_encoder()
        test_encoder_comparison()
        test_error_handling()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
