from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model import TripletNet1


def test_tripletnet1_output_shape_and_normalization() -> None:
    model = TripletNet1(d_model=768, seq_len=30, output_dim=256).eval()
    x = torch.randn(2, 30, 768)

    with torch.inference_mode():
        out = model.forward_once(x)

    assert out.shape == (2, 256)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_tripletnet1_handles_single_frame_delta_case() -> None:
    model = TripletNet1(d_model=768, seq_len=1, output_dim=256).eval()
    x = torch.randn(2, 1, 768)

    with torch.inference_mode():
        out = model.forward_once(x)

    assert out.shape == (2, 256)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_tripletnet1_rejects_invalid_sequence_and_feature_shapes() -> None:
    model = TripletNet1(d_model=768, seq_len=2, output_dim=256).eval()

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model.forward_once(torch.randn(1, 3, 768))

    with pytest.raises(ValueError, match="does not match d_model"):
        model.forward_once(torch.randn(1, 2, 767))
