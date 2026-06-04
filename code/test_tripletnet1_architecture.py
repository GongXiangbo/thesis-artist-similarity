from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dataset import infer_embedding_shape, stack_tensors
from model import TripletNet1


def test_tripletnet1_hierarchical_output_shape_and_normalization() -> None:
    model = TripletNet1(d_model=64, seq_len=6, max_videos=4, output_dim=16, video_dropout_p=0.0, model_dim=32).eval()
    x = torch.randn(2, 4, 6, 64)
    x[:, 2:] = 0.0  # padded videos must be masked

    with torch.inference_mode():
        out = model.forward_once(x)

    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_tripletnet1_keeps_backward_compatible_single_video_input() -> None:
    model = TripletNet1(d_model=64, seq_len=6, max_videos=4, output_dim=16, video_dropout_p=0.0, model_dim=32).eval()
    x = torch.randn(2, 6, 64)

    with torch.inference_mode():
        out = model.forward_once(x)

    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_tripletnet1_masks_zero_padded_videos() -> None:
    model = TripletNet1(d_model=64, seq_len=6, max_videos=4, output_dim=16, video_dropout_p=0.0, model_dim=32).eval()
    single_video = torch.randn(2, 6, 64)
    stacked = torch.zeros(2, 4, 6, 64)
    stacked[:, 0] = single_video

    with torch.inference_mode():
        old_format_out = model.forward_once(single_video)
        stacked_out = model.forward_once(stacked)

    assert torch.allclose(old_format_out, stacked_out, atol=1e-5)


def test_tripletnet1_video_dropout_keeps_at_least_one_valid_video() -> None:
    model = TripletNet1(d_model=64, seq_len=6, max_videos=4, output_dim=16, video_dropout_p=0.95, model_dim=32).train()
    x = torch.randn(2, 4, 6, 64)
    x[:, 3] = 0.0

    with torch.inference_mode():
        out = model.forward_once(x)

    assert out.shape == (2, 16)
    assert model.last_video_mask is not None
    assert model.last_video_mask.any(dim=1).all()


def test_tripletnet1_rejects_invalid_sequence_video_and_feature_shapes() -> None:
    model = TripletNet1(d_model=64, seq_len=6, max_videos=4, output_dim=16, model_dim=32).eval()

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model.forward_once(torch.randn(1, 7, 64))

    with pytest.raises(ValueError, match="exceeds max_videos"):
        model.forward_once(torch.randn(1, 5, 6, 64))

    with pytest.raises(ValueError, match="does not match d_model"):
        model.forward_once(torch.randn(1, 4, 6, 63))


def test_stack_tensors_zero_pads_without_repetition() -> None:
    tensors = [torch.ones(6, 64), 2.0 * torch.ones(6, 64)]
    stacked = stack_tensors(tensors, max_videos=4, num_frames=6)

    assert stacked is not None
    assert stacked.shape == (4, 6, 64)
    assert torch.allclose(stacked[0], torch.ones(6, 64))
    assert torch.allclose(stacked[1], 2.0 * torch.ones(6, 64))
    assert torch.count_nonzero(stacked[2:]) == 0
    assert infer_embedding_shape({"artist": stacked}) == (6, 64)
