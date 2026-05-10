from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from dataset import (
    build_negative_exclusion_map,
    create_artist_memory_bank,
    create_dataloaders_from_triplet_lists,
    create_triplets_with_ids,
)
from train import train


class TinyTripletNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.view(x.size(0), -1))

    def forward(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor):
        return self.forward_once(a), self.forward_once(p), self.forward_once(n)


def test_negative_exclusion_map_masks_two_hop_neighbours() -> None:
    df = pd.DataFrame(
        [
            {"anchor": "a", "positive": "p", "negative": "x"},
            {"anchor": "p", "positive": "q", "negative": "y"},
        ]
    )
    exclusion = build_negative_exclusion_map(df, symmetric=True, include_two_hop=True)
    assert "p" in exclusion["a"]
    assert "q" in exclusion["a"]


def test_memory_bank_semihard_train_step() -> None:
    df = pd.DataFrame(
        [
            {"anchor": "a", "positive": "p", "negative": "n1"},
            {"anchor": "p", "positive": "a", "negative": "n2"},
        ]
    )
    artists = {
        "a": torch.tensor([[1.0, 0.0]]),
        "p": torch.tensor([[0.95, 0.05]]),
        "n1": torch.tensor([[0.2, 0.8]]),
        "n2": torch.tensor([[0.0, 1.0]]),
    }
    triplets = create_triplets_with_ids(df, artists)
    train_loader, _ = create_dataloaders_from_triplet_lists(triplets, triplets, batch_size=2)
    memory_ids, memory_tensors = create_artist_memory_bank(df, artists)
    exclusion = build_negative_exclusion_map(df, symmetric=True, include_two_hop=False)

    model = TinyTripletNet()
    criterion = nn.TripletMarginLoss(margin=0.2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = train(
        model,
        train_loader,
        criterion,
        optimizer,
        "cpu",
        return_details=True,
        negative_mining="memory_bank_semihard",
        positive_map=exclusion,
        memory_bank_ids=memory_ids,
        memory_bank_tensors=memory_tensors,
        memory_bank_batch_size=4,
    )

    assert metrics["negative_mining"] == "memory_bank_semihard"
    assert metrics["memory_bank_size"] == 4
    assert 0.0 <= metrics["ranking_acc"] <= 1.0
    assert metrics["skipped_ratio"] < 1.0


if __name__ == "__main__":
    test_negative_exclusion_map_masks_two_hop_neighbours()
    test_memory_bank_semihard_train_step()
    print("All memory-bank mining sanity checks passed.")
