"""
Dataset, embedding-loading and triplet-building utilities.

The training pipeline usually consumes precomputed CLIP frame embeddings stored as
``data/video_embeddings/<artist_id>/embeddings/*.pt``. Video/CLIP extraction
dependencies are loaded lazily so importing this module for model training does
not require OpenCV or the CLIP package to be installed.
"""

from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRIPLET_COLUMNS = ("anchor", "positive", "negative")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "video_embeddings"
DEFAULT_RELATED_ARTISTS_CSV = PROJECT_ROOT / "data" / "metadata" / "related-spotify-artists.csv"


def _coerce_bool(value: Any) -> bool:
    """Convert common CSV boolean representations to real bools."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "done"}:
        return True
    if text in {"0", "false", "f", "no", "n", "", "nan", "none"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


@lru_cache(maxsize=4)
def _lazy_load_clip(device: str = DEFAULT_DEVICE):
    """Load CLIP once per device."""
    try:
        import clip  # type: ignore
    except ImportError as exc:
        raise ImportError("The 'clip' package is required only for extracting video embeddings.") from exc

    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.eval()
    return model, preprocess


def extract_embeddings(
    video_path: str | os.PathLike,
    num_frames: int = 30,
    start_ratio: float = 0.1,
    end_ratio: float = 0.9,
    device: str = DEFAULT_DEVICE,
) -> Tensor:
    """Extract fixed-length frame-level CLIP embeddings from a video.

    Returns a tensor of shape ``(num_frames, embedding_dim)``. If OpenCV fails to
    read a few selected frames, the last valid frame is repeated so downstream
    batching still receives a fixed sequence length.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not 0 <= start_ratio < end_ratio <= 1:
        raise ValueError("start_ratio and end_ratio must satisfy 0 <= start_ratio < end_ratio <= 1")

    try:
        import cv2  # type: ignore
        from PIL import Image
    except ImportError as exc:
        raise ImportError("OpenCV and Pillow are required only for extracting video embeddings.") from exc

    model, preprocess = _lazy_load_clip(device=device)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    start_frame = int(total_frames * start_ratio)
    end_frame = max(start_frame + 1, int(total_frames * end_ratio))
    selected_frames = np.linspace(start_frame, end_frame - 1, num_frames, dtype=np.int64)

    processed_frames: list[Tensor] = []
    for frame_idx in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(preprocess(Image.fromarray(frame_rgb)).unsqueeze(0))

    cap.release()

    if not processed_frames:
        raise ValueError(f"No valid frames extracted from video: {video_path}")
    while len(processed_frames) < num_frames:
        processed_frames.append(processed_frames[-1].clone())
    processed_frames = processed_frames[:num_frames]

    batch = torch.cat(processed_frames, dim=0).to(device)
    with torch.no_grad():
        embeddings = model.encode_image(batch).float().cpu()
    return embeddings


def save_embeddings(
    top_folder: str | os.PathLike,
    artists_csv_path: str | os.PathLike,
    device: str = DEFAULT_DEVICE,
    num_frames: int = 30,
) -> None:
    """Save CLIP embeddings into each artist folder.

    The artists CSV must contain ``musicbrainz_id``, ``done`` and ``extracted``.
    Rows are marked extracted only after all visible ``.mp4`` files in that artist
    folder have been processed successfully.
    """
    top_folder = Path(top_folder)
    artists_csv_path = Path(artists_csv_path)
    artists_df = pd.read_csv(artists_csv_path)
    required = {"musicbrainz_id", "done", "extracted"}
    missing = required - set(artists_df.columns)
    if missing:
        raise ValueError(f"artists CSV is missing required columns: {sorted(missing)}")

    artists_df["musicbrainz_id"] = artists_df["musicbrainz_id"].astype(str)
    valid_artist_ids = set(artists_df["musicbrainz_id"].values)

    for folder_name in sorted(os.listdir(top_folder)):
        folder_path = top_folder / folder_name
        if folder_name not in valid_artist_ids or not folder_path.is_dir():
            continue

        artist_mask = artists_df["musicbrainz_id"] == folder_name
        artist_row = artists_df.loc[artist_mask].iloc[0]
        done = _coerce_bool(artist_row["done"])
        extracted = _coerce_bool(artist_row["extracted"])
        if not (done and not extracted):
            continue

        video_files = sorted(path for path in folder_path.iterdir() if path.suffix.lower() == ".mp4")
        if not video_files:
            continue

        embeddings_folder = folder_path / "embeddings"
        embeddings_folder.mkdir(exist_ok=True)
        for video_path in video_files:
            embeddings = extract_embeddings(video_path, device=device, num_frames=num_frames)
            out_path = embeddings_folder / f"{video_path.stem}_embeddings.pt"
            torch.save(embeddings, out_path)

        artists_df.loc[artist_mask, "extracted"] = True
        artists_df.to_csv(artists_csv_path, index=False)


def _load_tensor(path: str | os.PathLike) -> Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a Tensor in {path}, got {type(tensor).__name__}")
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D embedding tensor in {path}, got shape {tuple(tensor.shape)}")
    return tensor.float().cpu()


def load_embeddings(root_dir: str | os.PathLike) -> tuple[list[Tensor], list[str]]:
    embeddings: list[Tensor] = []
    labels: list[str] = []
    root_dir = Path(root_dir)

    for class_name in sorted(os.listdir(root_dir)):
        embeddings_dir = root_dir / class_name / "embeddings"
        if not embeddings_dir.is_dir():
            continue
        for file_path in sorted(embeddings_dir.glob("*.pt")):
            try:
                embeddings.append(_load_tensor(file_path))
                labels.append(class_name)
            except Exception as exc:
                print(f"Skipping file {file_path} due to error: {exc}")
    return embeddings, labels


def load_pt_files(directory: str | os.PathLike) -> list[Tensor]:
    tensors: list[Tensor] = []
    for path in sorted(Path(directory).glob("*.pt")):
        try:
            tensors.append(_load_tensor(path))
        except Exception as exc:
            print(f"Skipping file {path} due to error: {exc}")
    return tensors


def _most_common_shape(tensors: Sequence[Tensor]) -> torch.Size | None:
    if not tensors:
        return None
    counts = Counter(tuple(tensor.shape) for tensor in tensors)
    shape_tuple, _ = max(counts.items(), key=lambda item: (item[1], item[0]))
    return torch.Size(shape_tuple)


def average_tensors(tensors: Sequence[Tensor]) -> Tensor | None:
    """Average tensors using the most common shape, ignoring malformed outliers."""
    target_shape = _most_common_shape(tensors)
    if target_shape is None:
        return None
    valid_tensors = [tensor.float() for tensor in tensors if tensor.shape == target_shape]
    if not valid_tensors:
        return None
    return torch.stack(valid_tensors, dim=0).mean(dim=0)


def filter_artists_by_common_shape(artists: dict[str, Tensor]) -> dict[str, Tensor]:
    """Keep only artists with the most common embedding shape."""
    if not artists:
        return {}
    counts = Counter(tuple(tensor.shape) for tensor in artists.values())
    target_shape, _ = max(counts.items(), key=lambda item: (item[1], item[0]))
    return {artist_id: tensor for artist_id, tensor in artists.items() if tuple(tensor.shape) == target_shape}


def infer_embedding_shape(artist_averages: dict[str, Tensor]) -> tuple[int, int]:
    if not artist_averages:
        raise ValueError("artist_averages is empty")
    first = next(iter(artist_averages.values()))
    if first.ndim != 2:
        raise ValueError(f"Expected artist embeddings to be 2D, got shape {tuple(first.shape)}")
    return int(first.shape[0]), int(first.shape[1])


def process_artists(
    base_dir: str | os.PathLike,
    *,
    keep_most_common_shape: bool = True,
) -> dict[str, Tensor]:
    artists: dict[str, Tensor] = {}
    base_dir = Path(base_dir)
    for artist_id in sorted(os.listdir(base_dir)):
        artist_dir = base_dir / artist_id / "embeddings"
        if artist_dir.is_dir():
            avg_tensor = average_tensors(load_pt_files(artist_dir))
            if avg_tensor is not None:
                artists[str(artist_id)] = avg_tensor
    return filter_artists_by_common_shape(artists) if keep_most_common_shape else artists


def prepare_data(embeddings: Sequence[Tensor], labels: Sequence[str]) -> dict[str, list[Tensor]]:
    embeddings_dict: dict[str, list[Tensor]] = defaultdict(list)
    for embedding, label in zip(embeddings, labels):
        embeddings_dict[str(label)].append(embedding.float().cpu())
    return dict(embeddings_dict)


def compute_distance(embedding1: Tensor, embedding2: Tensor) -> Tensor:
    return torch.norm(embedding1 - embedding2, p=2)


def generate_triplets(
    embeddings: Sequence[Tensor],
    labels: Sequence[str],
    csv_path: str | os.PathLike = DEFAULT_RELATED_ARTISTS_CSV,
    max_positives_per_related_artist: int = 10,
    seed: int = 3407,
) -> list[Tensor]:
    """Generate stacked triplets of shape ``(3, seq_len, embedding_dim)``."""
    if max_positives_per_related_artist <= 0:
        raise ValueError("max_positives_per_related_artist must be positive")

    rng = random.Random(seed)
    embeddings_dict = prepare_data(embeddings, labels)
    data = pd.read_csv(csv_path)
    required = {"musicbrainz_id", "id_related_artist"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"related artists CSV is missing required columns: {sorted(missing)}")
    data["musicbrainz_id"] = data["musicbrainz_id"].astype(str)
    data["id_related_artist"] = data["id_related_artist"].astype(str)

    triplets: list[Tensor] = []
    unique_labels = set(embeddings_dict)
    for label, embedding_group in embeddings_dict.items():
        related_artists = data.loc[data["musicbrainz_id"] == label, "id_related_artist"].tolist()
        related_artists_in_labels = [ra for ra in related_artists if ra in embeddings_dict]
        if not related_artists_in_labels:
            continue

        for anchor in embedding_group:
            for related_label in related_artists_in_labels:
                positive_group = embeddings_dict[related_label]
                sample_size = min(max_positives_per_related_artist, len(positive_group))
                for positive in rng.sample(positive_group, sample_size):
                    negative_candidates = sorted(unique_labels - {label, related_label})
                    if not negative_candidates:
                        continue
                    negative_label = rng.choice(negative_candidates)
                    negative = rng.choice(embeddings_dict[negative_label])
                    if anchor.shape == positive.shape == negative.shape:
                        triplets.append(torch.stack([anchor, positive, negative]).float())
    return triplets


def _normalise_triplet_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in TRIPLET_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Triplet CSV must contain columns {TRIPLET_COLUMNS}; missing {missing}")
    out = df.loc[:, list(TRIPLET_COLUMNS)].copy()
    for column in TRIPLET_COLUMNS:
        out[column] = out[column].astype(str)
    return out


def filter_triplets(df: pd.DataFrame, artist_averages: dict[str, Tensor]) -> pd.DataFrame:
    """Keep triplet rows whose three artists all have embeddings."""
    out = _normalise_triplet_df(df)
    valid_ids = {str(key) for key in artist_averages}
    mask = out["anchor"].isin(valid_ids) & out["positive"].isin(valid_ids) & out["negative"].isin(valid_ids)
    return out.loc[mask].drop_duplicates().reset_index(drop=True)


def split_triplet_dataframe_by_artist(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 3407,
    strict_artist_disjoint: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | bool | str]]:
    """
    Split a triplet dataframe into train/validation sets.

    It first tries artist-disjoint splitting, then anchor-disjoint splitting, and
    finally deterministic row-level splitting. The fallback behaviour avoids hard
    failure on sparse triplet graphs while still reporting the actual strategy.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    df = _normalise_triplet_df(df).drop_duplicates().reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least two triplets to create train/validation splits")

    rng = np.random.default_rng(seed)
    artists = sorted(set(df[list(TRIPLET_COLUMNS)].to_numpy().reshape(-1)))
    rng.shuffle(artists)
    train_artist_count = max(1, min(len(artists) - 1, int(round(train_ratio * len(artists)))))
    train_artists = set(artists[:train_artist_count])
    val_artists = set(artists[train_artist_count:])

    if strict_artist_disjoint and train_artists and val_artists:
        train_mask = df[list(TRIPLET_COLUMNS)].isin(train_artists).all(axis=1)
        val_mask = df[list(TRIPLET_COLUMNS)].isin(val_artists).all(axis=1)
        train_df = df.loc[train_mask].reset_index(drop=True)
        val_df = df.loc[val_mask].reset_index(drop=True)
        if not train_df.empty and not val_df.empty:
            return train_df, val_df, {
                "strategy": "artist_disjoint",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "train_artists": len(train_artists),
                "val_artists": len(val_artists),
                "artist_overlap": False,
            }

    anchors = sorted(df["anchor"].unique())
    rng.shuffle(anchors)
    if len(anchors) > 1:
        train_anchor_count = max(1, min(len(anchors) - 1, int(round(train_ratio * len(anchors)))))
        train_anchors = set(anchors[:train_anchor_count])
        train_df = df.loc[df["anchor"].isin(train_anchors)].reset_index(drop=True)
        val_df = df.loc[~df["anchor"].isin(train_anchors)].reset_index(drop=True)
        if not train_df.empty and not val_df.empty:
            train_all_artists = set(train_df[list(TRIPLET_COLUMNS)].to_numpy().reshape(-1))
            val_all_artists = set(val_df[list(TRIPLET_COLUMNS)].to_numpy().reshape(-1))
            return train_df, val_df, {
                "strategy": "anchor_disjoint",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "train_artists": len(train_all_artists),
                "val_artists": len(val_all_artists),
                "artist_overlap": bool(train_all_artists & val_all_artists),
            }

    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    cut = max(1, min(len(shuffled) - 1, int(round(train_ratio * len(shuffled)))))
    train_df = shuffled.iloc[:cut].reset_index(drop=True)
    val_df = shuffled.iloc[cut:].reset_index(drop=True)
    train_all_artists = set(train_df[list(TRIPLET_COLUMNS)].to_numpy().reshape(-1))
    val_all_artists = set(val_df[list(TRIPLET_COLUMNS)].to_numpy().reshape(-1))
    return train_df, val_df, {
        "strategy": "row_shuffle_fallback",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_artists": len(train_all_artists),
        "val_artists": len(val_all_artists),
        "artist_overlap": bool(train_all_artists & val_all_artists),
    }


def build_positive_map(
    triplet_df: pd.DataFrame,
    *,
    symmetric: bool = True,
) -> dict[str, set[str]]:
    """Build an anchor -> known-positive artist map from triplet rows.

    This is intentionally fold-local: callers should pass only the training
    dataframe for the current fold when using it for negative mining.
    """
    df = _normalise_triplet_df(triplet_df)
    positive_map: dict[str, set[str]] = defaultdict(set)
    for row in df.itertuples(index=False):
        anchor = str(getattr(row, "anchor"))
        positive = str(getattr(row, "positive"))
        if anchor == positive:
            continue
        positive_map[anchor].add(positive)
        if symmetric:
            positive_map[positive].add(anchor)
    return {artist_id: set(positives) for artist_id, positives in positive_map.items()}


def create_triplets(filtered_df: pd.DataFrame, artist_averages: dict[str, Tensor]) -> list[tuple[Tensor, Tensor, Tensor]]:
    triplets: list[tuple[Tensor, Tensor, Tensor]] = []
    df = _normalise_triplet_df(filtered_df)
    artists = {str(key): value for key, value in artist_averages.items()}
    for _, row in df.iterrows():
        anchor = artists.get(row["anchor"])
        positive = artists.get(row["positive"])
        negative = artists.get(row["negative"])
        if anchor is None or positive is None or negative is None:
            continue
        if anchor.shape != positive.shape or anchor.shape != negative.shape:
            continue
        triplets.append((anchor.float().cpu(), positive.float().cpu(), negative.float().cpu()))
    return triplets


def create_triplets_with_ids(
    filtered_df: pd.DataFrame,
    artist_averages: dict[str, Tensor],
) -> list[tuple[Tensor, Tensor, Tensor, str, str, str]]:
    """Create triplets that keep artist IDs for in-batch negative mining."""
    triplets: list[tuple[Tensor, Tensor, Tensor, str, str, str]] = []
    df = _normalise_triplet_df(filtered_df)
    artists = {str(key): value for key, value in artist_averages.items()}
    for _, row in df.iterrows():
        anchor_id = str(row["anchor"])
        positive_id = str(row["positive"])
        negative_id = str(row["negative"])
        anchor = artists.get(anchor_id)
        positive = artists.get(positive_id)
        negative = artists.get(negative_id)
        if anchor is None or positive is None or negative is None:
            continue
        if anchor.shape != positive.shape or anchor.shape != negative.shape:
            continue
        triplets.append(
            (
                anchor.float().cpu(),
                positive.float().cpu(),
                negative.float().cpu(),
                anchor_id,
                positive_id,
                negative_id,
            )
        )
    return triplets


class TripletDataset(Dataset):
    """Accepts tensor triplets, optionally followed by artist IDs.

    Valid items are ``(a, p, n)``, ``(a, p, n, anchor_id, positive_id,
    negative_id)``, or stacked tensors of shape ``(3, ...)``.
    """

    def __init__(self, triplets: Sequence[Any]) -> None:
        self.triplets = list(triplets)
        if not self.triplets:
            raise ValueError("triplets is empty")
        self.sample_shape = self._extract_shape(self.triplets[0])
        for idx, item in enumerate(self.triplets):
            if self._extract_shape(item) != self.sample_shape:
                raise ValueError(f"Triplet at index {idx} has inconsistent tensor shapes")

    @staticmethod
    def _extract_shape(item) -> tuple[torch.Size, torch.Size, torch.Size]:
        if isinstance(item, torch.Tensor):
            if item.ndim < 2 or item.shape[0] != 3:
                raise ValueError("Stacked triplet tensor must have shape (3, ...)")
            return item[0].shape, item[1].shape, item[2].shape
        if len(item) not in {3, 6}:
            raise ValueError("Triplet tuple must contain three tensors, optionally followed by three IDs")
        anchor, positive, negative = item[:3]
        return anchor.shape, positive.shape, negative.shape

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        item = self.triplets[idx]
        if isinstance(item, torch.Tensor):
            return item[0].float(), item[1].float(), item[2].float()
        anchor, positive, negative = item[:3]
        if len(item) == 6:
            anchor_id, positive_id, negative_id = item[3:]
            return (
                anchor.float(),
                positive.float(),
                negative.float(),
                str(anchor_id),
                str(positive_id),
                str(negative_id),
            )
        return anchor.float(), positive.float(), negative.float()


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def create_dataloaders_from_triplet_lists(
    train_triplets: Sequence[Any],
    val_triplets: Sequence[Any],
    batch_size: int = 128,
    num_workers: int = 0,
):
    train_dataset = TripletDataset(train_triplets)
    val_dataset = TripletDataset(val_triplets)
    return (
        _make_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        _make_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def create_dataloaders(
    triplets: Sequence[Any],
    batch_size: int = 128,
    train_ratio: float = 0.8,
    seed: int = 3407,
    num_workers: int = 0,
):
    """Backward-compatible row-level random split. Prefer artist-level split for final reporting."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    dataset = TripletDataset(triplets)
    if len(dataset) < 2:
        raise ValueError("Need at least two triplets to create train/validation loaders")
    train_size = max(1, min(len(dataset) - 1, int(round(train_ratio * len(dataset)))))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return (
        _make_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        _make_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
