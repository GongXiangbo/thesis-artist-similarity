# Code README

`code/` 目录包含本项目的全部训练、评估、notebook 实验和测试代码。代码围绕 triplet learning 组织：从本地预计算 CLIP frame embeddings 构造 artist tensor，再用 TripletNet 编码 artist，最后用 triplet loss 与 retrieval-style metrics 评估 artist similarity。

## 快速入口

从仓库根目录运行当前推荐配置：

```bash
python code/experiment.py \
  --model TripletNet1 \
  --artist-aggregation stack \
  --videos-per-artist 10 \
  --num-frames 30 \
  --video-dropout 0.15 \
  --margin 0.1 \
  --epochs 30
```

运行 Conv1D baseline 时使用 mean aggregation：

```bash
python code/experiment.py \
  --model TripletNet2 \
  --artist-aggregation mean \
  --margin 0.3 \
  --epochs 30
```

运行单元测试：

```bash
cd code
pytest -q
```

上面两个训练命令显式传入的是当前 margin grid 中的中间值 `0.3`。notebook 会完整 sweep `[0.1, 0.3, 0.5, 0.7, 0.9]`，并按 validation retrieval MRR 选择最佳 margin。

## 输入数据约定

训练默认读取：

```text
../data/video_embeddings/<artist_id>/embeddings/*.pt
../data/triplets/triplets_ids_music_spot.csv
```

在 notebook 中，工作目录通常是 `code/`，因此 notebook 使用 `Path("../data/video_embeddings")` 和 `Path("../data/triplets/triplets_ids_music_spot.csv")`。在 CLI 中，`experiment.py` 会把相对路径解析到项目根目录。

metadata 分析额外读取：

```text
../data/metadata/*.csv
```

`model1_posthoc_metrics_metadata.ipynb` 会从项目根目录的 `data/metadata/` 查找 metadata；当前保存输出使用 `artists_genre_country.csv`，并补充 `artists.csv` 中的 artist name。

每个视频 embedding 是 `(frames, dim)`，当前实验使用 `(30, 768)`。`dataset.process_artists` 支持两种 artist 聚合方式：

| aggregation | 输出形状 | 适用模型 | 说明 |
| --- | --- | --- | --- |
| `stack` | `(max_videos, frames, dim)`，默认 `(10, 30, 768)` | `TripletNet1` | 保留一个 artist 的多个视频，视频不足时用全零 tensor padding |
| `mean` | `(frames, dim)`，默认 `(30, 768)` | `TripletNet2` 到 `TripletNet4` | 将一个 artist 的多个视频 tensor 按视频维平均，兼容 legacy Conv1D baselines |

## Python 模块说明

### `model.py`

定义所有 artist encoder。

`TripletNet1` 是当前注册到 `MODEL_REGISTRY` 的默认模型，接受 `(batch, videos, frames, dim)` 或向后兼容的 `(batch, frames, dim)`：

1. 对 CLIP frame embedding 做 `LayerNorm -> Linear(768, model_dim) -> LayerNorm -> Dropout`。
2. 加 learned positional encoding。
3. 用 frame-level `TransformerEncoder` 编码每个视频的 30 个 frame。
4. 对 frame token 做 mean pooling 得到每个视频的 video token。
5. 根据全零 padded video 推断 valid-video mask，并在训练时应用 video dropout。
6. 用 masked artist-level `TransformerEncoder` 编码最多 10 个 video token。
7. 对有效 video token 做 masked mean pooling。
8. 经过 projection head、`SafeBatchNorm1d` BNNeck 和 L2 normalization 输出 256 维 artist embedding。

`TripletNet2` 到 `TripletNet4` 都继承 `ConvTripletNet`，输入是 legacy `(batch, seq_len, d_model)`：

| 模型 | Conv1D channels | kernel sizes | stride/pool 设计 | projection |
| --- | --- | --- | --- | --- |
| `TripletNet2` | `256, 256, 128` | `3, 3, 3` | 第 0、1 层后 max-pool | hidden `256` -> output `256` |
| `TripletNet3` | `256, 256, 128, 128` | `5, 5, 3, 3` | 第 0、2 层后 max-pool | hidden `256` -> output `256` |
| `TripletNet4` | `256, 256, 128, 128` | `3, 3, 3, 3` | stride `1, 2, 1, 2`，不额外 pool | hidden `256` -> output `256` |

`model.py` 还保留了一个更复杂的 `HierarchicalVideoArtistTripletNet`，包含 frame set branch、temporal branch、delta branch、artist set/context branch、gated fusion 和 CLIP residual gate。但这个类当前没有加入 `MODEL_REGISTRY`，因此 `build_model("TripletNet1")` 不会使用它。

### `dataset.py`

负责从磁盘读取 embedding、构造 triplet 数据和 dataloader。

主要功能：

- `extract_embeddings`：从视频中抽取固定数量 frame，用 CLIP 生成 `(num_frames, embedding_dim)` tensor。OpenCV、Pillow 和 CLIP 依赖是 lazy import，只在实际抽取视频 embedding 时需要。
- `save_embeddings`：遍历 artist 视频目录，生成并保存 `*_embeddings.pt`。
- `load_embeddings` / `load_pt_files`：读取 `.pt` embedding，并跳过形状或类型异常文件。
- `average_tensors`：按最常见 shape 过滤异常视频，再平均为一个 artist tensor。
- `stack_tensors`：截断或补齐 frame 数，保留最多 `max_videos` 个视频，并用全零视频 padding 到固定数量。
- `process_artists`：统一加载所有 artist，并根据 `aggregation="mean"` 或 `"stack"` 输出训练所需 tensor。
- `filter_triplets`：只保留 anchor、positive、negative 三个 artist 都有 embedding 的 triplet。
- `split_triplet_dataframe_by_artist`：默认执行严格 artist-disjoint hold-out split。
- `TripletDataset` 与 dataloader helpers：支持固定 `(a,p,n)` triplet 样本格式。

### `train.py`

实现一个 epoch 的固定 triplet 训练逻辑：每个 batch 直接使用 triplet CSV 中已有的 anchor、positive 和 negative。

返回指标包括 `loss`、`ranking_acc`、`margin_acc`、`mean_pos_dist` 和 `mean_neg_dist`。

### `evaluate.py` 与 `metrics.py`

`metrics.py` 定义 cosine/euclidean distance，以及 triplet ranking/margin accuracy：

- `ranking_acc`：`d(anchor, positive) < d(anchor, negative)`。
- `margin_acc`：`d(anchor, positive) + margin < d(anchor, negative)`。

`evaluate.py` 在 no-grad 模式下计算 validation loss、ranking accuracy 和 margin accuracy。默认返回 `(loss, ranking_acc)` 以兼容旧代码，`return_details=True` 时返回完整 dict。

### `cv_training.py`

支撑 notebook 的 5-fold cross-validation。

关键点：

- `make_artist_disjoint_kfold_splits` 严格保证 train/validation artist 不重叠。
- 跨 train/validation artist partition 的 triplet 会被丢弃。
- `run_one_fold_margin` 对一个 fold 和一个 margin 训练模型，按 validation retrieval MRR 选择 best checkpoint，而不是只按 triplet ranking accuracy。
- `summarize_cv_results` 输出 fold-level summary 和 margin-level summary CSV。

### `experiment.py`

命令行训练入口。默认配置面向当前 TripletNet1 stack 输入：

| 参数 | 默认值 |
| --- | --- |
| `--base-dir` | `data/video_embeddings` |
| `--triplets-csv` | `data/triplets/triplets_ids_music_spot.csv` |
| `--model` | `TripletNet1` |
| `--artist-aggregation` | `stack` |
| `--videos-per-artist` | `10` |
| `--num-frames` | `30` |
| `--video-dropout` | `0.15` |
| `--distance-fn` | `cosine` |
| `--margin` | `0.5` |
| `--epochs` | `30` |
| `--batch-size` | `128` |
| `--optimizer` | `adamw` |
| `--lr` | `2e-4` |
| `--weight-decay` | `1e-5` |
| `--output-dir` | `results/checkpoints` |

如果使用 `aggregation="stack"` 且模型不是 `TripletNet1`，程序会直接报错，提示 legacy baselines 应改用 `--artist-aggregation mean`。

### `utils.py`

只包含 `set_seed`，同步设置 Python、NumPy、PyTorch、CUDA 和 cuDNN deterministic 相关选项。

## Notebook 说明与结果

四个主训练 notebook 的结构基本一致：

1. 设置模型名、随机种子、路径、margin grid、batch size 和训练轮数。
2. 加载 artist embeddings 与 triplet CSV。
3. 建立 5-fold artist-disjoint splits。
4. 对每个 margin 和每个 fold 训练模型。
5. 汇总 fold summary 与 margin summary。
6. 选择平均 validation MRR 最好的 margin。
7. 聚合 out-of-fold validation triplet predictions，做 threshold、ROC-AUC/AP、错误分析。
8. 用最佳单 fold checkpoint 编码所有 artist，计算 nearest-neighbour retrieval metrics。
9. 尝试 metadata-aware latent-space 分析，并保存 OOF、retrieval、threshold 和 latent embedding CSV。

`model1_posthoc_metrics_metadata.ipynb` 是 TripletNet1 的独立补充 notebook：它不重训模型，只复用 TripletNet1 OOF prediction CSV、latent CSV 或 best checkpoint，补算 ROC-AUC/AP，并重跑 PCA/t-SNE、metadata merge、group similarity 和 silhouette 分析。

### `model1.ipynb` 到 `model4.ipynb`

四个主训练 notebook 当前统一为 fixed CSV negatives：

- `TRIPLETS_CSV = Path("../data/triplets/triplets_ids_music_spot.csv")`。
- `NUM_EPOCHS = 30`，`distance_fn="cosine"`，这两个口径不参与调参。
- margin grid: `[0.1, 0.3, 0.5, 0.7, 0.9]`。
- 理论选择的统一训练超参：`optimizer="adamw"`、`learning_rate=2e-4`、`weight_decay=1e-5`、`batch_size=128`、`early_stopping_patience=8`。

旧的 notebook 输出已经清空，因为之前保存的结果来自动态负例实验，不再对应当前方法。

### `model1_posthoc_metrics_metadata.ipynb`

用途：在 `model1.ipynb` 已经产出 TripletNet1 OOF prediction CSV、latent CSV 或 best checkpoint 后，只重跑训练后指标与 latent-space metadata 分析。

当前保存输出显示：

- 从 `TripletNet1_oof_triplet_predictions.csv` 重算 pair-level ROC-AUC `0.916974` 和 average precision `0.905226`，使用 `1132` 个 positive pairs 和 `1132` 个 negative pairs，并将结果保存到 `TripletNet1_pair_auc_metrics.csv`。
- latent 输入为 `3892` 个 artist、`256` 维 embedding。
- metadata 来源为 `data/metadata/artists_genre_country.csv`，匹配 `3892 / 3892` 个 artist。
- 可用标签列为 `country`、`broad_genre`、`genre`，artist name 来自 metadata 或 `artists.csv`。
- PCA 与 CPU sklearn t-SNE 默认使用全部可用 artist：country `3858`、broad_genre `3892`、genre `3860`；t-SNE perplexity 为 `40`。
- 输出目录为 `code/checkpoints/TripletNet1/analysis/posthoc_metrics_metadata/`，包含 ROC-AUC/AP CSV、修正后的 triplet summary、projection PNG/CSV、`TripletNet1_latent_with_metadata.csv`、`TripletNet1_group_similarity_<label>.csv` 和 `TripletNet1_silhouette_summary.csv`。

Silhouette summary 当前为 country `-0.023250`、broad_genre `-0.020109`、genre `-0.266220`，说明整体 latent space 对这些 metadata 标签不是强簇分离结构；group similarity summary 仍可用于观察局部风格/国家聚集现象。

### `model2.ipynb` 到 `model4.ipynb`

这些 legacy Conv1D baseline 当前同样使用 fixed CSV negatives，并统一为：

- `aggregation="mean"`。
- `NUM_EPOCHS = 30`，`distance_fn="cosine"`。
- margin grid: `[0.1, 0.3, 0.5, 0.7, 0.9]`。
- batch size、learning rate、weight decay 和 optimizer 与 `model1.ipynb` 保持一致。

旧动态负例实验的结果表已移除；请在真实 `data/video_embeddings/` 数据可用时重新运行 notebook 生成 fixed-triplet 结果。

## 测试覆盖

当前测试重点覆盖：

- `TripletNet1` stack 输入输出 shape 与 L2 normalization。
- legacy `(batch, frames, dim)` 单视频输入兼容。
- 全零 padded video 的 mask 行为。
- video dropout 至少保留一个有效视频。
- 非法 frame/video/feature shape 的错误。
- `stack_tensors` 零 padding 且不循环重复视频。

## 已知口径差异

`model1.ipynb` 的 markdown 描述和当前 `model.py` 中的 `TripletNet1` 实现并非完全同一个复杂度层级。实际运行入口以 `MODEL_REGISTRY` 为准；当前 `TripletNet1` 是简洁 hierarchical Transformer。若要使用 `HierarchicalVideoArtistTripletNet`，需要显式把它加入 registry 或在 notebook 中直接实例化。

四个 notebook 中的跨模型比较 cell 会尝试读取 checkpoint summary CSV。由于 checkpoint 目录不在仓库中，跨模型汇总可能来自旧运行缓存。做论文表格时建议优先使用每个 notebook 自己的 margin summary、OOF triplet summary 和 retrieval summary，并在同一环境中重新运行全部 notebook 以统一代码版本。

`model2.ipynb` 到 `model4.ipynb` 的 metadata cell 仍可能显示 t-SNE/metadata 分析被跳过，因为它们沿用旧的相对路径搜索。TripletNet1 的独立 post-hoc notebook 已使用项目根目录下的 `data/metadata/`，后续给其他模型补同类分析时建议复用这一套路径和输出约定。
