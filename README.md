# Artist Similarity Triplet Learning

本仓库是一个面向硕士论文实验的 artist similarity 项目。核心任务是使用预计算的音乐艺人视频 CLIP frame embeddings，训练 Triplet Network，使 anchor artist 的嵌入更接近已知相似 artist，并远离负例 artist。

项目当前重点是比较 `TripletNet1` 到 `TripletNet4` 四个 triplet encoder，在 artist-disjoint 5-fold cross-validation、固定 triplet CSV 负例和 retrieval-style evaluation 下的表现。

## 仓库结构

```text
.
├── code/
│   ├── model1.ipynb ... model4.ipynb   # 四个模型的训练与分析 notebook
│   ├── model1_posthoc_metrics_metadata.ipynb # TripletNet1 post-hoc 指标与 metadata/t-SNE 分析
│   ├── model.py                        # TripletNet1~4 及模型注册表
│   ├── dataset.py                      # embedding 加载、artist 聚合、triplet 构造、split
│   ├── train.py / evaluate.py          # 固定 triplet 训练与验证循环
│   ├── metrics.py                      # cosine/euclidean triplet metrics
│   ├── cv_training.py                  # 5-fold artist-disjoint CV 训练工具
│   ├── experiment.py                   # 命令行训练入口
│   └── test_*.py                       # 架构等单元测试
├── data/
│   ├── metadata/                       # artist metadata CSV
│   └── triplets/                       # triplet CSV
├── results/                            # CLI checkpoint 输出目录，已 gitignore
├── code/checkpoints/                   # notebook checkpoint 输出目录，已 gitignore
└── code/figures/model1 ... model4      # notebook 图像输出目录
```

`data/video_embeddings/` 没有提交到仓库，因为体积较大且已在 `.gitignore` 中排除。训练默认期望每个 artist 的视频 embedding 位于：

```text
data/video_embeddings/<artist_id>/embeddings/*.pt
```

每个 `.pt` 文件通常是一个 `(30, 768)` tensor，对应一个视频的 30 个均匀采样 frame 的 CLIP ViT-L/14@336px embedding。

## 当前数据与实验设置

四个 notebook 当前使用统一 fixed-triplet 配置：

| 项目 | 数值/设置 |
| --- | --- |
| 可用 artist embeddings | `3892` |
| 过滤后 triplets | `27350` |
| 主训练方式 | `5-fold artist-disjoint cross-validation` |
| 负例策略 | 使用 `triplets_ids_music_spot.csv` 中固定 negative |
| batch size | `128` |
| epochs | 最多 `30`，notebook 使用 early stopping |
| optimizer | 理论选择的统一配置：`AdamW(lr=2e-4, weight_decay=1e-5)` |
| loss | cosine `TripletMarginWithDistanceLoss(..., swap=True)` |
| seed | `3407` |
| metadata 覆盖 | `3892 / 3892` artists，来源 `artists_genre_country.csv` |

artist-disjoint split 是本项目评估口径里最重要的约束：同一个 artist 不能以 anchor、positive 或 negative 的任何身份同时出现在 train 与 validation 中。跨 train/validation artist partition 的 triplet 会被丢弃，而不是分配到任一侧。notebook 输出中的每个 fold 约保留 `51%~53%` triplet，validation rows 约 `216~235`，并且 `artist_overlap=False`。

## 模型概览

`TripletNet1` 是当前代码默认路径，用于新的 video-stack 输入格式：

```text
artist -> 最多 10 个视频 -> 每个视频 30 个 CLIP frame embeddings -> 768 维
```

在 `aggregation="stack"` 时，每个 artist tensor 形状为 `(10, 30, 768)`。视频不足 10 个时使用全零 video tensor padding，模型从全零视频自动推断 valid-video mask。训练时启用 `video_dropout_p=0.15`，但始终至少保留一个有效视频。

`TripletNet2` 到 `TripletNet4` 是 legacy Conv1D baselines，输入仍是 `(frames, dim)` artist tensor，因此运行这些模型时应使用 `--artist-aggregation mean`，先将一个 artist 的多个视频平均成单个 `(30, 768)` 表示。

## Notebook 结果摘要

旧的动态负例实验结果已经不再作为当前口径保留。当前 `model1.ipynb` 到 `model4.ipynb` 均使用 fixed CSV negatives、5-fold artist-disjoint cross-validation、cosine triplet loss 和 margin grid `[0.1, 0.3, 0.5, 0.7, 0.9]`。在真实 `data/video_embeddings/` 数据和 PyTorch 环境可用后，需要重新运行四个 notebook 生成新的 CV、OOF、threshold 和 retrieval 结果。

每个 notebook 生成的图会保存到对应子目录：`code/figures/model1/`、`code/figures/model2/`、`code/figures/model3/`、`code/figures/model4/`。

## Post-hoc 指标与 metadata 分析

`code/model1_posthoc_metrics_metadata.ipynb` 是 TripletNet1 专用的训练后分析 notebook。它用于在 `model1.ipynb` 训练完成后，不重训模型，只补跑 pair-level 指标、latent-space 与 metadata 一致性分析：

- 从 `TripletNet1_oof_triplet_predictions.csv` 读取 out-of-fold positive/negative pair cosine similarity，基于 `1132` 个 positive pairs 和 `1132` 个 negative pairs 重新计算 ROC-AUC `0.916974` 与 average precision `0.905226`。
- 优先复用 `code/checkpoints/TripletNet1/analysis/TripletNet1_artist_latent_embeddings.csv`；如果缓存不存在，会自动查找 TripletNet1 best checkpoint 并重新编码全部 artist。
- 从 `data/metadata/` 查找 metadata，当前保存输出使用 `artists_genre_country.csv`，并成功匹配 `3892 / 3892` 个 artist。
- 分析标签包括 `country`、`broad_genre`、`genre`，并补充 `artists.csv` 中的 artist name。
- 默认用 PCA 和 CPU `sklearn.manifold.TSNE` 对所有可用 artist 作图：country `3858` 个、broad_genre `3892` 个、genre `3860` 个，t-SNE perplexity 为 `40`。
- 输出保存到 `code/checkpoints/TripletNet1/analysis/posthoc_metrics_metadata/`，包括 `TripletNet1_pair_auc_metrics.csv`、`TripletNet1_triplet_summary_with_auc.csv`、projection PNG/CSV、`TripletNet1_latent_with_metadata.csv`、group similarity summary 和 silhouette summary。

当前保存的 silhouette summary 在 cosine metric 下为负值：country `-0.023250`、broad_genre `-0.020109`、genre `-0.266220`，说明整体 latent space 不是按这些 metadata 标签形成强分离簇；但 group similarity 表显示部分国家和风格组仍有较高的 intra-minus-inter similarity，可作为论文中的定性/诊断分析材料。

## 运行方式

安装依赖：

```bash
pip install -r code/requirements.txt
```

训练当前推荐的 TripletNet1：

```bash
python code/experiment.py \
  --base-dir data/video_embeddings \
  --triplets-csv data/triplets/triplets_ids_music_spot.csv \
  --model TripletNet1 \
  --artist-aggregation stack \
  --videos-per-artist 10 \
  --num-frames 30 \
  --video-dropout 0.15 \
  --margin 0.1 \
  --epochs 30
```

运行 legacy Conv1D baseline：

```bash
python code/experiment.py \
  --model TripletNet2 \
  --artist-aggregation mean \
  --margin 0.3 \
  --epochs 30
```

运行测试：

```bash
cd code
pytest -q
```

只重跑 TripletNet1 post-hoc 指标与 metadata/t-SNE 分析时，在 Jupyter 中打开并运行：

```text
code/model1_posthoc_metrics_metadata.ipynb
```

该 notebook 期望已有 `TripletNet1_oof_triplet_predictions.csv`，用于补算 ROC-AUC/AP；metadata/t-SNE 部分还需要 TripletNet1 latent CSV 或 best checkpoint。这些运行产物都在 `code/checkpoints/` 下，因体积/运行产物原因不会提交到 git。

## 重要注意事项

`model1.ipynb` 已对齐当前 `MODEL_REGISTRY` 中的 `TripletNet1` 实现。`model.py` 仍保留一个更复杂的 `HierarchicalVideoArtistTripletNet` 类，但它没有注册到 `MODEL_REGISTRY`；当前通过 `build_model("TripletNet1")` 和 `experiment.py --model TripletNet1` 实际使用的是简洁版 hierarchical Transformer：frame projection、frame-level Transformer、video mean pooling、masked artist-level Transformer、masked mean pooling、projection head、BNNeck、L2 normalization。

部分 notebook 的跨模型比较 cell 会读取既有 checkpoint summary CSV。由于 `code/checkpoints/` 被 gitignore，且 notebook 可能在不同代码版本下运行，这些跨模型缓存表可能与单个 notebook 当前保存的结果不完全一致。本 README 以每个 notebook 自己的训练输出、OOF 分析和 retrieval 分析为主。

`model2.ipynb` 到 `model4.ipynb` 的 metadata-aware t-SNE cell 仍可能因为相对路径搜索而跳过 metadata。最新的 `model1_posthoc_metrics_metadata.ipynb` 已改为从项目根目录的 `data/metadata/` 读取，并保存了 TripletNet1 的 metadata/t-SNE 结果。若后续要对 TripletNet2~4 做同样分析，应复用这个 notebook 的 metadata 路径逻辑。
