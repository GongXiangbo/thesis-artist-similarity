# Artist Similarity Triplet Learning

本仓库是一个面向硕士论文实验的 artist similarity 项目。核心任务是使用预计算的音乐艺人视频 CLIP frame embeddings，训练 Triplet Network，使 anchor artist 的嵌入更接近已知相似 artist，并远离负例 artist。

项目当前重点是比较 `TripletNet1` 到 `TripletNet5` 五个 triplet encoder，在 artist-disjoint 5-fold cross-validation、memory-bank semi-hard negative mining 和 retrieval-style evaluation 下的表现。

## 仓库结构

```text
.
├── code/
│   ├── model1.ipynb ... model5.ipynb   # 五个模型的训练与分析 notebook
│   ├── model1_posthoc_metrics_metadata.ipynb # TripletNet1 post-hoc 指标与 metadata/t-SNE 分析
│   ├── model.py                        # TripletNet1~5 及模型注册表
│   ├── dataset.py                      # embedding 加载、artist 聚合、triplet 构造、split
│   ├── train.py / evaluate.py          # 训练与验证循环
│   ├── mining.py                       # semi-hard negative mining
│   ├── metrics.py                      # cosine/euclidean triplet metrics
│   ├── cv_training.py                  # 5-fold artist-disjoint CV 训练工具
│   ├── experiment.py                   # 命令行训练入口
│   └── test_*.py                       # 架构与 mining 的单元测试
├── data/
│   ├── metadata/                       # artist metadata CSV
│   └── triplets/                       # triplet CSV
├── results/                            # CLI checkpoint 输出目录，已 gitignore
└── code/checkpoints/                   # notebook checkpoint 输出目录，已 gitignore
```

`data/video_embeddings/` 没有提交到仓库，因为体积较大且已在 `.gitignore` 中排除。训练默认期望每个 artist 的视频 embedding 位于：

```text
data/video_embeddings/<artist_id>/embeddings/*.pt
```

每个 `.pt` 文件通常是一个 `(30, 768)` tensor，对应一个视频的 30 个均匀采样 frame 的 CLIP ViT-L/14@336px embedding。

## 当前数据与实验设置

五个 notebook 的已保存输出显示：

| 项目 | 数值/设置 |
| --- | --- |
| 可用 artist embeddings | `3892` |
| 过滤后 triplets | `27350` |
| 主训练方式 | `5-fold artist-disjoint cross-validation` |
| 负样本挖掘 | `memory_bank_semihard` |
| batch size | `128` |
| epochs | 最多 `30`，notebook 使用 early stopping |
| optimizer | `AdamW(lr=2e-4, weight_decay=1e-6)` |
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

`TripletNet2` 到 `TripletNet5` 是 legacy Conv1D baselines，输入仍是 `(frames, dim)` artist tensor，因此运行这些模型时应使用 `--artist-aggregation mean`，先将一个 artist 的多个视频平均成单个 `(30, 768)` 表示。

## Notebook 结果摘要

下表按每个 notebook 自己保存的输出汇总，不依赖 gitignored checkpoint CSV。`CV MRR` 是 5-fold validation retrieval MRR 的平均值，也是 notebook 的最佳 margin 选择标准。`OOF ranking` 使用最佳 margin 下所有 out-of-fold validation triplets；每条 OOF 预测都来自没有训练过该 fold artist 的模型。`retrieval MRR` 是用最佳单 fold checkpoint 编码全部 `3892` 个 artist 后，在 `2795` 个至少有 ground-truth positive 的 anchor 上计算的 nearest-neighbour retrieval 指标。

| Notebook | 最佳 margin | CV MRR | CV ranking acc | CV margin acc | OOF ranking | OOF margin | ROC-AUC | AP | retrieval P@1 | retrieval hit@5 | retrieval MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `model1.ipynb` / TripletNet1 | `0.10` | `0.342917` | `0.923997` | `0.874760` | `92.40%` | `87.46%` | `nan` | `nan` | `0.551699` | `0.799284` | `0.663495` |
| `model2.ipynb` / TripletNet2 | `0.20` | `0.315455` | `0.914880` | `0.827464` | `91.43%` | `82.69%` | `0.9119` | `0.9049` | `0.365295` | `0.667263` | `0.502433` |
| `model3.ipynb` / TripletNet3 | `0.10` | `0.309094` | `0.926204` | `0.888465` | `92.58%` | `88.78%` | `0.9259` | `0.9167` | `0.299821` | `0.609660` | `0.438647` |
| `model4.ipynb` / TripletNet4 | `0.10` | `0.301722` | `0.915925` | `0.883373` | `91.61%` | `88.34%` | `0.9210` | `0.9089` | `0.256887` | `0.548479` | `0.393616` |
| `model5.ipynb` / TripletNet5 | `0.20` | `0.251099` | `0.897132` | `0.821559` | `89.66%` | `82.07%` | `0.8909` | `0.8776` | `0.256530` | `0.525939` | `0.382906` |

从 notebook 自身结果看，当前 `TripletNet1` 在 CV MRR 与 retrieval MRR 上领先；`TripletNet3` 的 OOF ranking/margin accuracy 和 ROC-AUC/AP 很强；`TripletNet2` 是 Conv1D baseline 中 retrieval 表现最稳的模型。

`model1.ipynb` 的 ROC-AUC/AP 为 `nan`，notebook 输出说明当时运行环境缺少 `sklearn`，因此阈值分析仍有 precision/recall/F1，但 ROC-AUC 和 average precision 被跳过。当前 `code/model1_posthoc_metrics_metadata.ipynb` 会从已保存的 OOF prediction CSV 补算这两个指标。

## 阈值与错误分析

每个 notebook 会把 OOF triplet 拆成 positive pair 与 negative pair，用 cosine similarity threshold 做 pair-level 分类，并选取 F1 最优阈值。

| 模型 | F1 最优阈值 | precision | recall | F1 | false-positive negative pairs | false-negative positive pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TripletNet1 | `0.2748` | `0.853018` | `0.861307` | `0.857143` | `168` | `157` |
| TripletNet2 | `0.2364` | `0.797940` | `0.889576` | `0.841270` | `255` | `125` |
| TripletNet3 | `0.3805` | `0.822240` | `0.894876` | `0.857022` | `219` | `119` |
| TripletNet4 | `0.2960` | `0.789352` | `0.903710` | `0.842669` | `273` | `109` |
| TripletNet5 | `0.3574` | `0.788415` | `0.865724` | `0.825263` | `263` | `152` |

## Post-hoc 指标与 metadata 分析

`code/model1_posthoc_metrics_metadata.ipynb` 是 TripletNet1 专用的训练后分析 notebook。它用于在 `model1.ipynb` 训练完成后，不重训模型，只补跑 pair-level 指标、latent-space 与 metadata 一致性分析：

- 从 `TripletNet1_oof_triplet_predictions.csv` 读取 out-of-fold positive/negative pair cosine similarity，重新计算并保存 ROC-AUC 与 average precision，避免 `model1.ipynb` 旧运行环境缺少 `sklearn` 导致表格里出现 `nan`。
- 优先复用 `code/checkpoints/TripletNet1/analysis/TripletNet1_artist_latent_embeddings.csv`；如果缓存不存在，会自动查找 TripletNet1 best checkpoint 并重新编码全部 artist。
- 从 `data/metadata/` 查找 metadata，当前保存输出使用 `artists_genre_country.csv`，并成功匹配 `3892 / 3892` 个 artist。
- 分析标签包括 `country`、`broad_genre`、`genre`，并补充 `artists.csv` 中的 artist name。
- 默认用 PCA 和 CPU `sklearn.manifold.TSNE` 对所有可用 artist 作图：country `3858` 个、broad_genre `3892` 个、genre `3860` 个，t-SNE perplexity 为 `40`。
- 输出保存到 `code/checkpoints/TripletNet1/analysis/posthoc_metrics_metadata/`，包括 ROC-AUC/AP CSV、修正后的 triplet summary、projection PNG/CSV、`TripletNet1_latent_with_metadata.csv`、group similarity summary 和 silhouette summary。

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
  --negative-mining memory_bank_semihard \
  --epochs 30
```

运行 legacy Conv1D baseline：

```bash
python code/experiment.py \
  --model TripletNet2 \
  --artist-aggregation mean \
  --margin 0.2 \
  --negative-mining memory_bank_semihard \
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

`model1.ipynb` 的 markdown 标题和部分说明提到了更复杂的 dual-branch / attentive-statistics / temporal-delta 设计；当前 `model.py` 中确实保留了一个复杂的 `HierarchicalVideoArtistTripletNet` 类，但它没有注册到 `MODEL_REGISTRY`。当前通过 `build_model("TripletNet1")` 和 `experiment.py --model TripletNet1` 实际使用的是简洁版 hierarchical Transformer：frame projection、frame-level Transformer、video mean pooling、masked artist-level Transformer、masked mean pooling、projection head、BNNeck、L2 normalization。

部分 notebook 的跨模型比较 cell 会读取既有 checkpoint summary CSV。由于 `code/checkpoints/` 被 gitignore，且 notebook 可能在不同代码版本下运行，这些跨模型缓存表可能与单个 notebook 当前保存的结果不完全一致。本 README 以每个 notebook 自己的训练输出、OOF 分析和 retrieval 分析为主。

`model2.ipynb` 到 `model5.ipynb` 的已保存输出仍显示 metadata-aware t-SNE 被跳过，因为这些 notebook 的旧 cell 只在 notebook 工作目录附近搜索 metadata。最新的 `model1_posthoc_metrics_metadata.ipynb` 已改为从项目根目录的 `data/metadata/` 读取，并保存了 TripletNet1 的 metadata/t-SNE 结果。若后续要对 TripletNet2~5 做同样分析，应复用这个 notebook 的 metadata 路径逻辑。
