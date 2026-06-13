# Artist Similarity Triplet Learning

本仓库是一个面向硕士论文实验的 artist similarity 项目。核心任务是使用预计算的音乐艺人视频 CLIP frame embeddings，训练 Triplet Network，使 anchor artist 的嵌入更接近已知相似 artist，并远离负例 artist。

项目当前重点是比较 `TripletNet1` 到 `TripletNet4` 四个 triplet encoder，在 artist-disjoint 5-fold cross-validation、固定 triplet CSV 负例和 retrieval-style evaluation 下的表现。

## 仓库结构

```text
.
├── code/
│   ├── model1.ipynb ... model4.ipynb   # 四个模型的训练与分析 notebook
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

旧的动态负例实验结果已经不再作为当前口径保留。`model1.ipynb` 到 `model4.ipynb` 已全部按 fixed CSV negatives、5-fold artist-disjoint cross-validation、cosine triplet loss 和 margin grid `[0.1, 0.3, 0.5, 0.7, 0.9]` 跑完。四个模型的最佳 margin 均为 `0.10`，每个模型的 out-of-fold validation triplets 均为 `1132`。

| 模型 | 输入聚合 | best margin | mean validation MRR | OOF ranking acc | OOF margin acc | OOF ROC-AUC | OOF AP | retrieval MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TripletNet1` | `stack` | `0.10` | `0.311444` | `90.46%` | `87.90%` | `0.9254` | `0.9127` | `0.405530` |
| `TripletNet2` | `mean` | `0.10` | `0.278678` | `91.61%` | `84.45%` | `0.9127` | `0.9003` | `0.326258` |
| `TripletNet3` | `mean` | `0.10` | `0.274923` | `92.05%` | `85.78%` | `0.9113` | `0.8895` | `0.262694` |
| `TripletNet4` | `mean` | `0.10` | `0.270821` | `90.46%` | `84.63%` | `0.9129` | `0.8998` | `0.333466` |

`TripletNet1` 在 mean validation MRR、OOF ROC-AUC/AP 和 full-corpus retrieval MRR 上都是当前最强模型；`TripletNet3` 的 OOF triplet ranking accuracy 最高，但 retrieval 指标明显弱于 `TripletNet1`。最佳 F1 pair threshold 分别为 `TripletNet1=0.273`、`TripletNet2=0.614`、`TripletNet3=0.598`、`TripletNet4=0.517`。四个模型都编码了 `3892` 个 artist，并在 retrieval evaluation 中使用 `2795` 个至少有一个 ground-truth positive 的 anchor artist。

每个 notebook 生成的图已保存到对应子目录：`code/figures/model1/`、`code/figures/model2/`、`code/figures/model3/`、`code/figures/model4/`。每个目录包含 margin sensitivity、CV history、cross-model best validation MRR、OOF positive/negative similarity、margin gap、pair threshold analysis，以及 country/genre 的 PCA/t-SNE latent-space 图。PCA/t-SNE 图使用同一批 sampled artists；t-SNE 会先 PCA 预降维到最多 50 维，再用 cosine t-SNE 生成一套 2D 坐标，country/genre 图只更换着色标签。

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
  --margin 0.1 \
  --epochs 30
```

运行测试：

```bash
cd code
pytest -q
```

## 重要注意事项

`model1.ipynb` 已对齐当前 `MODEL_REGISTRY` 中的 `TripletNet1` 实现。`model.py` 仍保留一个更复杂的 `HierarchicalVideoArtistTripletNet` 类，但它没有注册到 `MODEL_REGISTRY`；当前通过 `build_model("TripletNet1")` 和 `experiment.py --model TripletNet1` 实际使用的是简洁版 hierarchical Transformer：frame projection、frame-level Transformer、video mean pooling、masked artist-level Transformer、masked mean pooling、projection head、BNNeck、L2 normalization。

部分 notebook 的跨模型比较 cell 会读取既有 checkpoint summary CSV。由于 `code/checkpoints/` 被 gitignore，且 notebook 可能在不同代码版本下运行，这些跨模型缓存表可能与单个 notebook 当前保存的结果不完全一致。本 README 以每个 notebook 自己的训练输出、OOF 分析和 retrieval 分析为主。

`model1.ipynb` 到 `model4.ipynb` 的 metadata-aware t-SNE cell 会依次从 `../data/metadata/`、`data/metadata/` 和当前目录查找 metadata，因此从 `code/` 目录或项目根目录启动 Jupyter 时都能读取 `artists_genre_country.csv`。
