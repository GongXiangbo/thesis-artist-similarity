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
  --negative-mining memory_bank_semihard \
  --epochs 30
```

运行 Conv1D baseline 时使用 mean aggregation：

```bash
python code/experiment.py \
  --model TripletNet2 \
  --artist-aggregation mean \
  --margin 0.2 \
  --negative-mining memory_bank_semihard \
  --epochs 30
```

运行单元测试：

```bash
cd code
pytest -q
```

上面两个训练命令显式传入的是当前 notebook 结果中表现最好的 margin。`experiment.py` 自身的 `--margin` 默认值仍是 `0.5`，如果要复现 README 里的表格，请不要省略该参数。

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

`model1_latent_space_metadata.ipynb` 会从项目根目录的 `data/metadata/` 查找 metadata；当前保存输出使用 `artists_genre_country.csv`，并补充 `artists.csv` 中的 artist name。

每个视频 embedding 是 `(frames, dim)`，当前实验使用 `(30, 768)`。`dataset.process_artists` 支持两种 artist 聚合方式：

| aggregation | 输出形状 | 适用模型 | 说明 |
| --- | --- | --- | --- |
| `stack` | `(max_videos, frames, dim)`，默认 `(10, 30, 768)` | `TripletNet1` | 保留一个 artist 的多个视频，视频不足时用全零 tensor padding |
| `mean` | `(frames, dim)`，默认 `(30, 768)` | `TripletNet2` 到 `TripletNet5` | 将一个 artist 的多个视频 tensor 按视频维平均，兼容 legacy Conv1D baselines |

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

`TripletNet2` 到 `TripletNet5` 都继承 `ConvTripletNet`，输入是 legacy `(batch, seq_len, d_model)`：

| 模型 | Conv1D channels | kernel sizes | stride/pool 设计 | projection |
| --- | --- | --- | --- | --- |
| `TripletNet2` | `256, 256, 128` | `3, 3, 3` | 第 0、1 层后 max-pool | hidden `256` -> output `256` |
| `TripletNet3` | `256, 256, 128, 128` | `5, 5, 3, 3` | 第 0、2 层后 max-pool | hidden `256` -> output `256` |
| `TripletNet4` | `256, 256, 128, 128` | `3, 3, 3, 3` | stride `1, 2, 1, 2`，不额外 pool | hidden `256` -> output `256` |
| `TripletNet5` | `256, 256, 256, 128, 128` | `3, 3, 3, 3, 3` | 第 0 层后 max-pool，后续 stride 下采样 | hidden `512, 256` -> output `256` |

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
- `build_positive_map` / `build_negative_exclusion_map`：构建 negative mining 的排除集合，后者会排除直接正例和二跳邻居。
- `create_artist_memory_bank`：收集训练 split 中所有 unique artist tensor，用于 global memory-bank mining。
- `TripletDataset` 与 dataloader helpers：支持 `(a,p,n)` 和 `(a,p,n,anchor_id,positive_id,negative_id)` 两种样本格式。

### `mining.py`

实现 `BatchSemiHardNegativeMiner`。

每个 anchor-positive pair 会从候选 artist pool 中选择一个 negative。候选 mask 会排除：

1. anchor 自己；
2. 当前 paired positive；
3. `positive_map` 中已知的 direct positive；
4. 当使用 `build_negative_exclusion_map(..., include_two_hop=True)` 时，还排除二跳邻居。

选择优先级：

1. 距离满足 `d(a,p) < d(a,n) < d(a,p) + margin` 的 closest semi-hard negative；
2. 如果没有 semi-hard，则选择距离大于 positive distance 的 closest valid negative；
3. 如果所有 valid negatives 都比 positive 更近，则选择 closest valid negative 并产生正 loss；
4. 如果没有任何 valid candidate，则跳过该 anchor，并通过零 loss 保持 autograd graph 有效。

### `train.py`

实现一个 epoch 的训练逻辑，支持四种 negative mining mode：

| mode | 行为 |
| --- | --- |
| `fixed` / `random` | 使用 triplet CSV 中已有 negative |
| `batch_semihard` | 在当前 batch 的 unique artists 中挖 semi-hard negative |
| `memory_bank_semihard` | 每个 epoch 开始时编码全部训练 artists，作为 global candidate pool 挖 negative |

`memory_bank_semihard` 只用 memory bank embedding 做 negative 选择。选中 negative ID 后，会把对应原始 artist tensor 再 forward 一次，以便 selected negative 的梯度正常回传。

返回指标包括 `loss`、`ranking_acc`、`margin_acc`、`semi_hard_ratio`、`fallback_ratio`、`skipped_ratio`、`mean_pos_dist`、`mean_neg_dist` 和 `memory_bank_size`。

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
| `--negative-mining` | `memory_bank_semihard` |
| `--distance-fn` | `cosine` |
| `--margin` | `0.5` |
| `--epochs` | `30` |
| `--batch-size` | `128` |
| `--output-dir` | `results/checkpoints` |

如果使用 `aggregation="stack"` 且模型不是 `TripletNet1`，程序会直接报错，提示 legacy baselines 应改用 `--artist-aggregation mean`。

### `utils.py`

只包含 `set_seed`，同步设置 Python、NumPy、PyTorch、CUDA 和 cuDNN deterministic 相关选项。

## Notebook 说明与结果

五个主训练 notebook 的结构基本一致：

1. 设置模型名、随机种子、路径、margin grid、batch size、训练轮数和 mining mode。
2. 加载 artist embeddings 与 triplet CSV。
3. 建立 5-fold artist-disjoint splits。
4. 对每个 margin 和每个 fold 训练模型。
5. 汇总 fold summary 与 margin summary。
6. 选择平均 validation MRR 最好的 margin。
7. 聚合 out-of-fold validation triplet predictions，做 threshold、ROC-AUC/AP、错误分析。
8. 用最佳单 fold checkpoint 编码所有 artist，计算 nearest-neighbour retrieval metrics。
9. 尝试 metadata-aware latent-space 分析，并保存 OOF、retrieval、threshold 和 latent embedding CSV。

`model1_latent_space_metadata.ipynb` 是 TripletNet1 的独立补充 notebook：它不重训模型，只复用 TripletNet1 latent CSV 或 best checkpoint，重跑 PCA/t-SNE、metadata merge、group similarity 和 silhouette 分析。

### `model1.ipynb` / TripletNet1

配置：

- `aggregation="stack"`，artist tensor shape `(10, 30, 768)`。
- `video_dropout_p=0.15`。
- margin grid: `[0.1, 0.3, 0.5, 0.7, 0.9]`。
- memory bank size 约 `2405`。

Margin summary：

| margin | mean_best_val_mrr | mean_best_val_acc | mean_best_val_margin_acc | mean_best_val_loss | semi-hard | fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.342917` | `0.923997` | `0.874760` | `0.042022` | `0.718619` | `0.281381` |
| `0.3` | `0.304288` | `0.897113` | `0.638933` | `0.128319` | `0.939471` | `0.060529` |
| `0.5` | `0.288396` | `0.854684` | `0.428813` | `0.287421` | `0.956375` | `0.043625` |
| `0.7` | `0.260967` | `0.840156` | `0.367353` | `0.412660` | `0.996356` | `0.003644` |
| `0.9` | `0.243517` | `0.844133` | `0.343881` | `0.509479` | `0.997617` | `0.002383` |

Best margin is `0.10`. OOF ranking accuracy is `92.40%`; OOF margin-satisfied accuracy is `87.46%`. Best F1 threshold is `0.2748` with F1 `0.857143`. Retrieval summary: `precision@1=0.551699`, `hit@5=0.799284`, `MRR=0.663495`, `average_precision=0.402115`.

### `model1_latent_space_metadata.ipynb`

用途：在 `model1.ipynb` 已经产出 TripletNet1 latent CSV 或 best checkpoint 后，只重跑 latent-space metadata 分析。

当前保存输出显示：

- latent 输入为 `3892` 个 artist、`256` 维 embedding。
- metadata 来源为 `data/metadata/artists_genre_country.csv`，匹配 `3892 / 3892` 个 artist。
- 可用标签列为 `country`、`broad_genre`、`genre`，artist name 来自 metadata 或 `artists.csv`。
- PCA 与 CPU sklearn t-SNE 默认使用全部可用 artist：country `3858`、broad_genre `3892`、genre `3860`；t-SNE perplexity 为 `40`。
- 输出目录为 `code/checkpoints/TripletNet1/analysis/latent_metadata/`，包含 projection PNG/CSV、`TripletNet1_latent_with_metadata.csv`、`TripletNet1_group_similarity_<label>.csv` 和 `TripletNet1_silhouette_summary.csv`。

Silhouette summary 当前为 country `-0.023250`、broad_genre `-0.020109`、genre `-0.266220`，说明整体 latent space 对这些 metadata 标签不是强簇分离结构；group similarity summary 仍可用于观察局部风格/国家聚集现象。

### `model2.ipynb` / TripletNet2

配置：

- `aggregation="mean"`。
- margin grid: `[0.1, 0.2, 0.3, 0.5, 0.7, 0.9]`。

Margin summary：

| margin | mean_best_val_mrr | mean_best_val_acc | mean_best_val_margin_acc | mean_best_val_loss | semi-hard | fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.2` | `0.315455` | `0.914880` | `0.827464` | `0.068955` | `0.940238` | `0.059762` |
| `0.1` | `0.315058` | `0.926521` | `0.881193` | `0.038309` | `0.895029` | `0.104971` |
| `0.3` | `0.303155` | `0.911080` | `0.745368` | `0.101442` | `0.979364` | `0.020636` |
| `0.5` | `0.282218` | `0.898722` | `0.615089` | `0.194092` | `0.997487` | `0.002513` |
| `0.7` | `0.263576` | `0.876823` | `0.447342` | `0.349871` | `0.998445` | `0.001555` |
| `0.9` | `0.237132` | `0.861931` | `0.345797` | `0.485933` | `0.997065` | `0.002935` |

Best margin is `0.20`. OOF ranking accuracy is `91.43%`; OOF margin-satisfied accuracy is `82.69%`; ROC-AUC is `0.9119`; AP is `0.9049`. Retrieval summary: `precision@1=0.365295`, `hit@5=0.667263`, `MRR=0.502433`, `average_precision=0.267810`.

### `model3.ipynb` / TripletNet3

Margin summary：

| margin | mean_best_val_mrr | mean_best_val_acc | mean_best_val_margin_acc | mean_best_val_loss | semi-hard | fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.309094` | `0.926204` | `0.888465` | `0.040211` | `0.942181` | `0.057819` |
| `0.3` | `0.302996` | `0.902118` | `0.756390` | `0.100579` | `0.993158` | `0.006842` |
| `0.2` | `0.293057` | `0.911118` | `0.830851` | `0.070395` | `0.977928` | `0.022072` |
| `0.5` | `0.274859` | `0.895389` | `0.620637` | `0.197693` | `0.998011` | `0.001989` |
| `0.7` | `0.253922` | `0.883749` | `0.469128` | `0.334447` | `0.998468` | `0.001532` |
| `0.9` | `0.207604` | `0.849436` | `0.368629` | `0.498139` | `0.996716` | `0.003284` |

Best margin is `0.10`. OOF ranking accuracy is `92.58%`; OOF margin-satisfied accuracy is `88.78%`; ROC-AUC is `0.9259`; AP is `0.9167`. Retrieval summary: `precision@1=0.299821`, `hit@5=0.609660`, `MRR=0.438647`, `average_precision=0.219581`.

### `model4.ipynb` / TripletNet4

Margin summary：

| margin | mean_best_val_mrr | mean_best_val_acc | mean_best_val_margin_acc | mean_best_val_loss | semi-hard | fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.1` | `0.301722` | `0.915925` | `0.883373` | `0.043642` | `0.950809` | `0.049191` |
| `0.3` | `0.289566` | `0.912578` | `0.771310` | `0.101719` | `0.993373` | `0.006627` |
| `0.2` | `0.288040` | `0.911943` | `0.834835` | `0.070858` | `0.980536` | `0.019464` |
| `0.5` | `0.267549` | `0.881750` | `0.610898` | `0.208968` | `0.998261` | `0.001739` |
| `0.7` | `0.239320` | `0.871366` | `0.465483` | `0.340292` | `0.997986` | `0.002014` |
| `0.9` | `0.211955` | `0.856149` | `0.377649` | `0.514460` | `0.997907` | `0.002093` |

Best margin is `0.10`. OOF ranking accuracy is `91.61%`; OOF margin-satisfied accuracy is `88.34%`; ROC-AUC is `0.9210`; AP is `0.9089`. Retrieval summary: `precision@1=0.256887`, `hit@5=0.548479`, `MRR=0.393616`, `average_precision=0.189462`.

### `model5.ipynb` / TripletNet5

Margin summary：

| margin | mean_best_val_mrr | mean_best_val_acc | mean_best_val_margin_acc | mean_best_val_loss | semi-hard | fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.2` | `0.251099` | `0.897132` | `0.821559` | `0.096732` | `0.993637` | `0.006363` |
| `0.1` | `0.249525` | `0.909304` | `0.870621` | `0.061716` | `0.988471` | `0.011529` |
| `0.3` | `0.242906` | `0.890245` | `0.764782` | `0.127305` | `0.995392` | `0.004608` |
| `0.5` | `0.197034` | `0.880796` | `0.628561` | `0.230509` | `0.994059` | `0.005941` |
| `0.7` | `0.187445` | `0.857197` | `0.492263` | `0.372090` | `0.993520` | `0.006480` |
| `0.9` | `0.145951` | `0.810496` | `0.358295` | `0.605705` | `0.994477` | `0.005523` |

Best margin is `0.20`. OOF ranking accuracy is `89.66%`; OOF margin-satisfied accuracy is `82.07%`; ROC-AUC is `0.8909`; AP is `0.8776`. Retrieval summary: `precision@1=0.256530`, `hit@5=0.525939`, `MRR=0.382906`, `average_precision=0.193078`.

## 测试覆盖

当前测试重点覆盖：

- `TripletNet1` stack 输入输出 shape 与 L2 normalization。
- legacy `(batch, frames, dim)` 单视频输入兼容。
- 全零 padded video 的 mask 行为。
- video dropout 至少保留一个有效视频。
- 非法 frame/video/feature shape 的错误。
- `stack_tensors` 零 padding 且不循环重复视频。
- semi-hard negative selection、fallback、known positives/self exclusion、全 skip batch 的 backward。
- memory-bank mining 的二跳邻居排除和训练 step。

## 已知口径差异

`model1.ipynb` 的 markdown 描述和当前 `model.py` 中的 `TripletNet1` 实现并非完全同一个复杂度层级。实际运行入口以 `MODEL_REGISTRY` 为准；当前 `TripletNet1` 是简洁 hierarchical Transformer。若要使用 `HierarchicalVideoArtistTripletNet`，需要显式把它加入 registry 或在 notebook 中直接实例化。

五个 notebook 中的跨模型比较 cell 会尝试读取 checkpoint summary CSV。由于 checkpoint 目录不在仓库中，跨模型汇总可能来自旧运行缓存。做论文表格时建议优先使用每个 notebook 自己的 margin summary、OOF triplet summary 和 retrieval summary，并在同一环境中重新运行全部 notebook 以统一代码版本。

`model2.ipynb` 到 `model5.ipynb` 的已保存 metadata cell 仍可能显示 t-SNE/metadata 分析被跳过，因为它们沿用旧的相对路径搜索。TripletNet1 的独立 metadata notebook 已使用项目根目录下的 `data/metadata/`，后续给其他模型补同类分析时建议复用这一套路径和输出约定。
