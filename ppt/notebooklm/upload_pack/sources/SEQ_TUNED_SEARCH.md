# Seq-Tuned 权重搜索记录

## 目标

`Seq-Tuned` 不是新增黑盒模型，而是在 `Seq-Hybrid` 的同一批可解释特征上重新分配权重，让外卖场景更强的复购和商家转移信号占更大比例。这样做的目的是提高离线 Recall/NDCG，同时保持模型解释仍然来自真实参与打分的特征。

使用的特征：

- `fast_recency`：最近订单快速衰减。
- `slow_recency`：较慢时间衰减。
- `repeat`：历史复购频率。
- `transition`：最近商家到候选商家的全局转移概率。
- `category`：用户历史品类偏好。
- `popularity`：商家全局热度。
- `quality`：商家评分质量。

## 可复现脚本

轻量 smoke：

```bash
make seq-tune-smoke
```

更接近本项目调参时使用的搜索口径：

```bash
conda run --no-capture-output -n foodflow python scripts/search_seq_weights.py \
  --processed-dir data/processed \
  --output outputs/experiments/seq_weight_search.csv \
  --user-limit 120 \
  --candidate-limit 140 \
  --trials 48 \
  --seed 2026
```

正式指标不直接引用搜索脚本的轻量结果，而是把候选权重固化为 `Seq-Tuned` 和 `Seq-Tuned-xQuAD` 后，再通过标准评估命令确认：

```bash
conda run --no-capture-output -n foodflow python -m foodflow.cli eval-offline \
  --processed-dir data/processed \
  --output outputs/results/offline_metrics.csv \
  --top-k 10 20 \
  --seed 42
```

## 当前采用权重

`Seq-Tuned`：

| 特征 | 权重 |
|---|---:|
| fast_recency | 0.142601 |
| slow_recency | 0.093624 |
| repeat | 0.412158 |
| transition | 0.247023 |
| category | 0.091250 |
| popularity | 0.008945 |
| quality | 0.004398 |

`Seq-Tuned-xQuAD`：

| 特征 | 权重 |
|---|---:|
| fast_recency | 0.270998 |
| slow_recency | 0.106017 |
| repeat | 0.279395 |
| transition | 0.274380 |
| category | 0.048099 |
| popularity | 0.005218 |
| quality | 0.015892 |

## 正式指标

完整 TRD 处理结果、300 个测试用户评估口径：

| 模型 | Recall@20 | NDCG@20 | HitRate@20 | ExposureGini | CategoryJSD@20 |
|---|---:|---:|---:|---:|---:|
| Seq-Hybrid | 0.4441 | 0.3513 | 0.5933 | 0.9198 | 0.0152 |
| Seq-xQuAD | 0.4447 | 0.3538 | 0.5967 | 0.9141 | 0.0151 |
| Seq-Tuned | 0.4675 | 0.3652 | 0.6267 | 0.8927 | 0.0152 |
| Seq-Tuned-xQuAD | 0.4670 | 0.3613 | 0.6200 | 0.8664 | 0.0140 |

结论：`Seq-Tuned` 是当前离线准确率前沿；`Seq-Tuned-xQuAD` 在几乎保持同等 Recall 的同时，降低曝光集中并改善品类校准。
