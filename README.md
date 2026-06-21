# 基于动态知识图谱注意力的餐饮外卖个性化推荐

本项目用于推荐系统/大数据课程作业，主题是“不依赖骑手与配送数据的餐饮外卖 Top-K 商家推荐”。核心方法是把用户历史订单和下单前点击序列转成动态图谱信号：历史订单边按时间衰减，点击序列生成临时兴趣节点，再通过关系感知注意力聚合用户当前饮食偏好。

## 数据集

默认使用公开的 TRD: Takeout Recommendation Dataset from Meituan Takeout app。该数据集来自美团外卖 App，北京 11 个商圈，时间范围为 2021-03-01 至 2021-03-28，包含用户、商家、菜品、订单和下单前点击商家序列。项目只需要文本文件，不需要下载 1.8GB 的 `graph.bin`。

我已将 TRD 文本数据下载到 `data/raw` 并完成 MD5 校验。若需要重新下载，运行：

```bash
python3 scripts/download_trd.py --raw-dir data/raw
```

如果只想快速验证代码，可生成一个 TRD 字段兼容的小样例：

```bash
python3 scripts/make_demo_data.py --raw-dir data/raw_demo
```

## 运行流程

预处理并构建动态图谱样本：

```bash
python3 -m src.prepare_data \
  --raw-dir data/raw \
  --output data/processed/trd_small.pkl \
  --max-train-orders 50000 \
  --max-test-orders 5000
```

快速使用 demo 数据：

```bash
python3 -m src.prepare_data \
  --raw-dir data/raw_demo \
  --output data/processed/demo.pkl \
  --max-train-orders 2000 \
  --max-test-orders 300
```

训练完整模型：

```bash
python3 -m src.run_experiment \
  --data data/processed/trd_small.pkl \
  --model full \
  --epochs 5 \
  --batch-size 512
```

复现实验报告中的轻量真实数据结果：

```bash
python3 -m src.prepare_data --raw-dir data/raw --output data/processed/trd_fast.pkl \
  --max-train-orders 8000 --max-test-orders 1000 --eval-candidates 100 \
  --negatives 1 --max-food-attrs-per-poi 10 --max-interests 32
python3 -m src.run_baselines --data data/processed/trd_fast.pkl
python3 -m src.run_experiment --data data/processed/trd_fast.pkl --model full \
  --epochs 1 --batch-size 512 --embed-dim 24 --hidden-dim 48 \
  --checkpoint outputs/full_trd_fast.pt
```

复现最终报告使用的 GPU/100k 实验设置：

```bash
python3 -m src.prepare_data --raw-dir data/raw --output data/processed/trd_gpu_100k.pkl \
  --max-train-orders 100000 --max-test-orders 10000 --eval-candidates 200 \
  --negatives 5 --max-food-attrs-per-poi 20 --max-interests 64 \
  --max-history 30 --max-clicks 20

python3 -m src.run_baselines \
  --data data/processed/trd_gpu_100k.pkl \
  --output outputs/baselines_trd_gpu_100k.json

python3 -m src.run_experiment \
  --data data/processed/trd_gpu_100k.pkl \
  --model full \
  --epochs 10 \
  --batch-size 2048 \
  --embed-dim 64 \
  --hidden-dim 128 \
  --device cuda \
  --eval-every 5 \
  --output outputs/full_trd_gpu_100k.json \
  --checkpoint outputs/full_trd_gpu_100k.pt
```

当前报告主表使用 `outputs/summary_trd_gpu_100k_match.json` 汇总最终结果。该评估是 sampled Top-K ranking：每个测试请求含 1 个真实下单商家和 199 个采样负候选商家，因此单个测试点的 Recall@K 是 0/1，最终 Recall@K 是所有测试点的平均命中率。

## 进一步提升结果

当前代码已在 `src/prepare_data.py` 中加入动态兴趣匹配特征：候选商家属性与历史时间衰减兴趣、点击临时兴趣的加权重合分数和比例。该特征会把 `basic_dim` 从 16 提升到 22，需要重新预处理并重新训练。建议在 GPU 环境运行：

```bash
python3 -m src.prepare_data --raw-dir data/raw --output data/processed/trd_gpu_100k_match.pkl \
  --max-train-orders 100000 --max-test-orders 10000 --eval-candidates 200 \
  --negatives 5 --max-food-attrs-per-poi 20 --max-interests 64 \
  --max-history 30 --max-clicks 20

python3 -m src.run_experiment \
  --data data/processed/trd_gpu_100k_match.pkl \
  --model full \
  --epochs 10 \
  --batch-size 2048 \
  --embed-dim 64 \
  --hidden-dim 128 \
  --device cuda \
  --eval-every 5 \
  --output outputs/full_trd_gpu_100k_match.json \
  --checkpoint outputs/full_trd_gpu_100k_match.pt
```

也可以尝试与 ItemCF 分数做后融合，通常能提高 Recall@5/10：

```bash
python3 -m src.run_hybrid \
  --data data/processed/trd_gpu_100k_match.pkl \
  --checkpoint outputs/full_trd_gpu_100k_match.pt \
  --cf itemcf \
  --device cuda \
  --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
  --output outputs/hybrid_itemcf_full_match_trd_gpu_100k.json
```

运行非神经 baseline：

```bash
python3 -m src.run_baselines --data data/processed/trd_small.pkl
```

编译报告：

```bash
cd report
xelatex main.tex
xelatex main.tex
```

## 模型变体

`src.run_experiment` 支持下列模型/消融：

- `mf`: Matrix Factorization。
- `static_kg`: 静态知识图谱商家表示，不使用动态兴趣。
- `kg_time`: 使用历史订单时间衰减边。
- `kg_time_temp`: 使用时间衰减和点击序列临时兴趣节点。
- `full`: Dynamic KG + Relation Attention + Ranking Layer。

`src.run_baselines` 支持：

- Popularity。
- UserCF。
- ItemCF。

LightGCN 的核心实现放在 `src/lightgcn.py`，可作为图协同过滤 baseline 扩展入口。

## 目录

- `scripts/download_trd.py`: 从 Zenodo 下载 TRD 文本数据。
- `scripts/make_demo_data.py`: 生成小规模字段兼容数据。
- `src/prepare_data.py`: 构建知识图谱、时间衰减兴趣、临时兴趣节点和训练样本。
- `src/model.py`: 动态知识图谱注意力推荐模型、MF。
- `src/datasets.py`: PyTorch Dataset 与变长兴趣/属性 padding。
- `src/metrics.py`: Recall@K、Precision@K、HitRate@K、NDCG@K、MRR、AUC。
- `src/run_experiment.py`: 神经模型训练与评价入口。
- `src/run_baselines.py`: Popularity/UserCF/ItemCF 评价入口。
- `src/explain_recommend.py`: 基于注意力和图谱路径输出可解释 Top-K 推荐。
- `report/main.tex`: XeLaTeX 中文实验报告。
- `report/main.pdf`: 已编译的 13 页实验报告。
