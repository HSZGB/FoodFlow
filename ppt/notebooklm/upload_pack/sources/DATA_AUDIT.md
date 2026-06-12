# FoodFlow 数据审计

## 结论

- 必需 TRD 原始文件是否齐全：`True`
- 当前训练集处理模式：`full`
- 原始训练订单数：`1068495`
- 处理后训练订单数：`1068495`
- 训练订单使用比例：`1.0000`
- 骑手数据边界：TRD 不包含完整骑手状态与派单记录，本项目使用固定 seed 合成 proxy。

如果 `train_mode` 为 `sampled`，说明当前结果使用真实 TRD 数据的固定 seed 抽样版本；如果为 `full`，说明当前处理后的训练订单覆盖完整 `orders_train.txt`。

## 原始 TRD 文件

| file                      | exists   |    bytes |    rows |
|:--------------------------|:---------|---------:|--------:|
| users.txt                 | True     |  5165902 |  200000 |
| pois.txt                  | True     |  3120680 |   29072 |
| spus.txt                  | True     | 18249946 |  179778 |
| orders_train.txt          | True     | 60901522 | 1068495 |
| orders_test_poi.txt       | True     | 10660405 |  230550 |
| orders_poi_test_label.txt | True     |  6709915 |  230550 |

## 处理后文件

| file                  | exists   |    bytes |    rows |
|:----------------------|:---------|---------:|--------:|
| users.csv             | True     | 16543513 |  200000 |
| merchants.csv         | True     |  4697590 |   29072 |
| spus.csv              | True     | 16908698 |  179778 |
| orders_train.csv      | True     | 73900028 | 1068495 |
| orders_test.csv       | True     | 12841724 |  230550 |
| test_interactions.csv | True     |  4404411 |  230550 |
