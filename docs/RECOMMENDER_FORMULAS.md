# FoodFlow 推荐模型公式与仿真数据说明

本文档说明 FoodFlow 当前使用的推荐模型、主要公式、分数配置，以及真实数据、mock 数据、经纬度、骑手和 ETA 的来源。

## 1. 默认推荐器

`foodflow.recommenders.build_recommenders()` 当前返回 7 个代表模型：

```python
[
    PopularRecommender(),
    BPRMFRecommender(seed=seed),
    UserOnlyRecommender(),
    build_learned_ltr_recommender(seed=seed), # LightGBM-LTR，缺少 LightGBM 时为 Logistic-LTR
    SeqTunedRecommender(), # 用于与学习排序进行对比
    SeqXQuadTripartiteRecommender(),
    SessionSpuTripartiteRecommender(),
]
```

其中 `build_learned_ltr_recommender()` 是学习排序入口：当前项目依赖中包含 `lightgbm==4.5.0`，默认会返回 `LightGBMRankerRecommender`；如果某台机器缺少 LightGBM，代码会显式退到 `LogisticLTRRecommender`，结果表显示为 `Logistic-LTR`，避免把规则模型误写成学习排序。`SeqTunedRecommender` 仍保留，作为可解释规则基线和学习排序对照。`SessionSpuTripartiteRecommender` 是在三方重排基础上新增的训练期 session 点击与 SPU 菜品画像增强模型。

## 2. PopularRecommender

Popular 是全局热门商家基线，不使用用户个性化特征。

设训练集中商家 `m` 的订单数为：

```math
c_m=\left|\{o\in\mathcal{D}_{train}:o.wm\_poi\_id=m\}\right|
```

推荐分数为：

```math
score_{pop}(u,m)=c_m
```

它的输出会作为其他模型的补位列表：当候选不足 Top-K 时，用 `popular_list` 补齐。

## 3. BPRMFRecommender

BPR-MF 是隐式反馈矩阵分解模型。用户点过的商家视为正样本，随机未交互商家视为负样本。

预测分数：

```math
\hat{x}_{ui}=\mathbf{p}_u^\top\mathbf{q}_i
```

成对排序目标：

```math
\max_{\Theta}\sum_{(u,i,j)\in\mathcal{D}}
\log\sigma(\hat{x}_{ui}-\hat{x}_{uj})-\lambda\|\Theta\|^2
```

当前默认配置：

```text
factors = 24
epochs  = 4
lr      = 0.035
reg     = 0.002
```

若用户在训练集中不存在，模型回退到 Popular 推荐。

## 4. UserOnlyRecommender

UserOnly 是可解释用户画像模型，使用品类偏好、复购、价格匹配、时段热度、商家质量和新颖性。

复购分：

```math
repeat(u,m)=\frac{\log(1+cnt_{u,m})}{\log 5}
```

品类偏爱：（其中 $c(m)$ 表示 m 属于c类）

```math
category(u,m)=\frac{N_{u,c(m)}}{\sum_{c'}N_{u,c'}}
```

价格匹配：（价钱区间）

```math
price(u,m)=1-\min\left(\frac{|p_u-p_m|}{\max(p_u,p_m,1)},1\right)
```

商家质量：（商家评分为 $s_m$，进行归一化）

```math
quality(m)=\frac{s_m-s_{\min}}{s_{\max}-s_{\min}}
```

商家热度：（$o_m$ 表示订单量）

```math
poplarity_{norm}(m)=\frac{o_m-o_{\min}}{o_{\max}-o_{\min}}
```

则有

```math
novelty(m)=1-poplarity_{norm}(m)
```

综合用户分：

```math
\begin{aligned}
score_{user}(u,m,p)=&
0.20\,category(u,m)+0.52\,repeat(u,m)\\
&+0.10\,price(u,m)+0.06\,period(p,m)\\
&+0.07\,quality(m)+0.05\,novelty(m)
\end{aligned}
```

该模型中复购权重最高，符合外卖场景中用户常点老店的特点。

## 5. SeqTunedRecommender

Seq-Tuned 是可解释序列规则基线。它用固定权重组合以下 7 个特征：

```text
fast_recency : 0.142601
slow_recency : 0.093624
repeat       : 0.412158
transition   : 0.247023
category     : 0.091250
popularity   : 0.008945
quality      : 0.004398
```

最近性衰减：（$r_{fast}$ 表示极近期的偏爱信号，$r_{slow}$ 表示中期的行为惯性）

```math
r_{fast}(u,m)=e^{-age(u,m)/6},\qquad
r_{slow}(u,m)=e^{-age(u,m)/12}
```

商家转移概率：

```math
T(a,b)=\frac{count(a\rightarrow b)}{\sum_c count(a\rightarrow c)}
```

最近 5 个历史商家的转移分：

```math
trans(u,m)=\max_{k=1}^{5}0.85^k\,T(s_{t-k},m)
```

最终序列分：

```math
score_{seq}(u,m)=
\frac{\sum_{f\in F}w_f f(u,m)}{\sum_{f\in F}w_f}
```

Seq-Tuned 的优点是可解释、稳定、速度快；缺点是权重固定，不能自动学习非线性特征交叉。

## 6. LightGBMRankerRecommender

LightGBM-LTR 是新增的学习排序模型。它复用 Seq-Tuned 的候选生成与序列特征，但不再使用硬编码权重，而是用 LightGBM 的 LambdaRank 目标学习排序函数。

### 6.1 输入特征

每个 `(user, merchant)` 样本的特征向量为：

```math
\mathbf{x}_{u,m}=
[r_{fast},r_{slow},repeat,transition,category,popularity,quality]
```

这些特征由 `SequentialHybridRecommender` 和 `_sequence_feature_values()` 生成。

### 6.2 训练标签

对每个用户 `u`，候选商家集合来自：

```text
最近历史商家 + 最近商家的转移邻居 + popular_list
```

标签定义为：

```math
y_{u,m}=
\begin{cases}
1, & m\in H_u\\
0, & m\notin H_u
\end{cases}
```

其中 `H_u` 是用户在训练订单中真实下单过的商家集合。训练时按用户分组，传入 LightGBM 的 `group` 参数。

### 6.3 排序目标

代码使用：

```text
objective = lambdarank
metric    = ndcg
```

模型学习一个非线性排序函数：

```math
\hat{s}_{u,m}=f_{\text{LGBM}}(\mathbf{x}_{u,m})
```

推荐时按预测分数排序：

```math
Rec(u,K)=TopK_m\left(\hat{s}_{u,m}\right)
```

当前默认训练配置：（训练时间较长可修改，但性能也会下降）

```text
n_estimators      = 80
learning_rate     = 0.05
num_leaves        = 15
min_child_samples = 20
subsample         = 0.85
colsample_bytree  = 0.90
max_train_users   = 1000
candidate_limit   = 160
```

当前项目环境已安装 `lightgbm==4.5.0`，默认输出会包含真正的 `LightGBM-LTR`。若运行环境没有安装 LightGBM，`build_recommenders()` 会改用 `LogisticLTRRecommender`，它基于同一组序列特征训练带类别均衡的 Logistic Regression 后备排序器，并在结果中明确显示为 `Logistic-LTR`。

## 7. SeqXQuadTripartiteRecommender

Seq-xQuAD-Tripartite 是三方重排模型。它把用户偏好、商家曝光公平、ETA 和供给可行性放入同一个排序器，并在列表层面加入多样性和长尾曝光。

XQuad：

经典 xQuAD：

$$
Score(d)=(1-\lambda)Rel(d)+\lambda Div(d)
$$

其中：

* Rel(d)：相关性
* Div(d)：多样性收益

商家公平分：

```math
fair(m)=0.75(1-popularity_{norm}(m))+0.25\,quality_{norm}(m)
```

推荐阶段 ETA 分：

```math
eta\_score(u,m)=1-\min\left(\frac{\widehat{ETA}(u,m)}{70},1\right)
```

供给分：（$delivery\_score_m$ 为商家的供给评分）

```math
supply(m)=0.6\frac{delivery\_score_m}{5}
+0.4\frac{1}{1+\log(1+count_m)}
```

三方综合分：

```math
\begin{aligned}
score_{tri}(u,m)=&
0.93\,user(u,m)+0.025\,fair(m)\\
&+0.03\,eta\_score(u,m)+0.015\,supply(m)
\end{aligned}
```

实际排序前会在当前候选集合内对四个分量做 min-max 归一化，避免用户分、曝光公平、ETA 和供给分的原始尺度不同导致权重不可解释：

```math
\widetilde{x}_{m}=\frac{x_m-\min_{c\in\mathcal{C}}x_c}
{\max_{c\in\mathcal{C}}x_c-\min_{c\in\mathcal{C}}x_c+\epsilon}
```

最终用于排序的是：

```math
score_{tri}(u,m)=
\frac{
0.93\,\widetilde{user}(u,m)
+0.025\,\widetilde{fair}(m)
+0.03\,\widetilde{eta}(u,m)
+0.015\,\widetilde{supply}(m)}
{0.93+0.025+0.03+0.015}
```

xQuAD 逐步选商家：

```math
m^*=\arg\max_{m\notin S}
\left[
0.84\,\widetilde r(m)
+0.12\,\mathbf{1}(cat_m\notin C_S)
+0.04\,tail(m)
\right]
```

其中：

```math
\widetilde r(m)=
\frac{score_{tri}(u,m)-\min_{c\in\mathcal{C}}score_{tri}(u,c)}
{\max_{c\in\mathcal{C}}score_{tri}(u,c)-\min_{c\in\mathcal{C}}score_{tri}(u,c)}
```

```math
tail(m)=\frac{1}{1+\log(1+count_m)}
```

## 8. SessionSpuTripartiteRecommender

Session-SPU-Tripartite 复用 `Seq-xQuAD-Tripartite` 的三方候选归一化和 xQuAD 列表重排，并额外接入 TRD optional 文件中的两类真实行为信号：

```text
orders_poi_session.txt      -> session_interactions.csv
orders_spu_train.txt        -> order_spus_train.csv
spus.txt                    -> SPU category
```

`orders_spu_test_label.txt` 会被预处理为 `order_spus_test.csv`，但推荐器训练和排序只使用 `order_spus_train.csv`；测试期 session 也会被过滤掉，避免把离线评估真值或测试前行为泄漏进模型。

### 8.1 会话点击信号

预处理阶段把 `orders_poi_session.txt` 中的点击商家序列解析成长表。真实 TRD 文件中的点击串可能用 `#` 分隔，mock 数据也兼容逗号分隔：

```text
wm_order_id, user_id, wm_poi_id, rank, split, dt, order_timestamp
```

推荐器拟合时只保留 `split=train` 的记录；如果存在 `order_timestamp`，先按用户和订单时间倒序排列，再按点击 `rank` 正序排列，最后对同一用户点击过的商家去重，得到训练期点击序列：

```math
S_u=[m_1,m_2,\ldots,m_n]
```

会话分数按点击位置做指数衰减：

```math
session(u,m)=
\begin{cases}
\exp\left(-\frac{rank_u(m)-1}{4}\right), & m\in S_u\\
0, & m\notin S_u
\end{cases}
```

其中 `rank_u(m)` 是商家在用户训练期去重点击序列中的位置。越靠前的近期点击商家，分数越高。

### 8.2 SPU 菜品画像信号

预处理阶段将 `orders_spu_train.txt` 与训练订单表合并，得到每个训练订单中的菜品 SPU：

```text
wm_order_id, user_id, wm_poi_id, wm_food_spu_id
```

推荐器再把 `wm_food_spu_id` 与 `spus.txt` 的 `category` 字段合并，分别统计用户菜品类目计数和商家菜品类目计数：

```math
n_{u,c}=count(user=u,spu\_category=c)
```

```math
n_{m,c}=count(merchant=m,spu\_category=c)
```

SPU 亲和度使用用户与商家菜品类目的重合比例：

```math
spu(u,m)=
\min\left(
\frac{\sum_{c\in C_u\cap C_m}\min(n_{u,c},n_{m,c})}
{\max(\sum_c n_{u,c},1)},
1
\right)
```

这个分数表示：候选商家过去卖出的菜品类目，是否覆盖了用户训练期真实点过的菜品类目。

### 8.3 候选集扩展

Session-SPU-Tripartite 的候选集合不是只来自序列模型，而是按顺序拼接三类候选并去重：

```math
\mathcal{C}_{session}(u)=\{m:m\in S_u,\ m\in active\_set\}
```

```math
F_u=Top3Category(n_{u,c})
```

```math
\mathcal{C}_{spu}(u)=
\{m:m\in active\_set,\ C_m\cap F_u\ne\varnothing\}
```

最终候选为：

```math
\mathcal{C}_{ssp}(u)=
dedup\left(
\mathcal{C}_{session}(u)
\oplus Top120(\mathcal{C}_{spu}(u))
\oplus \mathcal{C}_{seq}(u)
\right)
```

其中 $\mathcal{C}_{seq}(u)$ 是原 `Seq-xQuAD-Tripartite` 继承的序列候选，包括近期历史商家、转移邻居和热门商家补位。这样做的目的，是把“下单前真实点看过的店”和“菜品类目相近的店”先放进召回池，再交给三方重排处理。

### 8.4 分数组合与重排

该模型继承三方分量，但基础权重从 `Seq-xQuAD-Tripartite` 的 `0.93/0.025/0.03/0.015` 调整为：

```text
user_score        : 0.86
merchant_fairness : 0.025
eta_score         : 0.03
supply_score      : 0.015
session_score     : 0.055
spu_score         : 0.015
```

`component_scores()` 中记录的增强原始分为：

```math
\begin{aligned}
score_{raw}^{ssp}(u,m)=&
score_{raw}^{tri}(u,m;0.86,0.025,0.03,0.015)\\
&+0.055\,session(u,m)+0.015\,spu(u,m)
\end{aligned}
```

当前推荐路径中，候选排序仍继承三方重排器的候选内 min-max 归一化：`user_score`、`merchant_fairness`、`eta_score` 和 `supply_score` 被归一化后合成 `final_score`，`session_score` 与 `spu_score` 会保留在组件明细和 `raw_final_score` 中。也就是说，Session/SPU 的稳定影响主要体现在候选扩展和解释信号上；最终 Top-K 仍用同一套 xQuAD 逐步选择：

```math
m^*=\arg\max_{m\notin S}
\left[
0.84\,\widetilde r(m)
+0.12\,\mathbf{1}(cat_m\notin C_S)
+0.04\,tail(m)
\right]
```

因此它与 `Seq-xQuAD-Tripartite` 的区别可以概括为：召回侧加入训练期点击会话和菜品类目相似商家，分数组件中记录 session/SPU 证据，重排侧继续沿用三方公平、ETA、供给和列表覆盖逻辑。

## 9. 仿真策略

`foodflow.simulator.DEFAULT_POLICIES` 当前包含：

```text
Popular + Nearest
UserOnly + MinETA
Seq-Tuned + MinETA
LightGBM-LTR + MinETA
Seq-xQuAD-Tripartite + Greedy
Seq-xQuAD-Tripartite + Batch
Session-SPU-Tripartite + Batch
Session-SPU-Tripartite + Greedy
```

其中 `LightGBM-LTR + MinETA` 用来测试学习排序模型进入履约链路后的表现：推荐列表先由 LightGBM-LTR 生成，再用最小 ETA 策略派单。
`Seq-xQuAD-Tripartite + Batch` 和 `Session-SPU-Tripartite + Batch` 将同一时间步内产生的一批订单和在线骑手容量槽位构造成二分图，用最大权匹配替代逐单贪心，减少局部最优派单。仿真中请求用户流、选择噪声和初始骑手池使用固定 seed；同一推荐器下不同派单策略面对同一批订单，因此可以更公平地比较逐单与批量匹配。

骑手侧不再只使用静态距离最近规则。合成骑手包含 `speed_kmph`、`service_radius_km`、`acceptance_rate`、`reliability`、`load` 和 `available_at` 等状态。单个订单的骑手分为：

```math
score_{rider}(o,r)=
0.50\left(1-\min\left(\frac{ETA(o,r)}{80},1\right)\right)
+0.20\,rel_r
+0.15\frac{1}{1+load_r}
+0.15\,P(accept\mid o,r)
```

接单概率用骑手基础接单率、可靠性、服务半径、负载和 ETA 衰减构造：

```math
\begin{aligned}
P(accept\mid o,r)=\operatorname{clip}(&accept_r\cdot rel_r
\cdot e^{-\max(d_{pickup}+d_{delivery}-radius_r,0)/3}\\
&\cdot \frac{1}{1+0.35\,load_r}
\cdot \left(1-\min\left(\frac{\max(ETA(o,r)-35,0)}{80},0.75\right)\right),0.02,0.99)
\end{aligned}
```

批量派单将订单集合 `O` 和骑手容量槽位集合 `S` 构造成二分图。一个骑手 `r` 会按剩余容量展开为 `max_load-load_r` 个槽位，槽位上的有效负载随槽位序号递增，用来近似排队接单的额外压力。边权为骑手分减去超时风险：

```math
W_{os}=score_{rider}(o,r(s))-0.20\min\left(\max\left(\frac{ETA(o,r(s))-45}{60},0\right),1\right)
```

最大权匹配目标为：

```math
\max_x \sum_{o\in O}\sum_{s\in S}W_{os}x_{os}
\quad
s.t.\quad
\sum_{s\in S}x_{os}\le 1,\quad
\sum_{o\in O}x_{os}\le 1,\quad
x_{os}\in\{0,1\}
```

用户选择从固定概率改为 softmax/MNL：

```math
P(choice=m|u,t)=
\frac{\exp(V_{u,m,t})}
\exp(V_{\varnothing})+\sum_{j\in Rec(u,t)}\exp(V_{u,j,t})
```

其中 `V_{\varnothing}` 是不下单基准效用，`V_{u,m,t}` 由归一化推荐分、排序位置和测试集命中奖励组成。这样推荐分数会真实影响仿真订单分布，同时仍允许用户不下单。

批量派单使用线性分配求解：

```math
\max \sum_{o\in O_t}\sum_{r\in R_t}x_{or}W(o,r),
\quad
\sum_r x_{or}\le 1,\quad
\sum_o x_{or}\le 1
```

代码中用 `scipy.optimize.linear_sum_assignment` 在成本矩阵上求解；`load_aware` 策略把骑手可靠性、ETA 和当前负载合成为边权。

## 10. 轻量 KG 路径解释

`foodflow.kg` 不训练 LightGCN/KGAT，而是把现有结构化字段转成可解释路径：

```text
user --ordered_poi--> poi
user --prefers_category--> category <--has_category-- poi
user --orders_in_area--> area <--located_in_area-- poi
user --has_price_range--> price <--has_price_range-- poi
```

这些路径来自 `orders_train`、`users.favorite_category`、`merchants.primary_first_tag_id`、`merchants.aor_id` 和价格段。`foodflow explain-case` 会把路径证据、ETA 和曝光补偿一起输出，保证解释引用真实参与推荐/重排的字段。

## 11. 数据与仿真边界

真实推荐数据来自 Takeout Recommendation Dataset (TRD)，Zenodo DOI：`10.5281/zenodo.8025855`。项目使用用户、商家、菜品、训练订单、测试订单和测试标签文本文件。

TRD 不包含完整骑手状态、真实骑手轨迹和派单记录，因此骑手位置、负载、可用时间、可靠性和收入均为固定 seed 合成的 proxy，仅用于履约仿真。

用户和商家的经纬度也由离散区域字段合成：

```math
lng\sim\mathcal{N}(116.40,0.045),\qquad
lat\sim\mathcal{N}(39.92,0.035)
```

推荐阶段 ETA：

```math
\widehat{ETA}(u,m)=prep(m)+\frac{dist(u,m)}{18}\times60+peak(p)
```

派单阶段订单 ETA：

```math
\begin{aligned}
ETA=&wait+prep+peak
+\frac{dist(rider,merchant)}{speed_r}\times60\\
&+\frac{dist(merchant,user)}{1.08\,speed_r}\times60
+5\cdot load
\end{aligned}
```

这些仿真变量用于比较推荐策略对履约时间、超时率、骑手负载、骑手收入分布、活跃骑手比例和平台效用的影响，不声明为真实骑手数据。

由于公开数据集中没有真实配送轨迹和配送时长，因此采用启发式 ETA 估计模型，仅用于构造履约约束和比较不同推荐策略下的相对表现，而非追求 ETA 的绝对预测精度。

## 12. 骑手数据仿真待办

* [ ] TODO：可以增加ETA `estimate_user_merchant_eta` 中对配送速度等估计值进行扰动，衡量模型敏感性
* [ ] TODO：骑手模拟时貌似没有到商家取餐的计算？
