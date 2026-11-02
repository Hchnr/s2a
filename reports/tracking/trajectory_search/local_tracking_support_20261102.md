# 完整跟踪瓶颈：终点旋转的局部支持下界

2026-11-02。small-progress未改善完整caps后，执行保存模型诊断，而非再调整
进展阈值。对应[设计与预算](../../../plans/v0.0.1/tracking/trajectory_search/local_tracking_support_20261102.md)。

## 问题与方法

固定已独立核验的small-progress anchor0、最终anchor23，重建各自因果Js/Je。
每个超标位置/旋转三维误差按原.01m/.05rad归一化，以当前单位误差方向n构造
支持函数nᵀ(E+Je*d)。分别保留原实际buffer和内部reserve .9，各12个超标cap，
共48LP。保留原root命令域、.25变化率及4369线性安全行；移除局部trust radius
及pose ceiling，形成比生产更乐观的线性松弛。原生产约束、预算和记录未改。

任何满足某cap的局部线性解均须使该支持值≤1；某支持下界>1可排除这个模型
中的单cap可行性。反之，下界≤1不证明该cap球可行，更不证明共同24caps可行。
这些模型来自有限差分和secant，跨域外推没有非线性误差界，因此不能据此证明
真实控制或物理不可达，也不能据此修改原参考或放宽验收。

## 实际发现

| anchor | 安全floor | 终点旋转支持下界 | 超过1的cap数量 |
| --- | --- | ---: | ---: |
| 初始0 | 原buffer | 1.4454572083 | 1/12 |
| 初始0 | reserve .9 | 1.4456716199 | 1/12 |
| 最终23 | 原buffer | 1.4519795340 | 1/12 |
| 最终23 | reserve .9 | 1.4521963830 | 1/12 |

终点当前误差norm约2.62；即使去掉局部半径和pose ceiling，两个局部模型都不能
在原安全线性约束和原root命令域下将其支持压至1。原buffer与reserve .9的差
约2.1e-4，远小于距达标约.45的差距。这不支持把内部reserve门视为终点旋转
主要瓶颈，支持把后续工作转向模型与路径，而非继续降低进展阈值。

其余11项乐观支持均≤1，但彼此使用不同方向，不能视为共同可行。第三点旋转
仍超原cap，prefix2的问题也未解决；终点瓶颈不是所有失败的唯一根因声明。

终点原buffer LP中，非零安全对偶集中在12条braking_upper及terminal，
inner上下限和braking_lower对偶为0；命令上下界、rate上下界也有非零对偶。
最终anchor的最大对偶行4360是braking_upper(period10,joint6，均零基索引)，
此外4353/4346/4339等行共同参与。不能仅按最大对偶推断唯一应修改的约束，
这些行全部保留。原当前最小raw slack在period5/joint4，与最大对偶行不同。

LP乐观点最大命令步接近.2，而原初始化探针尺度为1e-4。这是显著模型外推；
当前下界最直接暴露的是局部模型的完整可行性瓶颈，尚不判定真实机器人能力。
下一步固定检查该乐观方向的非线性响应及安全预测误差，先验证模型跨域可信度，
再决定是否需要新的非线性路径/参考可执行性方法。不能直接把LP点下发或采用。

## 审核、费用和来源

48/48LP正常完成，累计818次solver迭代。独立原单位primal/dual/stationarity/
gap审核通过；最大相对stationarity7.84e-16、gap6.31e-16。对偶负值投影为0后，
剩余stationarity按原有限命令box扣除最坏项，使用扣除后的支持下界；终点修正
约3e-16。求解器success不单独作为合格依据，超时/失败不自动重试。

38项jobs32专项通过，无skip/collection错误，worker回收、来源不变；Ruff/format
及Git空白检查通过。source/input SHA前后不变。CPU空CUDA/LD、数值库单线程。
tool session7059退出0并实际wait/reap，PID273635不存在。新增0响应/model/查询/
内步/QP/NNLS/物理/GPU/WAM/fresh/最终留出/实际参考消费。

以下路径相对outputs：

| 产物 | 路径 | SHA256 |
| --- | --- | --- |
| 完整48LP收据 | local_tracking_support_20261102/result.json | f029085ade6a4c846fffb50ce0eba294b0a87cbbf2aedc5cf694192555d90203 |
| 对偶分组与汇总 | local_tracking_support_20261102/summary.json | 3c0d335b756ea8eece24823f02437007ab15a50e0af434bc6b815c50a22f6c77 |
| owner终态 | local_tracking_support_20261102/owner_observation.json | 94359eab2654638ea8205115474b389436425e11d08621bd9bf28ff94aaad945 |
| 38项QA | qa/local_tracking_support_20261102/result.json | 18943b69a3f783eb6aa084562bd52b735b2389874687a99c3769b96ec3121439 |

当前资格仍cursor8/task0/grasp0，总目标和18/12/24完整验收未完成，最终留出关闭。
