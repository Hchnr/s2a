# 原 18 条正确参考首次动作分歧的缓存审计

2026-11-16。对[原正确参考矩阵](reset_correct_reference_cohort_20261116.md)中
每条轨迹的首次机械臂动作分歧，比较保存的状态与局部模型候选。分歧前状态的
最大差为 `5.92e-14`，重算所选候选的模型误差最大差为 `4.53e-15`。

18/18 个状态下，历史成功控制器的八周期计划作为近同状态代理时，已存在于当前
搜索评估过的合格候选缓存中，并通过原关节、制动、速率及前两点余量筛选，以及
独立 witness 检查。将该候选优先排序会在 18/18 处改变所选计划；旧排序与新排序
分别合计使用 56 次局部模型响应，排序本身不增加响应。新选择的后续预测软质量
slack 增加 `0.00547–0.29905`，因此不能预设这会改善后续跟踪。

这里的历史计划只是近同状态的代理，尚未证明当前控制器必然逐值生成相同的
seed；冻结局部模型检查也不能证明实际闭环或任务成功。现有统一候选仍为原正确
参考 `14/18`；WAM 开发最近六例为 `0/6`，最终留出验收尚未执行。

## 变体与后续验证

新增默认关闭的[候选排序](../../../src/s2a/research/joint_control/preview/cached_seed_preference.py)：
仅当当前因果计算的完整 seed 已在原有合格缓存中时优先选它，否则沿用原排序与
fallback。原硬约束、独立 witness 和响应预算不变，未来六点质量仍是软目标。
[预登记](../../../plans/v0.0.1/tracking/joint_control/cached_seed_preference_20261116.md)
要求在原 18 条上从 reset 重新执行并独立重放，保留失败和原分母；本次提交尚未
执行该物理对照、WAM、新留出案例或 GPU 任务。入口的零物理预检已通过。

缓存审计相关 CPU 回归 52 项通过，候选单元回归 48 项通过，均无跳过；它们不
替代物理对照。下表收据位于本仓库 `outputs/`，不随 Git 提交。

| 收据 | SHA256 |
| --- | --- |
| `cached_seed_first_divergence_20261116/result.json` | `d7765166c7593b10c8d7b6c23b229ef9d14fc58d193a51c24f1c600db3bd5116` |
| `cached_seed_preference_dry_20261116.json` | `071b343830bffcc16bfcc49f5cbcb2ee9cbd9346b366f5d7656c33f3c6f8ba0c` |
| `qa/cached_seed_first_divergence_20261116/result.json` | `1e8290fbcba23ebdfd8ee0c16d2e90fdb61ff426618681ffc2f511deab22c5ca` |
| `qa/cached_seed_preference_unit_v2_20261116/result.json` | `36157683f638aaadec6c8e1495bd33be683bddc28f7dbc4eb0c1e1f1ae4d3412` |
