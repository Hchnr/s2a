# 九条训练恢复轨迹的 H40 数据合同审计

2026-11-16。按[预登记](../../../../plans/v0.0.1/wam/training/recovery/recovery_h40_contract_20261116.md)，
对[九例独立重放合格轨迹](current_controller_recovery_repair_20261116.md)的全部
27 个实际图像 anchor，分别重建其后 40 个真实动作标签。**全部窗口通过当前
状态/图像绑定、20 Hz、SITE/BODY→TCP、mask、实测标签与原 D0 归一化合同检查；
原归一化对 39 个位置标量发生裁剪。** 这是数据合同审计，不授予拟合资格。

| 项目 | 结果 |
| --- | --- |
| 实际来源 | 3 个原训练 demo：bowl/demo_8、wine/demo_1、cream/demo_2 |
| 按三个偏移采集 | 9 条实际轨迹；每条 anchor 为 0、16、56 |
| H40 窗口 / 含重叠标签 | 27 / 1,080 行 |
| 不同的 case/时间转移 | 864；其余 216 行因窗口重叠而重复 |
| normalization31 裁剪 | 共 39 个活跃槽标量；位置槽 1 有 15，位置槽 2 有 24，其余为 0 |
| 27 窗口内下一端点实测开度 | 范围 `0.3500983312–0.9922121847`，无 `<0.35` 端点 |
| 实际发送的夹爪命令 | 取值 `-1、0、1`；单独保存，不能替代实测开度标签 |

39 个裁剪均发生在 anchor 56 的 6 个窗口：bowl 三个偏移各 5 个位置槽 1，
cream 三个偏移各 8 个位置槽 2；零偏移与非零偏移都包含在内。统计分母为
1,080 行 × 10 个活跃槽，含窗口重叠；裁剪由原冻结变换执行，未改阈值或
删除样本。`<0.35` 仅复用此前[开度事件诊断](../causal_opening_event_20261111.md)
的描述阈值，不等同抓取或任务成功，也不能说明整个轨迹没有闭合。

三个 demo 的任务、demo 名与初态 SHA 对应原训练准备，均未命中其已登记的
forbidden demo 或初态 SHA；同一 demo 的三个偏移和所有窗口必须归为同一数据
组，不能随机拆成训练/验证独立样本。该检查不排除未知的近重复。原准备文件
中的 `validation_initial_sync_certified=false` 针对旧 STATE 数据保持原样；
后来另建的[053 六案例](../../qualification/wam_synchronous_validation_20261024.md)
已完成实际同步采集与独立审核，其初态身份和本次三组的关联应单独核对，
不能把旧标记误写成 053 不存在。

## 证据与边界

每个窗口使用 41 个实际端点，独立调用原训练 `post_capture_labels` 重建
40 行并与采集保存的 Action128 逐值相等；归一化使用 D0 实际的 domain、
normalization31、projection0 和 FP32 clipping。27 张当前图像按各自的真实
状态哈希与时钟复核，画布独立重建最大通道差均为 1。所有消费文件前后 SHA
一致；审计源码 `fd1f95c2`。32 并发专项 37 项、64 并发相关回归 142 项均
通过且 0 跳过；全仓 Ruff lint 和格式检查通过。

| 本仓库 `outputs/` 路径 | SHA-256 |
| --- | --- |
| `recovery_h40_contract_20261116/result.json` | `0753c4c30f432e6b09e103152238cb4f6f47dffa30ce76e60636e59f2329dcec` |
| `qa/recovery_h40_contract_unit_20261116/result.json` | `ca7b53574d0b10edc57846490b6854e174baeb41eb79025dd0ef0c138250087b` |
| `qa/recovery_h40_contract_final_20261116/result.json` | `dbd9964081de0b00ec0eeca4221a86fb09f9b0a3bfa2dbc1a2c67e5a85e35bdf` |

当前窗口主要覆盖初始偏移、返回及 native 起始段；是否补充实际接触阶段
anchor、裁剪如何影响拟合、训练/验证如何分组和加权仍需另行固定并验证。
本项无新 WAM、GPU、物理步、fit 或 fresh 评估；标签可重建不等于模型能生成
恢复轨迹，也不改变原 24 条最终留出验收门槛。
