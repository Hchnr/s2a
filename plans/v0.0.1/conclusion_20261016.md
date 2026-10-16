# v0.0.1 结论 — 2026-10-16

本轮已完成 P0/P1、P2/P3 的实现、测试及可访问数据上的离线实验。结论是：**几何转换可精确验证；当前单 episode 响应模型与候选反馈没有达到完整的稳健控制验收，不提供真机部署增益。** P4/P5 的条件尚不具备，未执行的项目明确保留，不计为通过。

实现对应实验 commit `b82b0b619dcc8725182aecf6b6d59d065eb27eb1`，启动时工作树干净；后续提交仅维护验证入口、报告与文档。完整数据/代码 SHA256 和实际依赖版本保存在 `outputs/droid_episode0/sample_manifest.json`。seed=42；Python 3.12.3、NumPy 2.1.0、SciPy 1.17.1、PyArrow 23.0.1。

## 交付与验证状态

| 项目 | 状态与证据 |
| --- | --- |
| P0 数据合同/抽取 | 完成；300 转移，304 观测，来源/物理时钟/mask/夹爪检查通过 |
| P1 几何与参考接口 | 完成；单步/累计/TCP-flange 几何检查通过，历史控制器匹配未确认 |
| P2 响应辨识/逆解/PID | 实现并执行；144 个候选只在 fit 拟合、validation 选择；不将 rank=6 当全局可辨识 |
| P3 模型内对照 | 完成；两种前馈 × 5 种方法 × 7 场景，70 个 test rollout 全部有效，8 个反馈候选均未通过全扰动验收 |
| 独立软件测试 | **23 passed**；真实几何、因果性、状态与约束不变量，以及推进新状态的合成控制测试 |
| 外部数据集成测试 | **1 passed**；显式读取配置，默认 deselected，未把 deselected 当通过 |
| 静态/打包 | Ruff lint、format check、git diff check、wheel build、独立安装 wheel 的 runtime smoke 均通过 |
| 最终命令约束 | 全部 70 个 test trace 的最终向量范数与变化率界通过；原始/送出/进入 plant 的命令分别保存 |
| P4/P5 | 未执行；缺少已确认同设备多 episode、匹配控制器的独立仿真、真机与世界模型预测 chunk |

实现入口：[数据审计 data.py:121](../../src/s2a/data.py#L121)、[解析验证 analytic.py:18](../../src/s2a/analytic.py#L18)、[在线 adapter runtime.py:11](../../src/s2a/runtime.py#L11)、[PID control.py:83](../../src/s2a/control.py#L83)、[最终命令限速 interface.py:50](../../src/s2a/interface.py#L50)、[时序选择 study.py:68](../../src/s2a/study.py#L68)、[模型 rollout rollout.py:29](../../src/s2a/rollout.py#L29)。精确仓库路径均为 `src/s2a/...:line`。

## 五个研究问题的回答

**1. observed local delta 能否还原下一状态并转换到 flange？可以，针对当前发布契约。**

| 检查 | 最大位置误差 m | 最大 SO(3) 误差 rad |
| --- | ---: | ---: |
| 单步重建 | 5.7105e-8 | 5.5215e-8 |
| 全段累计积分 | 4.1635e-8 | 3.7667e-8 |
| TCP → R1 实测 flange | 3.0398e-8 | 3.0412e-8 |

均低于计划 `1e-5 m / 1e-4 rad`。TCP 力臂完整保留，未把 rot6d 当 Euler；夹爪反向编码最大误差 `2.9802e-8`。全 episode 末行 `action_valid=true` 但 pose 无效，已按 pose mask 与物理时间单独排除。[指标依据](../../outputs/droid_episode0/geometry_metrics.json)。

**2. 还原状态与原 commanded target 是否相同？不同。线性校准只出现局部诊断改善。**

全选段 observed 下一状态相对历史 commanded target 的位置/姿态 RMSE 为 **0.021375 m / 0.055780 rad**。公开 DROID 参考接口用原观测及原 velocity 解码历史 target 的 RMSE 仍为 **0.005183 m / 0.013447 rad**。无法仅凭日志确定是历史版本/模式差异，还是 command 构造时 robot_state 与导出 observation 的时刻偏移。

命令 NRMSE（尺度只来自 fit；共同支持 fit/validation/test=172/32/52）：

| 区间 | hold/zero | previous command | A | linear calibrated |
| --- | ---: | ---: | ---: | ---: |
| validation | 2.085350 | 0.165128 | 1.627156 | 1.766943 |
| test | 1.614890 | 0.080250 | 1.405228 | 1.125336 |

线性校准在 test 相对 A 改善约 19.9%，validation 未改善，且均远弱于 previous-command 诊断基线。该指标不支持闭环成功或接口匹配结论。[指标依据](../../outputs/droid_episode0/calibration_metrics.json)。

**3. 相同参考与扰动下 P/PI/PD/PID 改善什么？解析前馈的反馈降低模型内 tracking error，I 有局部收益，D 未显示明确收益；完整验收仍失败。**

以下为 test 初始偏差场景，所有方法使用同一解析前馈，warm-up 后计分 52 转移；位置单位 m、旋转单位 rad、变化率为归一化 command 的逐分量 RMS/s。

| 方法 | 位置 RMSE | 姿态 RMSE | 平衡 tracking score | 命令变化率 |
| --- | ---: | ---: | ---: | ---: |
| FF | 0.053972 | 0.313654 | 5.851565 | 0.068481 |
| P | 0.020179 | 0.078919 | 1.811524 | 0.104428 |
| PI | 0.018210 | 0.060609 | 1.546856 | 0.106032 |
| PD | 0.020166 | 0.079383 | 1.814833 | 0.104340 |
| PID | 0.018211 | 0.061008 | 1.550027 | 0.105885 |

PI 在此场景的 tracking score 下降约 73.6%，但变化率增加约 54.8%，超过预设 25% 的相对恶化界；增益变化、恒扰动及静止目标还有饱和约束失败。最终绝对限速是接口约束，不能据其通过而忽略相对 FF 的平滑性验收失败。

线性逆前馈的 test 初偏位置 RMSE 约 0.602–0.609 m，最终限制介入比例高；反馈没有达到 tracking score 降低 20% 的标准。它不适合作为当前版本的首选前馈。完整逐场景结果、稳态误差、超调、调节时间、饱和时长、选定增益和失败条目见 [完整报告](../../reports/droid_episode0_20261016.md) 与 [模型内指标](../../outputs/droid_episode0/closed_loop_model_metrics.json)。这些都不是独立物理仿真或真机证据。

**4. 数据能否支撑响应辨识与可靠整定？支持单轨迹的局部预测研究，不足以确认可靠控制。**

validation 选定 diagonal、history=3、lag=0、ridge=.0001，fit 有效样本 177。B scaled condition≈69.16，最小 scaled singular value≈.00237，瞬时 z 响应系数为负；history 项贡献明显，不能把模型当六轴静态正增益。

| 区间 | 一步 NRMSE | previous-motion NRMSE | 自主多步 NRMSE | 多步位置/姿态 RMSE |
| --- | ---: | ---: | ---: | --- |
| validation | 0.130463 | 0.220406 | 0.690299 | 0.053655 m / 0.192583 rad |
| test | 0.089895 | 0.122318 | 0.771023 | 0.101210 m / 0.344457 rad |

一步指标优于 persistence，但多步误差明显增加；P2 gate 只允许进入模型内探索。当前模型缺少独立关节/接触动态，不支持把扰动外推当真实动力学。依据方案第 9 节，本轮以明确识别这一限制收束，不再扩网格追 test 或训练小网络。

**5. 推进独立仿真/真机还缺什么？**

- 可追溯的机器人序列号、同设备多条 episode 和相同历史控制配置。
- 真实 action_space、独立 gripper 模式、controller/IK 版本及参数、command 构造时的 robot_state 和发送/反馈时间。
- 反馈新鲜度、掉帧状态、接触/力或可信接触分段，实际外环周期与限幅。
- 匹配该固定控制接口的独立模拟器或真机执行通道；当前代码不发送硬件。
- 冻结世界模型已生成且带明确执行时间轴的 delta chunk，用于 oracle 之后的第二阶段验证。

## 保守决策与复现

每个候选/场景/划分独立初始化历史、delay queue、I/D 状态，部署跨 chunk 延续；首帧不积分，不反复积分重复反馈，stale 反馈冻结，大间隔/时间倒退重置，接近 π 拒绝并要求小旋转重规划。`dt_command=0.070980216 s` 仅取 fit median，不读取未来实测 dt。正则逆后的范数/变化率投影明确标为投影，不称为约束最优解。

开发期间发现并修复两项真实问题：额外扰动应延迟 plant 输入而非整个自回归输出；最终编码后的 command 也必须限变化率并纳入 anti-windup。错误版本的产物留档，候选网格与性能阈值未因 test 调整。详细依据见 [实施记录](record_20261016.md)。

```bash
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage all --plots
LD_LIBRARY_PATH='' .venv/bin/python -m pytest
LD_LIBRARY_PATH='' .venv/bin/python -m pytest tests -m integration --data-config configs/droid_episode0.json
LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_artifacts.py --seed 42
```

完整产物位于 `outputs/droid_episode0/`；Git 内保留 [报告](../../reports/droid_episode0_20261016.md)、[精简指标](../../reports/droid_episode0_20261016.json)、[验证摘要](../../reports/validation_20261016.json)。本轮未执行项是已确认的证据边界，不是测试通过项。
