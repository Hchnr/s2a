# S2A 实现进度核查 — 2026-10-23

核查时间为2026-10-23 10:51（Asia/Shanghai），工作区为`/share-mixed/project/eai_pwm/home/hcr/repos/s2a`，Git HEAD为`59b12b9fcf2aabd3e436adb12cec075de410d85a`，包含已有未提交实现。读取源码、正式报告及最新运行产物，另行完成定向CPU验证；快照与指纹见[本次snapshot](../outputs/status_review_20261023/snapshot.json)。

**当前处于“可执行S2A与独立仿真正控已建立，真实WAM多任务稳定效果尚未达标”的阶段。** 新训练数据已合格，真实clean桥接曾启动但尚无完整资格结果。总体目标与后续验收统一见[036总目标](../plans/v0.0.1/036_project_goal_20261023.md)。

## 实现已经推进到哪里

| 层次 | 已实现或验证 | 仍存在的边界 |
| --- | --- | --- |
| 数据与几何 | DROID来源对齐、物理时钟/mask、column-rot6d、SE(3)积分、TCP/flange和SO(3)误差；跨数据集日志核验 | 表示转换正确不保证执行正确；历史控制器和部分上游同步信息仍有未知项 |
| 控制算法与在线接口 | 解析接口、ridge局部响应及正则逆、P/PI/PD/PID、最终命令限幅/anti-windup、状态隔离、固定锚点轨迹和异步接入 | DROID局部模型中的反馈候选未通过全部扰动门槛；原计划增益调度、直接逆MLP、6D arm残差MLP未完成独立对照 |
| 独立仿真 | LIBERO/MuJoCo固定OSC执行、原生命令/oracle正控、夹爪/旋转/重规划和因果日志检查 | 有实际物理仿真证据，适用范围限于所测控制合同；尚无真机证据 |
| 世界模型接入 | 固定checkpoint图像闭环、Action128 mask修正、state诊断、数值桥接、readout及自由轨迹诊断 | 有单例任务成功，缺多任务、新初态的稳定效果；旧teacher改善未通过完整生成推广门槛 |
| 最新训练准备 | 034同步native数据合格；clean残差、捕获桥接和流式ridge代码及CPU测试已实现 | 035真实sampler资格未完成；新clean路径无真实拟合或任务收益 |

源码依据：[geometry.py:14](../src/s2a/geometry.py#L14)、[runtime.py:28](../src/s2a/runtime.py#L28)、[control.py:83](../src/s2a/control.py#L83)、[identification.py:36](../src/s2a/identification.py#L36)、[rollout.py:96](../src/s2a/rollout.py#L96)、[async_reference.py:69](../src/s2a/async_reference.py#L69)。未执行的控制候选见[rollout.py:333](../src/s2a/rollout.py#L333)，其余同段“未提供仿真/WAM”仅描述原DROID单episode实验，不代表后续整个仓库仍无这些能力。

按001阶段划分：**P0–P3实现和首轮实验完成，P4部分完成并有大量独立仿真扩展，P5真机未执行**。P4中的同设备多episode学习逆/增益调度公平对照仍缺；不能将新增WAM readout计作该项完成。[001阶段:348](../plans/v0.0.1/001_state_to_action.md#L348)、[首轮结果:51](droid_episode0_20261016.md#L51)、[clean输出定义:129](../src/s2a/wam_clean_readout.py#L129)

## 关键效果证据

| 实验 | 实际结果 | 说明 |
| --- | --- | --- |
| 2026-10-20统一独立矩阵 | native 24/24、oracle＋固定S2A 24/24稳定；8400 0/24、8800 1/24稳定 | 正确参考执行已有正控，真实WAM端到端未达标 |
| 旧开发场景raw8复测 | seed42为1/3、seed43为0/3稳定 | 单例未形成重复性，不能按任务或seed拼接策略 |
| 027→028→029适配诊断 | 027 teacher改善57.45%，028完整生成gate=False；029第二seed teacher误差+51.42%，free clean误差+1.60% | 中间拟合指标不能代替生成与任务验证 |
| 原030与新034数据 | 原030为28/30、训练资格False；新034为30/30、每任务10/10，6180步/150当前canvas | 034通过独立审计，只取得后续训练协议登记资格 |
| 最终留出验收 | 12个新初态已预留，目录内无性能result | 24条WAM最终性能验收仍未执行；完整接入也有待完成 |

依据：[独立矩阵:7](wam_independent_20261020.md#L7)、[第二seed:3](wam_raw8_seed43_20261021.md#L3)、[029:19](wam_free_trajectory_diagnostic_20261021.md#L19)、[034:7](settled_native_training_20261022.md#L7)、[final peer:28](settled_native_training_20261022.md#L28)、[最终验收准备:3](wam_fresh_acceptance_preparation_review_20261021.md#L3)。本次按文件读取与指纹核对既有证据，没有重新采集物理结果或重新拟合模型。

## 035的最新状态修正

2026-10-22报告末尾“尚未初始化新模型”的描述已落后于产物，保留该历史报告，当前判断更新如下：

- [launch.json:3](../outputs/wam_stability_20261021/clean_bridge_qualification/orchestration/model/launch.json#L3)记录worker PID791403、owner PID791277及实际命令；使用允许的`GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`，CUDA/LD设置有记录。
- [command.log:4](../outputs/wam_stability_20261021/clean_bridge_qualification/orchestration/model/command.log#L4)记录VAE加载，末尾9–11行记录meta→CUDA及`set_up_model`阶段，说明真实初始化已经开始；不能由此推断8400权重完整加载或7个请求完成。
- 现有所有权记录仍为[pending](../outputs/wam_stability_20261021/clean_bridge_qualification/orchestration/model/ownership/pending.json)，`qualification/`无文件，缺完整结果、独立verified、wait/reap及GPU释放收据；本次`/proc`核查两个记录PID均不存在。

因此准确状态是：**曾启动并进入初始化，现有运行证据未闭环，终止原因未知，尚未获得真实桥接资格。** 不把缺少收据记为成功，也不推断为某个确定错误。近期先收尾既有运行证据，再按明确预算完成真实资格；本次未重启或停止任何模型进程。

## 工程检查与未完成项

本次显式`CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=''`执行：

| 检查 | 结果 | 产物 |
| --- | --- | --- |
| Ruff check src/tests/scripts | 通过 | [ruff_check.log](../outputs/status_review_20261023/ruff_check.log) |
| Ruff format --check src/tests/scripts | 413文件通过 | [ruff_format.log](../outputs/status_review_20261023/ruff_format.log) |
| 核心几何/控制/辨识/runtime/异步/计分及settled/clean桥接共15个测试模块 | 186 passed，13.91s | [pytest.log](../outputs/status_review_20261023/pytest.log) |

定向测试使用独立basetemp，没有执行全仓pytest。迁移后此前完整QA的1 failed/12 errors仍是已知未消除边界，主要涉及已删除旧根的历史fixture/审计入口；后续950 passed支持子集和本次186 passed都不能替代全仓通过。[迁移报告:14](workspace_relocation_20261022.md#L14)、[034 QA:30](settled_native_training_20261022.md#L30)

除任务效果外，待完成项还包括：迁移后的历史入口修复与完整QA、固定候选与fresh专用首请求来源/矩阵入口、实时延迟验证和真机oracle/预测参考闭环。当前仿真会暂停物理等待模型，不构成实时部署证据。[最终接入审查:15](wam_fresh_acceptance_preparation_review_20261021.md#L15)、[002:3](../plans/v0.0.1/002_wam_stability_20261021.md#L3)

后续顺序为：**035真实资格 → 有限真实训练及完整生成/开发任务评估 → 冻结候选和最终入口 → 原预留24条验收**。总目标保持总体≥20/24、各任务≥4/8、末40端点持续成功，完整定义见[036](../plans/v0.0.1/036_project_goal_20261023.md)。
