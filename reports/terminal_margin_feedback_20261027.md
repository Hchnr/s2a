# 末端制动条件：模型核验、容差修复与完整反馈结果

2026-10-27。总目标未完成。末端非向外速度条件在模型中可行，但完整原生反馈未改善上一版结果，也未产生抓取或任务成功。

## 模型与实现

`src/s2a/terminal_joint_plan.py` 在原H8动作范围、.25变化率、全部200个2ms关节位置.05rad内侧约束及原首步TCP限额上，增加末端速度线性约束。法向来自原seed自由预测中的近边界关节：上边界+e、下边界-e，要求对应末端signed qdot<=0，不强制其他关节停下。

两个72步原生快照都找到模型见证。raw最小内环余量.051008723rad、末端q7速度约-7.85e-16rad/s；位置/旋转限额保持不变，优化39迭代/76响应成功。terminal058无近边界法向，37迭代/73响应成功；该强制模型求解不替代实际旁路。

独立 `scripts/verify_terminal_joint_plan.py` 从原生状态重新取得M/J/q/qdot，以原关节模型逐周期重建并复算每个内环位置和末端速度，独立重建风险法向并检查符号；全部通过。它证明局部冻结模型见证，不证明真实不变集、递归可行性或实际制动。

- 模型producer `outputs/terminal_joint_plan_probe_20261027/verified.json`，SHA256 `1c5380ac98c391d95be4b8b6f0cc3a65b5e9a17a1c0b696d7a1bb5fb281c2ddc`，实际exit0。
- 独立模型核验 `outputs/terminal_joint_plan_verified_20261027/verified.json`，SHA256 `f0f150af09b95d49823b20fdea310f3a1aab63b38b843c796287a2acc02f18e0`，实际exit0。

原生选择顺序仍为固定余量阶段→自适应正余量阶段→软回退。只有固定余量阶段强制新增末端速度条件；回退明确terminal_constraint_enforced=false，不冒称满足制动条件。原TCP首步限额、XML、控制器/initial_joint/夹爪、原参考和时间表不变。

## 容差缺陷与v2修复

首版raw执行72步前缀和14步分支后，在发送控制步86动作前被外层TCP检查拒绝，terminal案例未启动。原因是子求解器最终候选使用归一化pose_values>=-1e-8，允许某些旋转误差绝对超额大于外层1e-10rad。外层正确停止，没有发送该候选；首版源与输出完整保留，owner/child实际exit1，actual wait/reaped收据已核验。

`src/s2a/pose_cap_check.py` 与两个strict子求解器改用米/rad绝对误差，内部1e-11、外层仍1e-10，保留舍入余量，未放宽外层阈值。v2 selector/native/worker为独立文件，旧源SHA不变。新版本从原初态完整重跑两分支，不将旧14步和新结果拼接，也不是自动重启旧失败动作。

相关14测试文件43 passed in11.61s，实际exit0。覆盖上下边界制动符号、末端速度仿射性、固定/自适应/软选择、历史隔离、绝对单位容差及此前几何/导数合同。模型准入阶段先前32/35项测试亦通过；完整结果以修复后的43项为准。

## v2完整实际结果

同一72至128窗口：72步原命令前缀，查询9/10各前8步，查询11完整40步。只使用当时当前保存chunk，沿用anchor；不计候选自身新图像驱动的世界模型闭环。

| 案例 / 方法 | 56步位置 RMSE mm | 旋转 RMSE rad | 首次q7超XML端点 | 超范围周期 | 抓取 / 成功步 |
|---|---:|---:|---:|---:|---:|
| raw baseline | 51.0314 | .178797 | 82 | 47 | 0 / 0 |
| raw terminal-v2 | 40.2581 | .178525 | 89 | 40 | 0 / 0 |
| terminal058 baseline | 10.2800 | .036863 | 无 | 0 | 0 / 0 |
| terminal058 terminal-v2 | 10.2800 | .036863 | 无 | 0 | 0 / 0 |

raw有11次固定余量+末端条件见证、2次自适应见证、41次软回退（11次保留seed）、2次无风险旁路。11条模型计划所选法向末端速度均通过<=1e-10检查；这不是实际执行了整条计划或真实末端速度为0。43个优化周期未满足原.05rad目标。

上版无末端条件的自适应方案同窗口39.1418mm/.177194rad、首次超XML89；本次两项误差都略差，且没有进一步推迟超范围，不支持推广该附加条件。最后40步本次47.4491mm/.209809rad，baseline60.2437mm/.210715rad，不能与56步均值混用。

terminal058全部56步命令、完整sim_state/hash、goal、夹爪、对象/任务字段与baseline精确一致。raw适配器平均2.49685s，terminal旁路.007940s，raw仍不满足.05s实时周期。XML范围比较不等于实际限位反力或真实机器人失控判定。

## 失效含义与后续

第85步固定阶段线性成功，但完整搜索以Positive directional derivative退出；随后只有85/86两步自适应余量见证，87步完整搜索无见证，88步忽略TCP的线性自由模型也无见证，89步实际超XML。加入末端速度条件没有消除这条失效链。模型近似、不断变化的参考与首步TCP保护之间的约束权衡仍需区分，不能只归为缺少末端制动。

当前首步TCP硬限额属于候选算法的开发保护，不是用户最终稳定任务目标。下一项可对照将首步误差保留在原整段状态跟踪代价中、独立报告限额超额的闭环方案，同时保持实际关节/动作约束和正式任务验收不变；不能提高阈值后声称旧合同通过，也不能以预测姿态重建代替实际任务成功。应先验证捕获点可行性，再做原生对照，不继续直接推广本版。

## 生命周期与可追溯结果

- v2 owner `scripts/execute_terminal_margin_feedback_v2.py` 统一会话82478实际exit0；两个子进程actual wait0/reaped，无自动重试。新完整分支256周期，首版失败86周期，总新增342；既有baseline未重跑。
- 独立 `scripts/verify_terminal_margin_feedback_v2.py` 实际exit0。检查新旧生命周期/预算、配对初态、参考切换、完整动作计划/变化率、原TCP限额及seed模型、阶段预算、末端模型日志/回退标记、原生goal、状态SHA及恢复TCP误差。它不独立重算每个未来模型计划，不将模型日志当作实际制动证明。
- 独立核验 `outputs/terminal_margin_feedback_v2_verified_20261027/verified.json`，SHA256 `7187299263f2619b3164d076aa0361b9d11dfa0c2960e7d8028f626b6c21809a`。
- 分段误差与terminal物理字段对照 `outputs/terminal_margin_feedback_v2_verified_20261027/segments.json`，SHA256 `4be2f4b2d78728079ce49954cdc757d2edb1b5007e87f0c9ce8441efb3c0f747`。
- 新增/修复模块、worker/owner/verifier和相关测试Ruff check/format check通过；未声明全仓检查通过。

本轮世界模型调用与fresh均0。原正确参考18/18、WAM开发和最终留出24条稳定成功门槛保持，尚未满足。
