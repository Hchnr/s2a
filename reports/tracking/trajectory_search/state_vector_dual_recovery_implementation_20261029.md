# pose紧活动子集：实现与CPU准入

2026-10-29，实现提交 `d527982`。本阶段完成代码、合成回归和真实输入准入；
生产搜索与全部响应独立复算尚未执行，跟踪收益仍待验证。
[预登记](../plans/v0.0.1/state_vector_dual_recovery_20261029.md)保持冻结。

## 改动及依据

前项整个目标缩放的10个QP均status0，但其中7个未通过原互补证书。
保存乘子的零拟合算术诊断显示，最大互补项落在正slack的rate行；直接裁剪又
损坏stationarity。背景与历史失败保留在[前项报告](state_vector_objective_scale_20261029.md)。

- [求解器:41](../src/s2a/state_vector_dual_recovery.py#L41)保持原SLSQP目标缩放、
  原约束、初点、100迭代预算及成功要求；pose只对返回原空间点的
  `slack <= .01*certificate_tolerance`活动子集做一次原单位梯度NNLS。
- [完整证书:183](../src/s2a/state_vector_dual_recovery.py#L183)仍记录
  `slack <= 10.0*certificate_tolerance`的原完整集合，NNLS迭代上限仍按完整集合
  计算。集合外mu精确为0；原primal、stationarity、complementarity及非负门保持，
  tolerance仍为1e-6。restore使用原完整活动集合。
- [拟合计数:199](../src/s2a/state_vector_dual_recovery.py#L199)在调用前计费，
  异常仍记一次；空集合记零次。没有阈值搜索、第二次拟合或失败fallback。
- [独立审计:66](../scripts/state_vector_dual_recovery_audit.py#L66)从原矩阵及返回点
  重建两个集合，复用完整GN/KKT和时序账本，核验集合外零值、原预算及代数mu/C。
  该代数映射不声明为SLSQP回传乘子。
- [生产准入:63](../scripts/probe_tracking_v5_state_vector_dual_recovery.py#L63)绑定
  已关闭前项及诊断指纹，从原3557与72条signed探针重建初始模型；不加载旧final模型。
  [独立入口:117](../scripts/check_tracking_v5_state_vector_dual_recovery.py#L117)
  核验拟合总账，审计本身不执行拟合或QP。

数值配置与前项逐字节相同：seed42，60阶段/122完整响应；原安全、接受、反馈、
repeat、信赖域停止规则保持。旧失败结果未追认，原剩余预算未追加。

## 已执行验证

使用仓库内Python3.11.15，显式空CUDA/LD、单线程及独占pytest临时目录。

| 项目 | 结果 |
| --- | --- |
| 合并专项QA | 7模块，182 passed in 47.57s；182来源指纹前后相同 |
| 新增专项覆盖 | 核心15项、独立审计31项、入口准入20项，均已计入上述182项 |
| 新文件Ruff / format --check | 均exit0 |
| producer / checker真实输入dry | 均exit0；各2738来源，QP/响应/查询均0 |
| 实际进程关闭 | QA session49583、准入session98921均实际wait/reap exit0，无信号、无自动重试 |

核心和审计测试使用手工证书及求解/拟合替身，覆盖正slack冗余列、紧集合不足、
空集合、NNLS异常、原primal/solver失败、restore原字段兼容、单次计费和外层停止。
这些测试未产生新实验QP、NNLS拟合或模型响应。

可复核产物（生成产物保留在 `outputs/`）：

| 产物 | SHA256 |
| --- | --- |
| `outputs/state_vector_dual_recovery_qa_20261029/result.json` | `d4cfb716fbdaf82777d05df0df773ba6990c0b3c922557885293dfa25982eff7` |
| `outputs/state_vector_dual_recovery_admission_20261029/result.json` | `1d4c9becede35534b64822054a45b1bd65ecf2116dccd5e474f35a5035be8807` |
| `outputs/state_vector_dual_recovery_stage_driver_20261029.py` | `10301ed53d27803b12c6b716e0e5ad007392970f6030a1fee01151ad77e55fda` |

全仓样式命令也已执行，均exit1：`ruff check src tests scripts`仍有三个既有I001，
位于`check_gripper_release_floor_pair.py:50`、`check_tracking_correct_reference_v2.py:50`、
`validate_temporal_response.py:3`；`ruff format --check src tests scripts`仍要求格式化
`audit_rotation_goal_precision.py`。这四个`scripts/`文件相对提交前HEAD无差异，
本阶段保持历史源冻结；上述新文件检查通过不表示全仓样式检查通过。

## 当前证据及下一步

本项生产搜索、全部响应独立复算及实际前向控制均待执行，没有新增跟踪收益结论。
已准备的保存数据分析器仍未执行；后续按预登记完成唯一有预算生产与独立复算，
若找到完整数值witness，再单独验证实际前向控制。

实际资格仍cursor8/task0/grasp0，历史完整witness仍0；原18正确参考/12WAM开发/
24最终留出门槛保持，最终留出未打开。GPU继续禁用，总目标尚未完成。
