# H12 callback方向完整响应回缩

本轮按[执行前登记](../plans/v0.0.1/prefix_callback_backtracking_20261029.md)检验首个非零callback方向的固定回缩。保持actual368实测状态、completed367控制历史、chunk46原H12参考、原控制器和完整模型；GPU禁用。目的是找到原全部目标的完整模型解，再决定是否登记实际控制pilot。原18条正确参考、12条WAM开发和24条最终留出门槛不变。

## 实现和证据范围

[求解器](../src/s2a/prefix_callback_backtracking_plan.py)保留原响应、约束、目标、差分与排序数学。每阶段捕获首个非零callback，用固定alpha从1到1/256检验完整响应；仅第一个通过原审计、保住prefix且产生下降的ray点可成为普通新anchor。差分探针不能成为普通下降anchor，任何已完成响应中的完整witness仍优先保留。全局4000响应、60实际迭代、60阶段共享预算，不把回缩次数计成SLSQP迭代。

[独立账本审计](../scripts/prefix_callback_backtracking_audit.py)分别验证请求、缓存、差分、callback捕获、alpha、首个合格点、接受历史、预算和最终排序。[完整复算入口](../scripts/check_tracking_v5_prefix_callback_backtracking.py)另行重构每条唯一轨迹，并核验全部保存ray的完整rollout和审计、每个接受anchor及initial/selected/closest；账本审计本身不代替数值重构。

[来源准入](../src/s2a/prefix_callback_backtracking_evidence.py)要求旧baseline与candidate的4000/3148响应全部独立核验、实际wait/reap成功、无信号且来源未变。完整96资产快照保持不变，仅XML路径迁移；没有复用旧v2 main的AST证明来宣称新入口相同。[进程owner](../scripts/execute_tracking_v5_prefix_callback_backtracking.py)使用独占路径、1200/5400秒时限及同一native QA，拒绝覆盖和自动重试。

实现提交为`046c1eb`（求解器与独立账本）和`c901abc`（来源、生产、完整复算及owner入口）。所有旧已执行来源、测试、计划和失败记录保留。

## CPU测试

统一native QA结果为`outputs/prefix_callback_backtracking_native_qa_20261029_vv9yu8uw/result.json`，SHA`05e3ec58822c9a3a612ab30a3212451c4cceffc0e6edf3cdecf317d1dd6a3add`。252项通过，27.59秒，实际wait退出0，28份来源前后指纹相同；日志SHA`b15de6e069c15aaa119c24f6fad722128ec7f6f9a3b9f97d4f0fcf4c27af2f27`。Python3.11.15、NumPy1.26.4、SciPy1.15.3，空CUDA/LD及单线程CPU。两条SciPy bounds-clipping warning仍来自保留的旧困难例。

覆盖旧156项和新增25项solver、29项独立账本、42项执行集成测试。新增测试包括真实SLSQP困难H2、弯曲前缀的完整响应回缩、精确缓存、末次预算响应中的witness、仅ray普通接纳、完整ray数据变造拒绝、两旧复算闭合门槛和实际状态/计费接线。困难H2真实合成例在预登记参数下找到完整witness，观察到102响应/7迭代/7阶段/17个ray尝试；严格内点例为14响应/1迭代。测试断言实际严格可行和预算，没有硬编码后端迭代数，也未改变生产参数追求测试通过。这些结果不是原H12或实际任务成功。

## 正式执行状态

两旧独立复算均已实际wait/reap0关闭，来源及严格标签一致，确认完整witness0，见[完整对照结果](prefix_trust_region_control_20261028.md)。新CLI准入也已关闭：`outputs/prefix_callback_backtracking_admission_20261029/result.json` SHA`a5f80084ec8bf65e31d3af79b251a42d7822f3ea6f992908af7f300eeaf62b76`。25个Python文件Ruff/format、新producer/checker dry及owner diagnose dry共5条命令实际wait0，无信号或重试，28份来源未变；2241个输入/来源引用通过，owner加QA/日志后2243。共同warm SHA仍为`16bdfbb266ef13d2588c8116e13511d93df936d72ae8a57d19df0b01ed3acee0`，新增响应/查询/内步均0。

本段准入记录生成时新候选尚未启动。后续只能使用已登记的独占输出和同一QA，不因观察超时重启；没有完整独立结果前不授予新数值资格。

实际机器人仿真进度仍为原cursor8、task/grasp0；本次新增物理、世界模型调用、fresh案例均为0。即使后续完整模型witness通过独立复核，也不自动授予实际控制、WAM或留出任务资格，总目标尚未完成。
