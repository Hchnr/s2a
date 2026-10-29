# fresh 决策诊断 v2：历史 owner 来源闭包迁移

本登记在第二次真实复算之前建立，继承
[原诊断合同](state_vector_dual_recovery_fresh_diagnostic_20261029.md)全部研究、数学、
环境和原122响应上限。起点实现提交`c909b13`。本次不是自动重试：第一轮进程
已经实际退出1并回收，诊断输出冻结；补充加载实现另经QA/dry、提交后才执行。

## 首轮失败与边界

首轮owner `outputs/state_vector_dual_recovery_fresh_diagnostic_owner_20261029/result.json`
SHA256 `27613d5c82404fceb059cb9f6a0018a0a1475d309067f6bf826573d6fb86697e`；
checker `outputs/tracking_v5_state_vector_dual_recovery_fresh_diagnostic_20261029/verified.json`
SHA256 `c9d714e9e9b9a78cda7f6a229eecea41f3ea2a035728399e89f3b8c4049c1cd5`。
owner complete=True/passed=False，child实际wait/reap1；checker两字段均False。
耗时119.002684680745s，响应/查询/数值内步/QP/NNLS/物理/WAM/fresh均0，来源与
原收据SHA前后相同。失败在导入MuJoCo并构造数值模型之前，不能称已复算122条。

失败栈为native initializer→closed_owner→旧registration.source_sha256，首先读到
已移动的`configs/wam_clean_bridge_20261023.json`。首轮dry已核helper和顶层资产
证明，却没有执行该历史owner闭包，因此不代表完整初始化已通过。原fresh_ratio
失败仍未定位；本登记不把新路径失败归因为数值问题。

## 最小修复与预准入

新增context-aware的完整closed_owner身份适配，逐一保留原complete/passed、
launch CPU环境/no retry、exit实际wait0、release reap0/no signal/无原异常、
PID/start_ticks一致、完整result及全部source_sha256守卫。source中的历史源声明
通过catalog/Git核原字节，outputs artifact按原SHA严格当前读取；不改旧SHA、不
追认旧源码当前可执行、不安装wrapper/hook。1751来源的只读预核确认722项来源
均与baseline/rootcatalog SHA一致，其余1029项均为repo outputs，无未知源码或
新历史版本缺口。此预核没有运行数值模型或测试。

新dry必须在模型构造前实际核这1751项闭包及原helper完整模块证明。新增测试
须拒绝owner布尔/exit/retry/PID/start_ticks/source SHA等篡改；原八native完整
函数AST、原fresh_metrics、原严格1e-10比较/顺序/接纳/半径均保持。

新config为`configs/tracking/trajectory_search/state_vector_dual_recovery_fresh_diagnostic_v2_20261029.json`。
新checker独占`outputs/tracking_v5_state_vector_dual_recovery_fresh_diagnostic_v2_20261029/`；
owner独占`outputs/state_vector_dual_recovery_fresh_diagnostic_v2_owner_20261029/`，dry独占
`outputs/state_vector_dual_recovery_fresh_diagnostic_v2_dry_20261029/`。
旧config/plan、两次旧checker失败及其owner均保留原字节。新config显式绑定原producer
及全部先前失败引用；最终当前来源与QA收据在新owner中独立登记。

验证、dry和提交完成后只启动一次v2，1800s上限、单线程CPU、空CUDA/LD、无自动
重试。保持原solve工具输入；完整122响应含末repeat，预期36600内步/38065查询，
自身QP/NNLS/物理/WAM/fresh/参考消费0。失败保留实际计数和上下文，成功也仅为
保存路线数值核验；不改变完整H12 witness、控制或原18/12/24资格。GPU继续禁用。
