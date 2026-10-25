# 054：同步当前端点上的新头配对采样准备

总目标与最终原24条门槛不变，fresh关闭。053实际同步数据/native稳定正控6/6已通过；052新头只取得数学资格与teacher拟合改善，尚无真实新采样或闭环收益。

[054计划](../plans/v0.0.1/054_synchronous_state_probe052_20261024.md)在任何新模型调用前固定99请求：原037三个zero完全回归，加053的24当前端点×seed42/43×raw/probe052。保持checkpoint8400/nonEMA/active10 mask/H40/20Hz/35步/g1/shift5/SDPA、1次初始化、0fit/物理/fresh，不扫描或重试。[不可执行policy](../configs/wam_synchronous_state_probe052_policy_20261024.json)列出尚缺的执行前提，未授权GPU。

## 已实现和实际验证

`src/s2a/synchronous_state_sampling.py`实现current-only白名单、完整配对矩阵、052自身schema/参数身份及独立QR资格核对。拒绝旧W2身份、原ridge、部分行、放宽容差、缺comparison、参数或target流不一致。请求与当前图像/任务/seed精确绑定，禁止future state、expert command和success。

`sample_wam_synchronous_state.py`以独立054身份复用原sampler主体，候选名称probe052，不伪造旧W2 owner。独立metadata检查对完整run函数作AST比较，只允许列举的admission/identity命名及required-source扩展，调度、mask、capture、zero和数值路径未改变。修改denoise计数35→34的攻击测试被拒绝。

23项CPU测试通过，Ruff及format通过。实际模型Python3.13成功解析--help；对明确未批准的登记执行--dry实际被拒，收据`outputs/synchronous_state_preparation_20261024/qa/actual_model_cli_rejection.json`。这是负向准入检查，不是成功dry或模型运行。

## 实际元数据结果

根目录`outputs/synchronous_state_probe052_20261024/`。原prepared.json生成时测试文件尚有一处import排序告警，0执行/模型；其文件及原快照保留。修正后prepared_v2只更新该测试源码/快照并记录原ref，所有数据/参数和采样数学不变，完整23项重跑通过。

| 产物 | SHA256 |
| --- | --- |
| prepared.json（历史未执行） | 796f5082a23c29463298d134800a13186ee2ebb598534e32dc0d973d8d513130 |
| prepared_v2.json | 06a0a61c3e9d2d42e11252dfeb266164945b83c78065d205b35ac9afb1454d84 |
| metadata_verified.json | 1902cd0b7b98b65129f71ce62caf28b299e5599cfac24ac931f3526196cfaaba |

独立metadata审查真实exit0：1435源/1435快照、完整48对当前输入、105历史模型节点可用性、实际31-bit训练seed与eval42/43隔离、当前manifest与053独立物理审查结果一致。输入allowlist只含JSON与当前PNG，不包含validation trajectory/HDF/未来标签。历史037模型archive解码专用于zero源hash可用性，不是validation未来读取。

## 未完成项

prepared及metadata仍execution_authorized=False，尚无执行registration或模型进程。必须先实现新身份下的完整3465节点独立数值审查和可核验正常wait/reap/释放收据，冻结最终源闭包，跑真正准入成功的model CLI dry，并核实允许GPU UUID/PCI/minor及空闲；之后才能启动。完整数值/生命周期通过后，CPU评分才读取053未来端点，按已登记的h8/h16/h40和双seed门槛评分；不得将prepared/软件通过作为任务收益。
