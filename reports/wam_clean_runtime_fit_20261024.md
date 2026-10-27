# 045：合格运行时上的W1实际拟合与独立QR

本项推进[036总目标](../plans/v0.0.1/036_project_goal_20261023.md)，最终验收仍为同一候选在原预留24条中稳定≥20/24、每任务≥4/8。本报告不以CPU拟合代替状态跟踪或任务成功。协议见[045](../plans/v0.0.1/045_clean_training_runtime_fit_20261024.md)。时间按北京时间。

2026-10-24 09:49：**W1实际生产和完整独立QR均已通过**。生产处理420000行、150标签窗口，1次新fit，耗时162.10秒；独立重建全部420000行、150窗口并进行augmented QR，耗时222.77秒。三类参数按原`rtol=1e-9/atol=1e-11`比较通过，完整行指纹一致。两角色均实际wait0/reap/no-signal、同birth已消失，supervisor正常完成，508源/508快照全部保持SHA。原043失败保留，累计W1尝试2次、成功1次；0新模型/物理/fresh，D1未启动。

随后在CPU上调用原041 bundle collector的真实`admit_execution`和`admit_round(D1)`，新参数可被原D1 head合同接受，1290特征/ridge=.1及原D0来源一致；未导入torch/MuJoCo、未解码数组、未创建D1目录。该读取检查不等于046完整owner dry或新模型准入。

## 输入、运行时与失败保留

沿用041完整新D0的300请求/10500节点，以及[044](wam_clean_runtime_qualification_20261024.md)完整150标签窗口和两路径420000行字节一致的资格；不重采D0。实际worker为Python3.11.15、NumPy1.26.4、SciPy1.15.3，空CUDA/LD、三线程限制8；另固定解释器实际二进制和NumPy/SciPy两个数值扩展的SHA，每次worker准入检查实际加载文件。

原043已有1次实际W1尝试，首标签检查失败，0成功fit/参数/独立QR。原失败路径不改。本项另用原prepared.root内的`runtime_fit_20261024/fit/W1`和`runtime_fit_20261024/review/W1_verified.json`，最多新尝试1次和独立重算1次；累计W1尝试上限2，不自动重试，D1/W2尚未授权。

原生产入口、完整行构造、归一化、ridge=.1、1290特征及原独立augmented QR不改。代码依据：[原生产入口:224](../scripts/fit_wam_clean_training.py#L224)、[原streaming:166](../src/s2a/wam_clean_streaming.py#L166)、[独立QR:120](../src/s2a/streaming_ridge_qr.py#L120)、[独立实际参数比较:450](../scripts/verify_wam_clean_training_fit.py#L450)。新worker/owner只有限替换准入、固定路径、worker解释器与额外来源字段；AST与生成模板一致，原498源/快照逐SHA保留。

## 合成测试与准入范围

实际sim运行时原三模块测试为**60通过、4失败**，原日志保留。4项均为同一nonuniform合成样例的4种分块，在`rtol=2e-12/atol=2e-14`下与NumPy `lstsq`参考解不一致。对FP64输入定义的相同ridge目标作80位Decimal独立消元：原stream/direct最大绝对误差≤8.89e-16、原QR≤1.12e-15，NumPy `lstsq`误差3.94e-14；SciPy gelsy误差8.89e-16，另外两种driver分别记录。该结果只解释这一合成参考差异，原4失败不改记通过，不修改冻结测试或实际拟合的`rtol=1e-9/atol=1e-11`。真实420000行仍须通过完整原独立重算。

第一份未授权提案因绑定脚本的导入格式检查失败而拒绝，保留提案、快照及拒绝说明；没有启动worker或消耗训练尝试。修正版固定508源/508快照，8份新入口Ruff/format、AST转换检查通过；原498文件及快照不变。独立metadata CLI核原043失败、044完整结果/正常wait0/reap、D0来源、预算、输出和运行时后，生成分离的verification及admitted。

实际admitted下producer、independent与owner三份dry-run均exit0；两个worker均通过原完整`load_context`，expected_rows420000，source-proof回调各1次并恢复，0fit/QR。owner路由两角色均使用实际sim解释器。随后仅启动一个持久化CPU supervisor，生产成功后才串行启动独立QR，每角色沿用原7200秒上限。

## 证据索引

所有下列路径相对于`outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/runtime_fit_20261024/`：

| 证据 | 路径 / SHA256 |
| --- | --- |
| 原合成测试日志 | `qa/math_pytest.log`，保留4失败/60通过 |
| Decimal独立诊断 | `qa/math_diagnosis_v2.json`，4种分块 |
| 路由与原498源核查 | `qa/routing_v2.json` |
| 未准入提案拒绝 | `qa/proposal_v1_rejected.json`、`execution/proposal.json`及原`source_snapshot/` |
| 修正版提案 | `execution/proposal_v2.json` / `f3fd28a366a56ff1bcae7a5b669b1b50ba36b12d8dc5f9c04dd50fd215c92e4a` |
| 独立metadata核查 | `execution/verified.json` / `1c9ee59f9715b900a983b4eb44006470e696ba83420d0a52dd26d29fbd91e449` |
| 实际准入 | `execution/admitted.json` / `6eb0d233c05b3859f1bd50ddeaa6679089df0f7e186b9cf85a463030b238436a` |
| 实际三角色dry | `qa/admitted_cli_dry.json` / `28b13c0b17fa7d5d4bb407603e47dc8cf40d5f057edcbecd65a369d64cea6967` |
| 实际W1参数 | `fit/W1/parameters.json` / `e70ab79facfbb42667700d823c7ca54fbafb03de03aa198767afeb3c5917dea8` |
| 全量独立QR | `review/W1_verified.json` / `a0fc11599bd325246ea4e2b995054f1ed6c0b21af41138a22a1273ff07c4d56f` |
| 完整supervisor | `supervisor/result.json` / `6e97469af30a8ff662606c911bf44efa831850ae4cc39500099465759120cf5a` |
| 两角色正常收尾及508源复核 | `qa/terminal_audit.json` / `480154904dac3a9133c095ae7e093e107ba942039bebe0edbe0b89330a556dc2` |
| 原D1合同只读接入 | `qa/d1_original_contract_readonly.json` / `de51fca145a0c76bbb27e8a09009c4d712a138cff3bd2b2f9825af12bc317f50` |
| 正常短launcher | `supervisor/launcher_result.json` / `de1d5f1edc838375519816b272417feb148a5a5d212b938f906a2e5de942ed04` |

实际配置：[045 v2配置](../configs/wam_clean_training_runtime_fit_20261024_v2.json)。本项通过只证明一个固定W1及其独立数值重算，不证明适配改善、开发任务成功或留出泛化。下一步按[046接续计划](../plans/v0.0.1/046_clean_training_d1_continuation_20261024.md)实现并验证新W1 owner/参数绑定以接续D1，不改原失败owner，不直接打开最终预留集合。
