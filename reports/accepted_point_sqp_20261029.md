# 实际接纳点曲率保留的显式 SQP：实现与组件验证

本候选搜索已正常关闭（tool session33552实际exit0）：1919响应、27个合格QP、24次普通接纳与24次曲率更新后，三档半径回缩耗尽，仍为prefix9、完整H12 witness0。独立完整复算owner已启动（session6487），尚无其终态结果。本稿的搜索数字来自已关闭producer及保存数据分析，不冒充独立数值或实际控制资格。

按[执行前登记](../plans/v0.0.1/accepted_point_sqp_20261029.md)，保持 actual368 当前状态、completed367 因果控制历史、已生成 chunk46 的原 H12、共同 warm、完整资产模型和历史误差 profile；不使用 actual369 以后的真实状态作输入。原18正确参考、12 WAM开发、24最终留出门槛保持。前置[固定优化余量完整复算](prefix_optimization_reserve_20261029.md)已关闭，仍为 prefix9、完整 H12 witness0。

新机制在实际接纳点重新建立显式 QP，保存命令块曲率，而不是每次取得首个 callback 就丢弃整个内部优化状态。它同时改变方向生成与曲率生命周期，未来差异不能单独归因于 BFGS。

QP变量为 `[d,t]`，`t` 是绝对非负 epigraph；目标为 `t + 1e-4*grad(sum(q))·d + 0.5*dᵀBd`。全部安全、前缀、目标、rate、box和信赖域约束写成 `A[d,t]+b>=0`。已达前缀仅在优化模型中保留 `1e-4` 余量；实际 pose caps 仍为10mm和0.05rad。普通回缩点仍须通过原完整 joint/command 审计、原连续前缀、`1e-8` 非线性安全余量及原下降条件；完整 witness 要求全部 H12 原 caps 和审计通过，不额外要求优化余量。依据：[QP构造:291](../src/s2a/accepted_point_sqp_plan.py#L291)、[实际接纳:425](../src/s2a/accepted_point_sqp_plan.py#L425)。

QP使用公开 SciPy SLSQP，解析目标梯度和约束 Jacobian，`ftol=1e-14`、最多100个内部代数迭代。固定行缩放后，以 NNLS 拟合全部 active 行的非负乘子，独立检查 scaled primal、stationarity 和 complementarity，证书阈值 `1e-6`，active 阈值为其10倍。后端 success 与证书必须同时成立；乘子是近似局部 QP 乘子，不是精确物理乘子。依据：[公开QP:48](../src/s2a/accepted_point_sqp_core.py#L48)、[证书:156](../src/s2a/accepted_point_sqp_core.py#L156)。

同 cursor 的下一实际接纳点完成整套 FD 后，使用前一合格 QP 的同一组原单位非线性乘子计算两端 Lagrangian 梯度差；`s` 来自实际接纳 ray 的命令差，而非未接纳的完整 QP 步长。固定 Powell 阻尼系数0.2；实现采用对称 outer 公式，随后做严格对称性及 Cholesky 检查，未另做矩阵平均或谱修补。无效更新保留旧 B；不以 FD 间距作为最小接纳位移阈值。依据：[同乘子实际secant:311](../src/s2a/accepted_point_sqp_plan.py#L311)、[BFGS:219](../src/s2a/accepted_point_sqp_core.py#L219)。

拒绝的 ray 不产生新 secant，部分 FD 不更新 B；待更新 pair 只能消费一次。半径缩小保留 B，cursor 增长重置为单位阵并清空 pending pair。合格非零 QP 方向按固定九个 alpha（1至1/256）实际回缩，普通接纳只取第一条合格 ray；完整缓存 witness 优先。依据：[回缩:391](../src/s2a/accepted_point_sqp_plan.py#L391)、[状态转换:461](../src/s2a/accepted_point_sqp_plan.py#L461)。

H12每套完整差分为72轴，固定单边 nominal step 为 `1e-4`，符号由本阶段上界决定。差分、回缩和 warm 共用精确命令字节缓存；每个新的完整 H12 响应计入原4000预算，缓存命中不重复计费。最多60次非线性外层尝试、60阶段；QP内部代数迭代单独计数，最多6000次，不混称为非线性迭代或模型响应。半径固定为 `.01/.005/.0025`；预算中断保留已有完整响应及 witness，不追加预算。依据：[固定参数:14](../src/s2a/accepted_point_sqp_plan.py#L14)、[差分:270](../src/s2a/accepted_point_sqp_plan.py#L270)、[终止:508](../src/s2a/accepted_point_sqp_plan.py#L508)。

独立 checker 将逐条重构全部唯一响应；每条 compact trace 的 TCP error 和完整安全 slack 向量另按原数值容差核对，严格 prefix/joint/witness 标签不放宽。每份实际 ray 与 anchor 的完整审计也必须核验。算法代数则用已核对的保存原精度 trace 重建 FD、QP、乘子证书与 B 生命周期，避免将不同浮点重放路线的误差放大后冒充原算法 Jacobian。账本与代数审计不导入生产求解器，也不重新求解 QP。依据：[逐响应核验:264](../scripts/check_tracking_v5_accepted_point_sqp.py#L264)、[账本:59](../scripts/accepted_point_sqp_ledger_audit.py#L59)、[独立QP证书:88](../scripts/accepted_point_sqp_math_audit.py#L88)。

## 已关闭的正式 CPU QA

直接读取 [QA receipt](../outputs/accepted_point_sqp_native_qa_20261029_ubofewed/result.json) 与其绑定的 [pytest日志](../outputs/accepted_point_sqp_native_qa_20261029_ubofewed/pytest.log)，确认：

| 项目 | 实际记录 |
| --- | --- |
| 测试 | 425 passed，2 warnings，49.35秒 |
| 来源 / 测试模块 | 52 / 17 |
| 完成 / 通过 / 实际wait / 退出码 | true / true / true / 0 |
| QA来源前后不变 | `source_unchanged=true` |
| Runtime | Python3.11.15、NumPy1.26.4、SciPy1.15.3 |
| CPU环境 | CUDA/LD为空，禁bytecode，OPENBLAS/OMP/MKL/NUMEXPR/LP均为1 |
| receipt SHA256 | `15fcabb5e5a3f09df41cf0fc7dfce469c011bf54fdf6ff944bde874ee6466143` |
| 日志 SHA256 | `e4c2a84f97f1d321c17d4c86161940aed8e41da1f40da2a8a765a42656a446e9` |

组件证据覆盖解析 QP/dual、缩放、失败证书、Powell 阻尼与 SPD、真实合成响应的缩短 ray 与同乘子 secant、cursor reset、部分 FD 预算及全缓存 witness；还验证差分换绑、错误位移/梯度差、伪证书、缓存计费、错误 anchor 和放宽 caps 均被拒绝。执行接线测试另核共同初态、来源/QA绑定及 checker 在独立数值检查前不转交未验证数据。依据：[外层测试:49](../tests/test_accepted_point_sqp_plan.py#L49)、[账本反例:101](../tests/test_accepted_point_sqp_ledger_audit.py#L101)、[执行接线:413](../tests/test_accepted_point_sqp_execution.py#L413)。

## 准入与搜索状态

[准入记录](../outputs/accepted_point_sqp_admission_20261029/result.json) SHA为`ee6440624e8fb932d2f07c3ae2e9b5a7e748dbd088a44a8f5e3ece8462833869`。Ruff、47个Python文件format检查、producer dry、checker dry、diagnose owner dry五个命令均实际wait0，无信号、无重试，52份来源前后不变。producer/checker dry各核2299引用，owner dry核2301引用；新响应/查询均0，warm SHA保持`16bdfbb266ef13d2588c8116e13511d93df936d72ae8a57d19df0b01ed3acee0`。

## 已关闭的搜索与保存数据分析

[producer](../outputs/tracking_v5_accepted_point_sqp_20261029/verified.json) SHA为`66b5b2f89905b7e8be17d92926ffe96015d2433c5f9ce9dc02860870b3d739da`；[owner](../outputs/tracking_v5_accepted_point_sqp_execution_20261029/result.json) SHA为`07301c0ac1fc413dba8f677a5bfbd123af707ed73589dd5e9a932fd807203da6`。原子进程4181009/start615941428正常wait0/reap，无信号或自动重试。未耗尽4000响应上限，也未追加预算；实际在三档信赖半径均无接纳点后停止。

[保存数据分析](../outputs/accepted_point_sqp_result_analysis_20261029/result.json) SHA为`43c72b4136faa55a60142ac264f6a0e7790b918b878eb0d21679b739d7f5c22d`，driver实际exit0，2311引用重读不变。它只分析保存值，没有执行新响应；完整独立数值核验仍另行进行。

| 项目 | 实际保存结果 |
| --- | --- |
| 完整唯一响应 / 数值内步 / 查询 | 1919 / 575700 / 598729 |
| 响应来源 | warm1、FD1800、回缩118；FD另有144次缓存命中 |
| 外层阶段 / QP代数迭代 | 27 / 466；27个QP后端成功且证书合格 |
| 回缩 / 普通接纳 / 含warm的anchor | 118 / 24 / 25 |
| 曲率更新 / 其中Powell阻尼 | 24 / 8；无cursor增长重置 |
| 终止 | 最后三阶段半径.01/.005/.0025，各九条ray全部拒绝 |
| prefix / 完整witness | 9→9 / 0 |
| 原cursor9最大归一化平方误差 | 1.3249481018012241→1.3167648082536445 |
| selected cursor9位置 / 旋转误差 | 0.011474935525262358m / 0.057375186454024805rad |
| selected身份 | 全缓存排名第1918响应；不是最后接纳anchor1819 |

实际s与同乘子曲率机制确实执行，但没有产生完整解。全部回缩中的拒绝原因计数为joint/command81、安全buffer83、prefix丢失20、无目标下降11，原因可重叠；不能把本次终止归因为QP证书失败。B条件数最高约`1.465e10`，已用s的无穷范数与FD步长之比约0.390625至100；这些诊断本身不证明病态曲率或差分误差是唯一原因。

## 独立复算状态

[check-owner准入](../outputs/accepted_point_sqp_check_admission_20261029/result.json) SHA为`e31c9652b1362215144659740874cb8c4733aa96969d59f9612a0d421bcfa410`，实际wait0/reap，核2307引用、新响应0。新check owner通过同一QA后已启动，目标是完整1919响应、118份ray与25份anchor及27个QP/24次曲率更新的独立核验；不得以部分复算替代全量关闭。

[运行中快照](../outputs/accepted_point_sqp_execution_checkpoint_20261029/checker_started.json) SHA为`f9aba7bd12ce274d89104c82c834310124a8fbac167c4dfb3d66237f9990334f`，记录原session6487、child24766/start616006536、owner21673/start616001104及100/1919进度；两进程均实际存活，空CUDA/LD及单线程环境核对通过，52份QA源码重读不变。这只是运行中证据，不能代替最终wait/reap或完整复算结果。

## 保存数据的终端行诊断

[只读诊断](../outputs/accepted_point_sqp_terminal_linearization_20261029/result.json) SHA为`951eb83b022938d1d908170c8bd8da32bca7a7687ad7b60cc6219e16963db3b9`，[执行记录](../outputs/accepted_point_sqp_terminal_linearization_20261029/execution.json) SHA为`2913a0e9ec50dec3681e73326a88dc3d3bef32af225962eba7ca4b4b3781fd91`。单次实际wait/reap0，2314引用前后不变；新响应、查询、内步、物理、模型及fresh均0。它从保存的完整安全向量、逐轴FD关联和实际ray位移重建线性预测，证据等级仍是保存数据代数。

末三阶段共用anchor1819及同一组72轴FD。该anchor的`terminal[0]`余量为`1.320428817e-8`；27条ray均保持原prefix9并降低cursor9目标，但最小安全行全部为`terminal[0]`。26条违反原joint审计，最后一条仅低于`1e-8`优化buffer。因此全缓存selected1918与最后接纳anchor不同是既定排名与接纳规则的结果，并非已取得推进资格。

| stage / 半径 | alpha=1实际terminal余量 | alpha=1/256线性预测 | alpha=1/256实际值 |
| --- | --- | --- | --- |
| 24 / .01 | -6.47022e-6 | 1.31918e-8 | -2.80245e-8 |
| 25 / .005 | -3.65684e-6 | 1.31918e-8 | -4.85556e-9 |
| 26 / .0025 | -1.69002e-6 | 1.31918e-8 | 8.65477e-10 |

设R为实际余量减线性预测。stage24在alpha=1/16至1/128的R/alpha约为`-8.2e-6`且近稳定，R/alpha²绝对值却约增8倍；stage25/26的R/alpha也未趋近零。这支持有限采样区间内的一阶预测不一致，不能据此宣称已证明渐近误差阶数或纯二阶效应。FD截断、控制裁剪等不光滑性、浮点噪声尚未区分；末三阶段B条件数实际均约`6.36e5`，此前峰值也不足以证明BFGS是唯一原因。

下一项拟验证问题是固定anchor与这三个已保存方向的方向导数一致性：独立登记预定步长的双侧/单侧差分，核对原命令合法性并计入每个新完整响应；双侧不合法则明确记录，不能裁剪后冒称原方向。该诊断尚未登记或执行，须先关闭当前完整复算。不会因当前失败放宽原caps/buffer、追加搜索预算或自动采用二阶修正；若不同步长差分不稳定，“减小FD步长即可解决”也应被否定。

实际任务仍停在原 cursor8、task/grasp0，本候选新增 physics/world-model/fresh 均0；GPU继续禁用。只有完整 H12 witness 与全部独立核验通过，才可另行登记物理 pilot；组件成功与局部 QP 证书均不升级控制资格。
