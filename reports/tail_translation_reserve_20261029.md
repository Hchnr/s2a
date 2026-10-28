# 末周期平移恢复终端余量

固定14次生产响应、隔离 v2 的全部14条独立复算、2485来源闭合及全12候选分析均已正常完成。6/12候选符合原恢复资格，选择请求5（末周期 y `+0.001`）；终端经验余量达到预登记目标，`improvement_found=True`、`sufficient_reserve_restored=True` 均经独立核验。前11周期、全部旋转 goal、历史 anchor 和首末 repeat 的所需 exact 检查均通过，strict 分类差异为空。

完整 H12 witness 仍为0。所有候选的 cursor9 误差保持 `0.01147493763021341m / 0.05737519919843557rad`，仍超原 `.01m / .05rad` caps；恢复不构成参考推进或任务成功。实际仍 cursor8、task/grasp0，GPU禁用，原18正确参考/12WAM开发/24最终留出门槛与总目标不变。

## 固定合同与选择结果

按[执行前计划](../plans/v0.0.1/tail_translation_reserve_20261029.md)，仅改变第12周期一个平移轴，前11周期和全部旋转命令保持。12候选加首末 anchor 共14完整调用、13唯一命令；全部命令在模型构造前独立重建并准入。每调用重新复制 actual368 q/v、固定 finger 和 completed367 latch，不使用后续反馈。原控制器、模型、pose caps、关节/制动/terminal 约束保持，无 QP 或参数拟合。

恢复资格要求原九组审计、全部非线性原始 slack `>=1e-8`、原 prefix9（cursor0..8）及历史/重复/因果 exact；按终端值降序、实际扰动 L∞、请求序号严格排序。规则见 [tail_translation_reserve.py:186](../src/s2a/tail_translation_reserve.py#L186)。实际合格排序为 `[5,10,7,12,2,4]`，没有按新结果选择步长或放宽门槛。

原 anchor 的 `terminal[0]` 为 `1.3204288171648844e-8`；请求5提高到 `7.734855175988188e-5`，扣除原 `1e-8` buffer 后为 `7.733855175988188e-5`。固定前项全15原模型残差给出 `E17=2.3626357467448553e-6`，原式 `terminal_selected - 1e-8 >= 10*E17` 通过；展示用 slack 门槛为 `2.3636357467448554e-5`。

所选候选其他安全项最小值仍为 `3.267282409069594e-7`。因此这里只确认终端经验余量恢复，不能称为全约束鲁棒内点；`10*E17` 也不是72维误差上界或下一次 QP 的可接纳保证。

## 全部12候选

以下数值仅作展示舍入；选择和资格使用归档原精度。所有12点均 prefix9、完整 witness=False、clipping flags 相对新 anchor 改变数0；不合格六点均因 terminal 违反原审计和额外 buffer。完整误差/因果/选择字段见[已闭合分析 JSON](../outputs/tail_translation_reserve_v2_analysis_20261029/result.json)及[逐点分析表](../outputs/tail_translation_reserve_v2_analysis_20261029/analysis.md)。

| 请求 | 轴 | 命令扰动 | terminal[0] | 全部非线性 slack 最小值 | 恢复合格 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | x | +0.001 | -2.82190920404e-5 | -2.82190920404e-5 | 否 |
| 2 | x | -0.001 | 2.82467227857e-5 | 3.26728240907e-7 | 是 |
| 3 | x | +0.0005 | -1.41030966613e-5 | -1.41030966613e-5 | 否 |
| 4 | x | -0.0005 | 1.41298107660e-5 | 3.26728240907e-7 | 是 |
| **5** | **y** | **+0.001** | **7.73485517599e-5** | **3.26728240907e-7** | **是，选中** |
| 6 | y | -0.001 | -7.73195325440e-5 | -7.73195325440e-5 | 否 |
| 7 | y | +0.0005 | 3.86805516905e-5 | 3.26728240907e-7 | 是 |
| 8 | y | -0.0005 | -3.86534904591e-5 | -3.86534904591e-5 | 否 |
| 9 | z | +0.001 | -6.37099581201e-5 | -6.37099581201e-5 | 否 |
| 10 | z | -0.001 | 6.37372156261e-5 | 3.26728240907e-7 | 是 |
| 11 | z | +0.0005 | -3.18484830332e-5 | -3.18484830332e-5 | 否 |
| 12 | z | -0.0005 | 3.18751038430e-5 | 3.26728240907e-7 | 是 |

## 独立数值与因果核验

v2 从同一因果初态重新执行全部14条，包括计费的末 anchor repeat；完整数值记录、原安全向量和 strict 标签均核验。随后独立检查保存精度账本，以及复算路线内部14条的完整前11周期、前275内步、前11端点和全部12个旋转 goal；见 [check_tracking_v5_tail_translation_reserve_v2.py:347](../scripts/check_tracking_v5_tail_translation_reserve_v2.py#L347) 与 [复算恢复门检查:74](../scripts/check_tracking_v5_tail_translation_reserve_v2.py#L74)。保存和独立路线各自的因果/repeat exact 均通过；跨实现浮点数值沿原容差比较，二者没有混同。

| producer 与独立复算的字段 | 最大绝对差 |
| --- | ---: |
| 请求/实际力矩 | 1.2505552149377763e-12 |
| 内步关节位置 | 1.3322676295501878e-15 |
| 内步关节速度 | 5.10702591327572e-15 |
| 端点 pose 矩阵 | 1.1102230246251565e-15 |
| goal 矩阵 | 6.661338147750939e-16 |
| 完整 safety_slack | 2.4498191832349886e-15 |
| TCP 误差分量 | 1.3600232051658168e-15 |
| 归一化平方误差 | 2.9265478929119126e-13 |

最大归档响应比较差约 `1.25e-12`，来自力矩；strict 分类分歧0，实际 native quaternion JIT signature 非空。另行记录的 JIT 与 Python quaternion 对照最大差为 `2.9802322387695312e-8`、非零记录98项：这是两条目标计算语义的辅助对照，不属于上表生产与独立响应之差，不能隐去或称逐 bit 相同。

## QA、失败保留与实际成本

原95源/29模块正式 native CPU QA 为 **766 passed**、122.69秒；新100源/30模块为 **796 passed**、120.35秒。两次均有既有 SLSQP 合成用例的2个警告，实际 wait0/reap、无信号/重试；新100源包含完全相同 SHA 的旧95源，原95源还保留旧85源子集。原5命令准入、原 check owner 单命令准入、新4命令准入均实际正常关闭，未产生模型响应。

v1 checker 句柄58216实际 exit1：它完成14条/4200内步/4369查询及原数值、strict 分类检查后，在 historical full entry 的 `clipped` 布尔数组比较发生 TypeError。原 audit 默认比较器转 float，但实际 checker 注入的比较器直接相减；组件 QA 未覆盖该接线。该次 ledger 和 fresh 恢复核验未完成，原失败不能追认为通过。child569716/start616909863 已 actual wait1/reap，无信号、无自动重试，原95源和失败产物保持。

[隔离 check-only v2](../plans/v0.0.1/tail_translation_check_v2_20261029.md)仅修复布尔/整数标签的 exact 类型、shape、值比较，浮点仍用原比较器和容差；不重跑 producer，不改候选、选择、E17 或控制参数。v2 句柄41688实际 exit0，全部14条再次完整复算并完成先前未完成的审计。修复提交 `5496c42`、登记/失败记录提交 `3ee6142`，原实现提交 `0503033` 保留。

| 实际阶段 | 完整响应调用 | 数值内步 | 查询 | 退出与资格 |
| --- | ---: | ---: | ---: | --- |
| 原 producer | 14 | 4200 | 4369 | exit0，保存全部12候选 |
| 原 v1 checker | 14 | 4200 | 4369 | exit1，元数据核验失败保留 |
| 隔离 v2 checker | 14 | 4200 | 4369 | exit0，完整独立核验通过 |
| **实际合计** | **42** | **12600** | **13107** | 同13唯一命令的三次完整执行 |

来源闭合句柄11624、保存分析句柄92812均实际 exit0。闭合重读2485来源不变；这两个只读步骤新增响应/查询0。所有阶段新增物理、world-model、fresh、参考消费及优化迭代均0，`control_qualified=False`。

## 已闭合证据

| 证据 | 相对路径 | SHA256 |
| --- | --- | --- |
| producer | `outputs/tracking_v5_tail_translation_reserve_20261029/verified.json` | `0305dc39480abf4512205f8628729fd789595d40aeca189343bc19779f029d70` |
| producer owner | `outputs/tracking_v5_tail_translation_reserve_execution_20261029/result.json` | `f604b9e31b1a4aa00a576b099065334fd04aae8722ac5710165b080c2c76b6cb` |
| v1失败 checker | `outputs/tracking_v5_tail_translation_reserve_check_20261029/verified.json` | `f7582560bc9dce92f56632b8c7688c9b4403b50ce23be4dc1d68b818d3600751` |
| v1失败 owner | `outputs/tracking_v5_tail_translation_reserve_check_execution_20261029/result.json` | `22a0be80635c2023c2cc95b2a088eb0c9b51aa5e2c362c7b4a52c97eae88da8f` |
| v1 traceback | `outputs/tracking_v5_tail_translation_reserve_check_execution_20261029/capture/command.log` | `876e35928e2245bf60f093cf22cd2e1abd24af0e23506bafb1ad234b6d3df7a9` |
| v2 checker | `outputs/tracking_v5_tail_translation_reserve_check_v2_20261029/verified.json` | `142b0d9ad478b30c1e707c054cec48bd29326b0b24b4e07e64b6ea8cb37e6bde` |
| v2 owner | `outputs/tracking_v5_tail_translation_reserve_check_v2_execution_20261029/result.json` | `3fa4fb454a1d03584d2216982acec4eae1f9d06e635e38f4b8c23c6ba3df6843` |
| 原95源 QA | `outputs/tail_translation_reserve_native_qa_20261029_lkpp55kq/result.json` | `e421b3860faa93407194fad3cc57eabd56d16d6eea58bb46998a90dc62dde663` |
| 新100源 QA | `outputs/tail_translation_check_v2_native_qa_20261029_y5rxr7gl/result.json` | `fe36dfd12c29f6a55fd73d789cc91132e3fb726069100cf3a729743e421590d8` |
| 原5命令准入 | `outputs/tail_translation_reserve_admission_20261029/result.json` | `a02ef18417130cdd2aa74a299e6e9245def56c3627a388495a066626ce3c8e44` |
| 原 check owner准入 | `outputs/tail_translation_reserve_check_admission_20261029/result.json` | `40db64dbca0b784f2b890a91fa0e7b5e3211f4048fe4dae13494d938ec555908` |
| 新4命令准入 | `outputs/tail_translation_check_v2_admission_20261029/result.json` | `159cae7a80c7907e934869a388aab26a7269a02b44ebb105940f54be9dc6729f` |
| 完整来源闭合 | `outputs/tail_translation_reserve_v2_final_closure_20261029.json` | `52783b3ef40e8cf2e208277a8d841c2a53553551c971f68f2e33d7e64dc958e8` |
| 全12候选分析 | `outputs/tail_translation_reserve_v2_analysis_20261029/result.json` | `00cb2a2d740047d4839c417a3d80e807ac92cb15d879fd0f6d3fb74d93e9135c` |

## 下一阶段

[从恢复候选重新开始完整 H12 SQP](../plans/v0.0.1/restored_anchor_sqp_20261029.md)已登记，尚未执行。从 selected5 的完整命令重新开放72变量，保留原初态、控制器、caps、buffer、FD/QP及4000响应预算；首次完整响应必须与已闭合所选记录 canonical exact，否则在任何 FD/QP 前停止。tail 的前11周期冻结不继承到新搜索。只有新的完整 H12 witness 通过全量独立复算和来源闭合，才能另行登记原生物理验证；当前恢复结果不直接授权 pilot。
