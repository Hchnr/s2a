# Cap-distance GN：设计与数值契约前置

2026-10-29。本文件只登记研究设计，**尚未获得实现或数值执行准入**；不启动
producer、checker、物理、GPU、世界模型或最终留出。本设计不修改已关闭运行、
原False收据或用户未跟踪文件。后续实施、QA、数值合同和唯一执行配置须另行冻结。

## 目标、已有证据与可证伪判断

总目标仍是固定机器人/控制接口下，desired state 经S2A产生action后，实际状态
可靠跟踪原参考并稳定完成任务。原18正确参考、12 WAM开发、24最终留出保持；
最终稳定至少20/24且各task至少4/8。当前实际仍cursor8、task0、grasp0，H12完整
witness0。软件、保存数据算术和数值witness都不能代替实际控制资格。
[总目标:15](../../goals/036_project_goal_20261023.md#L15)、
[门槛:25](../../goals/036_project_goal_20261023.md#L25)、
[实际控制停止:15](../../../../reports/tracking/trajectory_search/fk_first_feedback_native_20261028.md#L15)。

唯一dual-recovery producer为
`outputs/tracking_v5_state_vector_dual_recovery_20261029/verified.json`，SHA256
`436fbe20f20cf6678aa329c05e928845b563f76cb0a976c11395e16481aba551`。
它保存122响应、52QP、52次NNLS，51QP通过原证书，最终anchor为response116；
无完整witness或执行选点。初始与末anchor的原q分解如下：

| 原归一化平方误差 | 初始response0 | 最终anchor116 |
| --- | ---: | ---: |
| 全12点q总和 | 29.62160641175905 | 27.457719997186015 |
| 前9点q总和 | 10.81940681040697 | 8.49817347558708 |
| 后3点q总和 | 18.80219960135208 | 18.959546521598934 |
| 全窗口最大q | 6.908710040755111 | 6.977793265972015 |
| 连续达标prefix / 完整witness | 9 / 0 | 9 / 0 |

总误差改善来自已达标前九点，后三点整体变差；保存的球外距离loss也由
2.2306862832935668增至2.276085550158358。该统计支持检验目标分配，不能证明
更换目标一定有效。保存后验分析明确不是独立重放或新目标实验：
`outputs/state_vector_dual_recovery_focus_diagnostic_20261029/result.json:4327`、
`:4332`，公式见同目录`analyze.py:88`。

研究判断是：在当前完整响应和因果模型下，优化距原执行容许集合的距离，能否把
有限方向与响应预算用于尚未达caps的组，并找到全部12点同时合格的解。没有该解
就不接入控制；局部loss下降单列记录。

## 必须先关闭的数值契约问题

最新v2独立native复算已经实际覆盖全部122响应、36600数值内步、38065查询，
但`complete=False / passed=False`。结果为
`outputs/tracking_v5_state_vector_dual_recovery_fresh_diagnostic_v2_20261029/verified.json`，
SHA256 `5d83387542539f08ad79bbf201b3ef9011c664a40b46c970f8a03e62b13aa7e4`。
原严格错误仍为`2.137171550842254e-10 > 1e-10`，不能追认通过。

本次已定位首失败为stage2的restore，anchor3至response4：base最紧行为
`braking_upper[4346]`，raw slack差约`+4.44e-16`；target最紧行为
`terminal[4368]`，raw slack差约`-8.66e-16`。经过原`reserve_target=1e-5`
及原terminal权重形成reserve、再除predicted reduction后，fresh ratio相对保存值
差约`-2.137e-10`。这是restore安全余量的派生比较，不能归为TCP或pose merit问题。
当前单点torque代数定位仅属另行零query诊断，不构成本设计已解决数值合同的证据。
[v2登记:44](state_vector_dual_recovery_fresh_diagnostic_v2_20261029.md#L44)。

实施前须先给出可独立验证的算术合同，不放宽原`1e-10`，不跳过比较、改变比较
顺序、代入保存ratio或追认旧False。受保护的八个native函数保持原完整证明：
`query`、`reconstruct`、`compare_response`、`reconstruct_trajectory`、
`compare_rollout`、`command_groups`、`classify`、`check_audit`。

若数值修复选择让新的production predictor采用native算术，该predictor变化必须
独立登记和验证。须先用**同一新predictor的sum-q GN基线**与独立native checker
关闭来源、完整响应、原KKT、严格接纳/半径及repeat，再登记cap-distance对照。
两条研究路线使用同一新predictor、同一初始化和预算，不能把旧predictor的sum-q
与新predictor的cap-distance复合差异归因于merit。新基线不是本文件自动授权的
运行，也不是旧122响应预算的接续；其完整成本另计。若不改变predictor，也须先
解决并验证本节严格核验问题。该前置未关闭时，本设计保持不可执行。

2026-11-02提交前复核：更换predictor算术后，旧3557响应和72条探针不能自动
作为新predictor的逐位相同历史。初始化、完整initial/repeat对照及探针适用性须
另行登记并验证；若必须重新采集初始响应或差分，其费用须在启动前单独列明，
不得暗中追加到下文122响应预算，也不得把旧exact标签移交给新predictor。

## 与已有方法的区别

旧FK一致预览已经定义`1-||error_group||/cap`，并最小化其负部的平方和：
[FK scalar hinge:171](../../../../src/s2a/research/trajectory_search/fk_consistent_trajectory_joint_plan.py#L171)、
[旧目标:255](../../../../src/s2a/research/trajectory_search/fk_consistent_trajectory_joint_plan.py#L255)。
其标量目标与本设计在实数代数下相同；旧H12耗尽160响应仍无完整witness。
[旧结果:8](../../../../reports/tracking/trajectory_search/fk_consistent_trajectory_20261028.md#L8)。
本设计不宣称首次LS、首次球外目标或首次完整H12优化。

新比较只检验：把该已有目标作为**完整投影残差**，接入当前完整状态相关响应、
raw有符号E/Je、因果secant、GN曲率和严格信赖域账本。它不同于只优化最坏q的
epigraph，不同于只修cursor9并锁住前缀，也不同于末周期平移/terminal余量修复。
后几类路线的既有失败和不同预算保持，不能拼接成功点。

## 误差、投影与GN模型

沿用原12点TCP平移和SO(3)有符号误差；平移除`.01m`、旋转除`.05rad`，每点
分成两个三维组，共24组、72维。desired pose、TCP、坐标系、时间、参考顺序和
旋转误差定义不变。[原E与尺度:31](../../../../src/s2a/research/trajectory_search/state_vector_gn_search.py#L31)。

对每个归一化组z，令rho=||z||，固定：

```text
rho <= 1:  r = 0, D = 0
rho > 1:   r = (1 - 1/rho) z
           D = (1 - 1/rho) I + z z.T / rho**3
Jr_group = D @ Je_group
```

rho=1时选择零广义Jacobian，不增加epsilon、平滑半径或扫描阈值。r在边界的
完整经典导数不存在，不能用双侧差分声称其存在；标量平方距离的梯度在边界为0。
每阶段从该阶段的raw E/Je重新构造r、D、Jr，不把Je简单乘一个径向标量。
继续更新raw Je，再施加完整投影导数，保留当前球内组以后重新激活的信息和原
严格因果零结构；不新增差分采集、不拟合新的响应模型。

实际merit的规范表达式固定为原24个q计算的：

```text
Lcap(entry) = .5 * sum(max(sqrt(q_j) - 1, 0)**2 for j in 0..23)
```

其零集正是原24个caps同时成立，不能将它解释为重写参考。r用于局部GN模型；
`.5*r.T@r`与上式在实数中等价，但不宣称两条浮点表达式逐位相同。实现准入前须
冻结实际merit的运算顺序、dtype和独立重算表达式，不允许按运行结果切换版本。

pose QP采用`B=I+Jr.T@Jr`、`g=Jr.T@r`，原damping=1、辅助变量、整目标缩放、
紧活动子集NNLS及原完整KKT保持。无阻尼的预测模型明确为：

```text
Lhat(d) = Lcap(anchor) + g.T@d + .5*||Jr@d||**2
predicted_reduction = -g.T@d - .5*||Jr@d||**2
actual_reduction = Lcap(anchor) - Lcap(full_response)
ratio = actual_reduction / predicted_reduction
```

预测下降不包含阻尼，零step的预测下降为0。采用该展开形式是本次新merit数值
合同的一部分，不能与另一种有限精度求值方式混用后隐去差异。

## 同单位的pose下降门与保持项

不能只替换B/g并声称原停止数学完全不变。旧pose门使用
`min(original_minimum, .1*F_anchor)`；单个near-cap组的新loss约为`F**2/8`，
可用下降小于旧门时，响应会在witness检查前被挡住。
[原预检查:447](../../../../src/s2a/research/trajectory_search/state_vector_dual_recovery.py#L447)。

本设计因此同步登记唯一必要的单位一致变化：pose模式的predicted和actual最小
下降分别为`min(原对应minimum, 原pose_reduction_fraction*Lcap(anchor))`。
原minimum均为`1e-4`、fraction仍`.1`，不新增数值常数或调参网格。只在完整
响应后按新merit计算实际下降和ratio；原witness优先级保持。restore继续使用原
reserve/violation及原下降门，其数学、权重和模式切换不变。

以下全部保持：

- 原actual368、completed367命令/rotation latch/finger/因果历史、chunk46原H12
  参考、完整机器人模型及资产。predictor不变时，从原3557和72 signed nominal1e-4
  探针重新初始化；predictor改变时先满足上文单独的初始化/来源/费用合同，不能
  默认复用旧exact历史。不预载旧final Je、旧优化器状态或后验最优候选。
- 全部原命令box、rate<=.25、global displacement<=.1、内步关节/制动/terminal
  安全、reserve_target=1e-5、safety_buffer=1e-8、原数值搜索上限、完整KKT1e-6；
  数值上限不授予放宽的执行资格。原真实执行caps仍`.01m/.05rad`。
- 原rejection反馈来源、raw Js/Jq/Je因果更新、alpha次序、信赖域半径和扩缩、
  restore/pose模式切换、完整initial/repeat及失败留存；不额外加入前9硬caps保护。
- 原独立`1e-10`决策比较及严格分类、原全部24个caps和完整audit的witness门；
  已达标点可以按原搜索机制暂时变化，但不完整合格的响应绝不派发到实际控制。

原配置数值依据：
[固定配置:2](../../../../configs/tracking/trajectory_search/state_vector_dual_recovery_20261029.json#L2)、
[原完整witness:342](../../../../src/s2a/research/trajectory_search/state_vector_dual_recovery.py#L342)。

## 有限预算、实施前检查与停止门

拟议cap-distance路线保持seed42、60stage、122完整响应上限，预留最后initial
repeat；每次QP仍最多100迭代，NNLS按原规则单次计费，无fallback或自动重试。
原半径initial=.005、minimum=.00015625、maximum=.02及alpha次序保持。
不因新目标便宜或昂贵而增减预算。

| 拟议路线 | 完整响应上限 | 数值内步上限 | query上限 |
| --- | ---: | ---: | ---: |
| 唯一cap-distance producer，含repeat | 122 | 36600 | 38065 |
| 独立checker，重放全部已收费响应 | 122 | 36600 | 38065 |
| 两者合计上限 | 244 | 73200 | 76130 |

这是未来路线的上限设计，不是新增已执行成本。若必须先登记新predictor的sum-q
基线，其producer/checker各自的同规模预算另计，不挪用旧余量、不计入上表后冒称
总成本仅244；必须在任何启动前明确登记总预算及各阶段墙钟/初始化上限。本文件
不生成执行config或自动授权任一路线。CPU显式空CUDA/LD、各数值库单线程，独占
outputs、来源SHA冻结、实际wait/reap；GPU继续禁止。

实施前必要检查包括：投影球内/外/边界与切向导数、原因果mask、已达caps组重新
激活、QP/预测与实际merit的一致单位、近caps不被旧尺度形成死区、restore未变、
原安全/完整witness拒错及预算末次repeat。这些合成检查不计研究响应或控制收益。
随后还须独立输入/源码准入；仅代码存在、测试通过或dry不授予执行资格。

结果固定同时报告24组q、12点位置/旋转误差、前9/后3分解、Lcap、sum-q、maxq、
prefix、安全最小行、restore次数、QP/NNLS与完整费用，不按最终更好指标挑口径。
没有完整witness时，只记录是否改善尚超限组，不能以新loss下降宣布跟踪成功。

单次QP不满足原证书、或某次trial不满足安全/接纳条件时，仍按原状态机拒绝该
候选、计费并执行原反馈/缩域规则；不得把这些原本允许的拒绝改成整轮立即终止，
也不得把不合格候选标为合格。来源、initial/repeat、独立证书审计或完整重放、
`1e-10`、严格分类/账本一致性等整轮资格门失败时，才保留False并停止；任何绕过
原安全门或派发不合格轨迹同样不得继续。达到122响应、60stage或原半径停止规则
仍无完整witness，则本有限研究没有获得可执行解；不自动增预算、换权重、保护
前缀、换初态或据此声称物理不可达。

只有完整12点/24组原caps、完整安全、来源和独立严格核验全部通过，才可另行
登记一条有限原生闭环pilot及独立命令重放，检查实际state error和控制约束。
该物理预算与停止门必须事前固定，不包含在本数值预算中。本文件不启动或授权
pilot，不打开WAM开发或fresh。实际控制、完整任务、实时和真机资格分别判定；
原18/12/24及最终20/24、各任务4/8门槛始终保持。

本阶段截至落盘仅完成设计；新增实现、测试、QP、NNLS、响应、query、物理、
世界模型及GPU均0。
