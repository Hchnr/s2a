# 当前目标与工作状态

本页维护当前事实，按日期的完整研究记录由索引定位。目录整理不改变研究目标。

2026-11-02算法域核查：保留原安全/变化率行，仅诊断完整input域时，终点支持
下界由原root约1.45变为约-6.58。旧局部排除不能升级为机器人不可达，负支持
也不证明完整caps可行。29项专项及2LP审核通过，0新响应/物理/GPU。
已重新审查旧4000响应非线性搜索的失败，后续将设计不同起点的非线性可行性
恢复，避免简单重跑或只扩大线性域；见[域诊断报告](../reports/tracking/trajectory_search/input_domain_support_20261102.md)。

2026-11-02固定终点方向核查已完成：10双侧响应逐值exact，61项专项通过。
三种步长终点旋转改善但仍超cap，实际reserve分别-171.8/-39.7/-8.2，全部joint
audit失败；预测却保有约.9，说明较大步的非线性安全误差不能忽略。没有候选
下发、没有新物理/GPU。后续需区分算法域限制与非线性安全模型瓶颈，不能把
局部下界升级为物理不可达；见[完整报告](../reports/tracking/trajectory_search/local_support_direction_probe_20261102.md)。

2026-11-02完整瓶颈诊断：48个逐cap乐观LP及原单位对偶检查通过；初始和最终
局部模型的终点旋转支持下界，在原实际buffer下仍为1.445/1.452（达标须≤1），
reserve .9仅改变约2.1e-4。提示需审查终点方向与非线性模型可信度，而非继续
降低进展门。LP步长接近.2，不能把局部模型下界当作物理不可达证明。
38项专项通过，0新响应/物理/GPU；见[完整诊断](../reports/tracking/trajectory_search/local_tracking_support_20261102.md)。

2026-11-02最新：小进展阈值组的保留36响应已另行独立重放通过，原审核接线
失败结果保持False。新组12stage/7接纳，仍caps12/24、prefix2、witness0，
最终loss2.008885略高于历史安全反馈2.008818。按预登记停止继续调整标量
进展阈值，转向完整轨迹误差、约束方向与可执行性瓶颈审查。94项专项通过，
无GPU或物理执行；总目标与完整验收均未完成。
见[实际结果与独立补齐报告](../reports/tracking/trajectory_search/small_progress_search_20261102.md)。

## 总目标

在固定机器人和控制接口下，将世界模型给出的期望状态轨迹，结合当前实测状态
与因果历史，转换为可执行action，使实际状态可靠跟踪目标，并在留出新初态上
稳定完成任务。主要指标是经过固定前向控制系统后的状态、安全及任务稳定性。

原18正确参考/12 WAM开发/24最终留出门槛保持。最终稳定需≥20/24且各任务≥4/8；
最终留出尚未打开。当前总目标未达成，局部数值与测试通过不能代替该验收。

## 仓库整理：目录迁移与分轮验收

[结构设计](architecture/repository_design.md)定义层次、目标目录、身份迁移协议、
分阶段提交及验收。先建立来源清单与路径基础，再迁移trajectory-search完整切片，
随后按依赖闭包整理其他领域。算法、控制策略、数值容差和验收门槛保持。

所有文件及产物在当前仓库维护；根目录用户未跟踪文件不纳入整理。
生成的outputs、现有环境、XML和运行证据本次保持原位置。GPU继续禁用。

已完成设计提交`0c5ed51`和3276文件基线；D1路径/来源身份基础已实现，36项CPU
专项及真实catalog接线通过。迁移前测试发现4976节点，5项既有PyArrow/NumPy
导入错误已单列。[迁移记录](migrations/README.md)说明证据和接口边界。

D2首批85项已按主题移动，配置/计划/报告原字节保持，8算法模块的38个顶层
函数/类AST保持；迁移后182个原QA节点完整保留并通过。根测试辅助导入统一包
身份。最终302项统一CPU QA、153个Python文件Ruff/格式及两真实输入dry通过；
旧4976测试节点完整保留、新增120，仅原5项环境collection错误。结构/身份检查
通过，D3其余领域尚待迁移；未重跑生产或升级原实验资格。

D3a已把其余377个库模块迁入领域包，src/s2a根目录只保留包入口。库层反向
依赖scripts的两处函数闭包已提取；当前脚本统一采用scripts.*导入身份。全量发现
初次5169节点、修复新增拒绝边界后最终5171节点；原5096节点的函数、参数和顺序
完整保留，仍只有原5项
PyArrow/NumPy环境错误。32进程674项CPU回归用时约49秒，首次669通过、5项
旧路径/导入匹配失败按原记录保留；修复5项独立并行复核全部通过，另两相关模块
21项通过（包含新增2项拒绝边界测试）。两真实dry通过，
QP/响应/查询为0。脚本、测试及历史证据的其余物理迁移仍待D3b–D4执行。

D3b首批775个共享/审核脚本已按领域移动，1141个当前调用方同步明确包路径；
独立metadata审核仍保持仅标准库，原数学校验函数AST保持。累计清单登记3280个
历史身份，其中1237个已移动。全量发现5172节点、原节点及参数保持，仅原5项
环境错误。2254项初测2107通过、147项旧路径等迁移适配失败；修复后35个完整
相关模块486项复核482通过，最后4项独立复核全部通过。全仓2470个Python文件
Ruff/format通过。上述为分轮验证，不冒称完整pytest或同版本全量重跑已通过。

D3b其余690个脚本已按设计移动，57个测试调用方明确当前包和存储位置，新增17个
包标记；scripts根目录只保留README和包入口。累计3280个历史身份中1927个已
移动。全量发现仍为5172节点，原节点/参数/顺序保持；全仓2487个Python文件
Ruff/format及静态导入边界检查通过。816项32进程CPU初测781通过、34失败、1项
按原条件跳过；修复后11个完整模块148项复核139通过、9失败，随后9项复核8通过，
最后1项单独修复复核通过。原失败收据均保留，未称最终版本全部816项重跑通过。

PyArrow相关5个模块另在仓库现有Python3.12/NumPy2环境中执行14项，全部通过；
3.11数值环境的原5项collection错误仍保留，未安装或修改共享环境。历史CLI帮助
与发布桥接仅在原字节的隔离子进程中回归，禁止模型/数组读取及混入当前源码，
不授予当前运行或控制资格。

D3c已实际迁移439个测试文件，新增71个真实包标记，测试根只保留README、
__init__和原字节conftest。pytest保留原scope，仅移除旧scripts搜索路径；并发CPU
入口已写入`scripts/maintenance/run_cpu_tests.py`。原5172节点完整保留，新增5项
原生运行时来源篡改拒绝用例。全仓2559个Python文件Ruff/format通过；本阶段迁移
回归已分轮收尾，原环境限制与验证版本分别保留如下。

首次1205项回归1141通过、64失败。分工修复来源/CLI定位后，首次64进程全量
5177项报告4987通过、188失败、2跳过，另有原5项环境collection错误；该轮还
暴露迁移时遗漏的模块替身导致一份真实config被合成fixture覆盖，source_unchanged
为False，因此整轮不作为有效通过记录。全部进程已回收，原config已从Git精确
恢复并核原SHA，完整3280项复核没有其他漂移。测试替身和临时写入边界已修复，
其余历史loader/独立来源证明及数值环境差异继续分别处理，原失败收据保留。

其中streaming的4项严格数值失败已用迁移前原test及原数学库，在隔离3.11环境中
实际复现：原26项22通过、同4项失败，误差逐项相同；当前开发环境原容差下26项
通过。两环境的Python/NumPy/OpenBLAS均不同，不能单独归因其中一个版本。本次
不修改算法或容差，后续全量记录须分开这些既有数值限制与新增迁移失败。

修复后的64进程全量5177项报告5169通过、6失败、2跳过，来源前后相同；6项为
上述4项既有数值差异和最后2项迁移适配失败。随后仅修改这两个测试，完整相关
模块51项并发复核全部通过，没有声称最终版本重新执行全部5177项。兼容开发
环境另有Arrow14项与streaming26项共40项全部通过，来源冻结且实际回收。当前
没有未闭合的已知新增迁移失败，但不能称单一环境完整pytest全绿。源码身份与
结构检查通过，累计2366个原文件已移动；完整raw/nav迁移和最终D5仍待执行。

D4已实际移动其余898份配置/计划/报告，全部928份原始文件的字节、大小和Git
blob身份保持。README收敛为目标与导航，三份基线README原文另行归档；AGENTS
只更新001目标导航并追加CPU偏好，原约束保持。25个测试模块只适配当前I/O边界，
Ruff/format通过，算法和raw JSON值保持。累计3264个原文件已移动，根平铺为
src1/scripts2/tests3/configs1/plans1/reports1。D4提交时最终D5尚待执行；以下记录
本次提交前的D5验收，研究目标仍未达成。

D5首次64进程全量发现5177节点，5138通过、37失败、2跳过，另有原5项环境
collection错误；37项中33项为深层raw读取迁移适配，4项为已复现的原数值差异。
修复后全量v2为5170通过、5失败、2跳过，源码前后相同；其中仅1项仍为遗漏的
合成fixture PLAN读取，现已补齐。独立审查另补齐两个preview真实BASE_REF读取，
显式传入各自引用并严格核原声明/catalog/current SHA，替换引用不回退加载基线。

最终冻结版本的14个完整相关模块179项全部通过，覆盖原33项迁移失败；兼容环境
40项全部通过。重新发现5177节点及原顺序不变，仅保留原5项环境collection错误。
两条最新真实入口dry、完整结构及静态导入检查通过；四个最终运行owner的3646项
源码/输入清单相同，验证期间无漂移且所有子进程实际回收。全仓2559个Python文件
Ruff/format通过，928份raw再次核原字节一致，累计3264个实际移动符合设计。

最后修复后没有再完整执行全部5177项，不能把上述分轮结果称为单一版本完整
pytest全绿。目录、身份及本轮读取回归验收不授予所有历史入口执行资格；原控制
验收、失败证据和GPU禁令保持。当前累计清单SHA及各轮收据见
[D5迁移记录](migrations/README.md#d5最终结构与读取边界验收)。

## 整理前的控制与实验状态

- 已有明确DROID adapter与LIBERO OSC接口、因果参考/历史、前向预测及独立物理
  评估链路；各自证据等级和控制语义分别记录。
- 当前实际资格仍cursor8/task0/grasp0，尚无新的完整H12执行轨迹资格。
- 最新dual-recovery唯一生产已正常关闭：122响应、52QP、52次拟合；51QP通过
  原证书，完整数值witness仍0。保存数据表明总误差改善主要来自已达标前九点，
  后三点仍超原caps，不能据此宣称可靠跟踪。
- 首次checker在来源读取阶段失败，原False与零响应失败记录保持。隔离check-only
  v2实际复算了全部122响应/36600内步/38065查询，原完整KKT账本通过，但最终
  fresh_ratio差2.137171550842254e-10超过原1e-10，complete/passed仍False。
- 整理后仍须解决严格决策核验及轨迹目标分配问题，再按独立登记继续控制研究。
  不因为搬目录追认失败、放宽容差、追加旧预算或自动恢复模型任务。

最新证据路径与SHA见[设计中的迁移前快照](architecture/repository_design.md#8-迁移前实验状态快照)。

## 验证边界

2026-11-02：新增尚未接入搜索的独立 OSC 算术实现。原生 NumPy 环境下 129 个
输入与独立回放函数逐字节一致，三个专项测试通过；原预测器及容差未改。
这只证明力矩代数一致性，完整响应、严格 ratio 及任务资格尚未通过。
证据和下一步见[算术契约报告](../reports/tracking/trajectory_search/native_osc_arithmetic_20261102.md)。

后续固定命令完整响应对照：两侧共8条H12响应通过原响应容差，但并非完全同值。
首次失败命令对的条件性 ratio 差降至5.18e-11，尚未授予完整搜索资格；另发现
Jacobian存储布局影响，下一步核查dot/@路径。失败、额外成本和5项专项测试见
[完整响应报告](../reports/tracking/trajectory_search/native_replay_response_contract_20261102.md)。

最新v4已在固定四条命令上得到与独立回放完全同值的完整H12响应：修正任务
dot/@并在采样时保留J布局，响应及独立指标最大差0，13项专项通过。
来源、实际成本及边界见[布局契约报告](../reports/tracking/trajectory_search/native_replay_layout_contract_20261102.md)。

新预测器初始化及72个signed1e-4探针现已重新验证：两侧各74完整H12响应、
评分和Js/Jq/Je矩阵精确同值，因果零结构通过。生产端32批并发，全部进程实际
回收；23项专项通过。原18/24初始化caps尚未全部满足；证据与成本见
[新探针报告](../reports/tracking/trajectory_search/replay_order_stencil_20261102.md)。

同预测器sum-q对照已完成：122生产响应及全部独立复算、完整KKT/反馈/原1e-10
fresh决策审核通过，响应同值。总q29.62→27.47，但后3点18.80→18.96，仍18/24
caps、prefix9、witness0；未获得物理资格。61项专项通过，下一步按设计执行
同预测器cap-distance对照。费用与边界见
[sum-q报告](../reports/tracking/trajectory_search/replay_order_sumq_20261102.md)。

Cap-distance目标与显式objective共享驱动已实现，58项专项通过；共享驱动的
sum-q路径与闭合原基线完整软件回放同值，没有新增研究运行。浮点边界的有理化
求值已另行固定。该实现阶段见
[实现报告](../reports/tracking/trajectory_search/cap_distance_implementation_20261102.md)。

独立cap审核及唯一对照现已完成：79项专项通过，89生产响应和全部独立复算、
原KKT/反馈/严格fresh及owner源码/回收审核通过。loss2.231→2.010、后3点改善，
但caps18/24→12/24、prefix9→2、witness0，因60阶段上限停止；不派发到控制，
不授予物理或任务资格。下一步核查末端旋转、活动约束与响应失配；完整指标、
成本与来源见[cap-distance报告](../reports/tracking/trajectory_search/cap_distance_gn_20261102.md)。

保存模型诊断显示安全约束与局部梯度显著竞争，不能仅凭接近global盒解释失败。
随后固定单旋转方向诊断完成1QP、两侧6响应并通过独立数值/源码审核：旋转误差
.13113→.12820rad，但制动安全slack为负，整体cap loss变差，候选未下发。
75项专项通过。下一步区分Js局部失配与有限步曲率，原安全门保持；见
[约束诊断](../reports/tracking/trajectory_search/cap_distance_constraint_diagnostic_20261102.md)和
[旋转探针报告](../reports/tracking/trajectory_search/rotation_constraint_probe_20261102.md)。

固定缩步/小扰动诊断已完成：两侧16响应全部独立核验、85项专项通过。两个小
尺度的一侧安全斜率约-8.9e-5，旧模型约-7.2e-6，支持已测方向的局部安全模型
失配；不能只归因于大步曲率。alpha=.125及更小虽数值安全，完整caps仍失败，
未下发候选。下一步验证因果安全模型局部修正及新方向，原门槛保持；见
[局部性报告](../reports/tracking/trajectory_search/rotation_direction_locality_20261102.md)。

一次原有因果安全更新在保存的小尺度样本上改善预测，独立实现逐值一致；但随后
新方向完整复算仍不安全：min slack=-5.50e-6，caps13/24、prefix2、witness0。
实际1QP/1NNLS及生产/独立各3响应（共6/1800内步/1874查询），46项专项通过，
未下发候选。秩一修正不保证其他方向安全，下一步需判别新方向局部失配与大步
误差；见[修正探针报告](../reports/tracking/trajectory_search/corrected_rotation_probe_20261102.md)。

修正后方向固定8样本的独立复算已完成，共16响应/4800内步/4994查询，0QP/拟合。
两小尺度斜率约-7.9e-6/-7.7e-6，模型-7.2e-6，未重现旧方向12倍失配；半步数值
安全，但所有非零尺度完整cap loss更差，完整witness仍0。24项相关QA通过。
下一步回到全部位置/旋转联合目标，不再只围绕尾部旋转下降选方向；见
[修正后局部性报告](../reports/tracking/trajectory_search/corrected_rotation_locality_20261102.md)。

恢复完整cap-distance目标的一次QP与固定五尺度验证完成（1QP/1NNLS、14响应/
4200内步/4370查询，47项专项通过）。alpha .25及更小尺度数值安全且联合loss
下降，但caps仍12/24、witness0；安全余量消耗会再次触发原restore模式。
小步预测误差仍明显，下一步核查当前anchor完整局部模型重建的采样域与数值条件，
不靠反复单方向修正代替；见[完整目标探针报告](../reports/tracking/trajectory_search/corrected_cap_distance_probe_20261102.md)。

anchor85的±1e-4单轴探针只能通过64/72轴原command门；固定连续区间联动枚举
可获得72维、condition约4.049且各因果前缀满秩的采样域。几何选择/因果拟合
基础实现及24项专项通过，并逐值复现实际anchor的72方向；尚无新机器人响应、
实际局部模型重建或控制资格。下一步先登记并行采集/独立审核预算再执行；见
[采样域报告](../reports/tracking/trajectory_search/anchor85_sampling_domain_20261102.md)。

anchor85的72联动探针已实际采集并独立复算：148响应/44400内步/46209查询、
33model，32不同worker均exit0且回收；31项专项通过。Js/Jq/Je新模型及独立正规
方程重建通过原1e-10审核，完整响应/anchor/repeat逐值一致。13个非线性不安全
数值探针如实保留，未下发。已见联合alpha .25方向安全预测误差由3.80e-5降至
8.50e-8，但该后验比较不授予新方向资格。下一步用新模型生成完整目标方向并另行
验证；见[联动局部模型报告](../reports/tracking/trajectory_search/anchor85_coupled_stencil_20261102.md)。

新模型生成的联合方向已完整审核（1QP/1NNLS、14响应/4200内步/4370查询、43项
专项通过）。alpha .125下降2.56e-4、实际/预测比.991、reserve .929，保持原pose
进入条件；但旧顺序会先接纳alpha .5并重新restore。全部caps仍12/24、witness0。
下一步显式设计保护reserve的接纳变体及有限迭代比较，不追认旧规则已选较小步；
见[刷新模型新方向报告](../reports/tracking/trajectory_search/refreshed_cap_distance_probe_20261102.md)。

显式reserve接纳与原根域继承已实现，默认55QP/122响应软件回放保持完整答案
SHA不变；新方向单stage回放按原顺序选alpha .125、reserve .929，独立审核及
三类篡改拒绝通过。29项专项通过，尚无新响应/新有限搜索，不能将软件回放计为
控制进展或连续收敛证据；下一步独立登记有限迭代及完整响应审核，见
[接纳变体实现报告](../reports/tracking/trajectory_search/reserve_entry_acceptance_implementation_20261102.md)。

reserve接纳的实际匹配有限对照已完成且两侧独立审核通过：baseline60stage/91生产
响应、reserve7stage/28生产响应，共238双侧响应/71400内步/74260查询、4model、
67QP/67NNLS。baseline loss2.007688，reserve2.009812；两组caps12/24、witness0。
reserve无restore但只接纳1步，后续安全reserve拒绝未满足原反馈资格，缩步至下限。
因此不采用为优胜默认；下一步核查这些已付费拒绝数据的模型校正作用。40项最终
专项通过，所有进程回收；见[实际有限对照报告](../reports/tracking/trajectory_search/reserve_entry_search_20261102.md)。

后验检查不支持盲目增加reserve拒绝反馈：最小安全拒绝样本原Js误差仅2.81e-8，
秩一更新未稳定改善其他尺度。更直接的是原pose QP只保留半数reserve，而实际
接纳要求.9；24个reserve拒绝的模型预测均已低于该门。停滞anchor .928585时，
最小alpha .0625预测.899566，半径缩小仍贴原下限。下一步显式对齐QP与接纳条件，
保留旧默认和原门；见[后验核查](../reports/tracking/trajectory_search/reserve_rejection_shadow_20261102.md)
与[下限对齐设计](../plans/v0.0.1/tracking/trajectory_search/reserve_aligned_floor_design_20261102.md)。

pose下限对齐策略已实现并完整验证一次新方向：1QP/1NNLS、14响应/4200内步/
4370查询，43项最终专项通过，旧默认55QP/122响应回放完整答案SHA保持。
首个原规则合格步alpha .5下降1.88e-4、reserve1.056（旧对应步.855），无需立即
restore；但全部caps仍12/24、witness0。下一步另登记有限连续迭代验证，不能用
单步效果宣称解决停滞；见[下限对齐新方向报告](../reports/tracking/trajectory_search/aligned_cap_distance_probe_20261102.md)。

下限对齐的实际有限迭代已独立审核通过：9stage/4接纳/0restore，最终loss2.009187、
reserve .911682，仍caps12/24、witness0，radius到下限。新费用44双侧响应/
13200内步/13730查询、2model、9QP/9NNLS；50项专项通过，旧baseline60QP/91响应
闭合回放完整答案不变。相比只加接纳门有更多下降，但未解决停滞；末段预测reserve
.9088而实际.8901，下一步核查当前安全拒绝的临界模型误差/校正，不能复用旧结构
不一致解释。见[连续迭代报告](../reports/tracking/trajectory_search/aligned_reserve_search_20261102.md)。

对齐后stage5的最小安全reserve拒绝response13用于后验更新时，独立实现逐值一致；
同方向未拟合alpha .5的Js最大误差由3.90e-6降至2.50e-8，其他已见方向也有不同
程度改善，但仍有reserve假阳性，Jq在最远方向变差。0新响应/查询，未改旧接纳。
下一步仅隔离Js安全反馈，保留Jq/Je及完整实际门，不把后验改善提升为新方向资格；
见[后验校正报告](../reports/tracking/trajectory_search/aligned_rejection_shadow_20261102.md)
与[安全反馈设计](../plans/v0.0.1/tracking/trajectory_search/reserve_safety_feedback_design_20261102.md)。

仅Js的reserve安全反馈已显式实现并独立审核，原pose ceiling来源保持优先、
Jq/Je及anchor不变。最终52项专项通过；六stage保存前缀回放选择response13并
通过五类篡改拒绝，旧baseline60QP/91响应完整答案SHA保持。0新响应/查询，尚未
验证反馈后的新方向或连续迭代；下一步另登记有限新执行，见
[安全反馈实现报告](../reports/tracking/trajectory_search/reserve_safety_feedback_implementation_20261102.md)。

仅Js反馈实际新搜索已完整native审核：12stage/6接纳/2安全only更新、0restore，
loss2.008818、reserve .903607，仍caps12/24、witness0。新费用40双侧响应/
12000内步/12482查询、2model、12QP/12NNLS；71项专项通过，旧aligned9QP/22响应
闭合回放答案不变。末3stage共15候选未执行，因全步预测下降低于原1e-4而缩步
停止，不能据此证明物理无可行下降。下一步固定核查未调用方向及其完整跟踪意义，
不直接放宽门或仅按微小loss选优；见[实际搜索报告](../reports/tracking/trajectory_search/reserve_safety_search_20261102.md)。

停止前3个固定未调用方向已新增完整核查：10双侧响应/3000内步/3122查询、2model，
0新QP/拟合，43项专项通过。三方向均joint/buffer安全且loss下降，前两reserve
失败；最小方向reserve .901292、下降3.10e-5，低于旧进展门，仍caps12/24、witness0。
这证明有小幅局部下降，不证明累计跟踪改善，也未追认旧接纳。下一步仅作为显式
新算法对照设计更小进展门、保持原实际安全和完整资格及最大预算；见
[固定方向报告](../reports/tracking/trajectory_search/deferred_cap_directions_20261102.md)与
[进展门对照设计](../plans/v0.0.1/tracking/trajectory_search/small_progress_comparison_design_20261102.md)。

迁移前实现阶段的专项、样式及真实输入记录见
[实现准入报告](../reports/tracking/trajectory_search/state_vector_dual_recovery_implementation_20261029.md)。
最终源码的Ruff/format、结构、导入图、179项修复复核、兼容环境40项和两真实
dry通过；5177项发现及顺序保持。全量v2、后续修复和最终专项分别登记，原5项
PyArrow/NumPy导入错误与4项已复现数值失败不改容差、不伪装通过。最终验收后
当时仅更新本状态、迁移说明和PR文案，未再修改源码、测试或原始输入；上述
2026-11-02 研究实现与专项检查单列，不继承迁移阶段全量验证结论。
