# 固定前缀优化余量对照

本轮按[执行前登记](../plans/v0.0.1/prefix_optimization_reserve_20261029.md)检验优化器中固定1e-4的归一化平方误差余量。原actual368实测状态、completed367控制历史、chunk46原H12目标、共同warm和完整96资产模型保持。GPU禁用；原18正确参考/12WAM开发/24最终留出目标不变。

## 改变与资格范围

上一轮[完整callback回缩](prefix_callback_backtracking_20261029.md)的1012响应、75份完整ray和13份anchor审计已独立关闭，严格标签一致、完整witness0。最后三个半径均在已达前缀旋转边界停滞；不能据此归因于单一数值因素，也不能放宽实际验收。

新[求解器](../src/s2a/prefix_optimization_reserve_plan.py)仅将SLSQP已达prefix约束改为`1-1e-4-q>=0`。普通ray接纳仍按原严格prefix、原完整审计、1e-8非线性余量和原下降判据；全部H12的witness仍只要求原10mm/.05rad caps与完整审计，不要求优化余量。目标cursor的epigraph约束、目标、梯度、Jacobian、安全约束、差分、缓存、ray次序和预算均保持。

[独立源码审计](../scripts/prefix_optimization_reserve_audit.py)严格逆向恢复唯一prefix表达式、顶层常量、函数名、说明文字和返回余量字段，再比较完整模块AST。冻结旧solver和账本各按固定SHA验证；额外变化或缺失变化均拒绝。证明仅覆盖solver差异，不冒称新main与旧main相同，也不替代数值验证。

独立checker绑定证明的全部来源，继续使用冻结数值重构与callback账本审计，重构全部唯一响应并检查每份完整ray/anchor。来源准入要求旧checker原owner真实wait/reap0、1012/303600/315745计数、75/13覆盖、严格分类无分歧及来源不变；已实测共同warm最小prefix余量为.000846374733，大于登记常数。

实现提交为`be0d6ad`（solver、独立审计及测试）和`bf36aa1`（完整执行/检查/来源入口）；登记提交`abe22a0`。旧已执行代码、测试、计划及结果均保留。

## CPU验证

统一native QA为`outputs/prefix_optimization_reserve_native_qa_20261029_h56dgw22/result.json`，SHA`a79441b12430d2988f8de30cc1b973563ee444f29220cf267c60da4e142e1042`。318项通过/36.79秒，实际wait退出0，38份来源前后SHA一致；日志SHA`ceb13b8300f525365e82df5064a7b4dd4bfe0b7c16b654378aa2553c8a06ae85`。Python3.11.15/NumPy1.26.4/SciPy1.15.3，空CUDA/LD与单线程CPU。两条旧SciPy warning来自保留困难例。

覆盖旧252项，以及新11项solver语义、24项独立AST/账本、31项执行接线测试。新测试验证prefix长度0/1/2的唯一常数变化，原caps内但未达优化余量的完整warm/callback/FD witness仍接纳，普通ray接纳与目标/安全/Jacobian未变，以及AST篡改、部分旧复算、缺少ray/anchor和不同QA的拒绝。真实SLSQP困难H2仍取得完整witness（观察到102响应/7迭代），内点例14响应/1迭代；未据此调整生产参数，不能将合成成功计入实际任务结果。

## 正式状态

五项启动前检查已完整关闭：`outputs/prefix_optimization_reserve_admission_20261029/result.json` SHA`96cbe5e3691d06915f6263083856e8641b775a39f2ba3f3ddf96f5e67acab036`。34个Python文件Ruff/format、新producer/checker dry及owner diagnose dry均实际wait0，无信号或重试，38份来源未变。2267个输入/来源引用通过，owner加QA/日志后2269；8个冻结数学函数与四来源的余量AST反向核对均通过。warm SHA仍为`16bdfbb266ef13d2588c8116e13511d93df936d72ae8a57d19df0b01ed3acee0`，新响应/查询/内步均0。

此准入记录生成时尚未启动新producer。之后必须使用本轮唯一独占输出、同一QA和原4000响应/60迭代/60阶段预算，producer1200秒、checker5400秒，无自动重试；完整独立结果产生前无新数值资格。

随后唯一producer启动，原tool session17677、child PID3827991/start_ticks615340066、owner PID3825995。运行快照`outputs/prefix_optimization_reserve_execution_checkpoint_20261029/producer_started.json` SHA`00ecb0dfd94c0fb977a817f8bb8fb0bd23cb630b16caa0b9e606bfe88294767b`核对当时300响应、真实/proc父子身份与存活、8项CPU环境匹配、38份QA来源不变。owner登记2269份来源，登记SHA`0bde8f36d4f2a29c91aeb0bbcafc33a23938a670985cc25fe84e3cf5eaff5af2`，launch SHA`cdd56527287b73e9defe109a1c943bfcfce56545ef769dfb4a0d74c00b4f81a5`。该快照保留主机实际UTC时间；下面的完成记录取代其运行中状态，但不改写原快照。

## 搜索完成，完整独立复算运行中

原producer及owner现已正常退出：tool17677返回exit0，保存的actual wait/reap均0，无信号或重试。结果`outputs/tracking_v5_prefix_optimization_reserve_20261029/verified.json` SHA`321a096afd1f0a571cc70f51e3ba3dba6ec3da2973eb83981665b754bc1bcf4a`；owner结果`outputs/tracking_v5_prefix_optimization_reserve_execution_20261029/result.json` SHA`b14ff55ef82ebb6b5a28f20f2f76af7244e774bad15b35fafd9a1c50a384c581`；release SHA`0689ad83f72413c805a14cfa9e2dc87c92cbeebbbf4824b6cd19fab454b25b29`。执行后已用`closed_owner`、完整来源重读及同QA的check-owner dry核验。

以下是**保存的producer结果及只读账本诊断，完整独立数值复算尚未完成**：

| 指标 | 本轮结果 |
| --- | --- |
| 唯一完整H12响应 / 数值内步 / 查询 | 4000 / 1200000 / 1248001 |
| 迭代 / 阶段 | 53 / 54 |
| 原连续达标prefix / 完整witness | 9 → 9 / 0 |
| 普通下降anchor / 全部anchor（含warm） | 53 / 54 |
| ray尝试 / 新响应 / 缓存复用 | 177 / 124 / 53 |
| 最终selected | response3993，等于最后接受anchor |
| cursor9位置 / 旋转 | .011473285411891755m / .05737175685151507rad |
| cursor9归一化平方误差（位置 / 旋转） | 1.3163627814272814 / 1.3166073936917462 |
| 终止原因 | response_budget；最后stage53在gradient_probe axis6请求被拒绝 |

53个有callback的阶段各接受第一条合格ray，原prefix均为9；阶段全部使用半径.01，未进入.005或.0025。stage53只有6条新差分响应、无callback/ray，随后达到原4000响应预算。因此本轮未复现前轮三个半径连续回缩失败，但不能把预算耗尽写成收敛或已经找到执行解。

177个ray中53接受、124拒绝；缓存命中53个中5接受，新响应124个中48接受。alpha尝试次数为1:53、1/2:48、1/4:40、1/8:25、1/16:6、1/32:4、1/64:1，后两个更小alpha未使用。53个下降的target最大平方误差合计减小.008340708109477957，由共同warm的1.3249481018012241降至1.3166073936917462，仍大于原门槛1；只消除了初始超限量约2.57%。与前轮callback最终selected的1.3188159044570198相比也没有新增达标端点。

优化余量不是额外接纳门槛：53个新anchor中46个的原prefix最小余量小于1e-4，7个达到；全部仍满足原caps。最小余量出现在stage5/response455，约5.3140769e-8。最终anchor的最紧prefix为cursor5位置，余量.0001064182884392；cursor4旋转为.04999687293281797rad。不能宣称整个接纳轨迹都维持了优化器余量。

末个完整stage52的alpha1（response3990）同时违反prefix与非线性安全并无target下降；alpha1/2和1/4（3991/3992）仅无target下降；alpha1/8（3993）接受，下降1.818508720630163e-5。最终非线性安全最小值为braking_upper[cursor6,joint7]的1.0273032891028322e-6。原关节/命令审计容差与实际pose caps均未调整，不能用这些局部改善替代物理或任务成功。

搜索关闭后的独立owner启动准入记录为`outputs/prefix_optimization_reserve_check_admission_20261029/result.json` SHA`7ac95f933a295e805ed5a590e8808bfebcf5081d77145f0c16d70b4512582189`。check-owner dry实际wait/reap0、来源前后不变、2275个来源引用通过，新增响应/物理/模型/fresh均0。驱动SHA`7ddea4f8074a59d015eb36a5b8a8078bb22df58f5a42dd6b6f93f54100d45267`；新观察驱动经过Ruff/format检查，不改动已冻结的算法、测试与登记。

完整独立checker已按同一QA、原5400秒预算启动，tool session28112，child PID3878329/start_ticks615426310，owner PID3876177/start_ticks615422488。owner登记SHA`256968d27a32684dcc7c6a0e8c62bc52489f332d04de0e02061cd00210223b0d`，launch SHA`d1864619b79c6db883f2cdb5c947d2f4e3f68b495dea907cfa55cf7d12c6f178`。运行快照`outputs/prefix_optimization_reserve_execution_checkpoint_20261029/checker_started.json` SHA`881a4f5852676df3efa84d847e5c20726a5cbd6ded480001a30d712a7c54c1f0`确认当时50/4000响应已检查，真实/proc父子身份仍活跃、8项CPU环境一致、38份QA来源未变。完整4000响应、177份完整ray和54份anchor审计尚未全部结束；局部日志不构成完整资格。接续只观察原句柄，不因观察超时重启，也不自动追加新搜索候选。

[后续物理入口与路径调查](native_runtime_path_followup_20261029.md)补充了本任务bowl/plate BDDL、79维已绑定初态、默认OSC JSON及OSMesa静态依赖候选。BDDL/OSC JSON及系统依赖的历史身份、实际动态加载和bootstrap资产仍未闭合；没有建立新运行时或执行物理。只有完整H12解及全部独立核验成功，才可另行登记物理pilot。

实际仍止于原cursor8、task/grasp0，新物理/世界模型调用/fresh均0，总目标尚未完成。
