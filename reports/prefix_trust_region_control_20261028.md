# 同warm H12信赖域对照

本轮按[执行前登记](../plans/v0.0.1/prefix_trust_region_control_20261028.md)比较冻结前缀求解器与有限信赖域候选。两组使用同一actual368状态、completed367控制历史及chunk46原时序H12参考；共同初始命令取自已完成独立核验的固定回缩响应`cases.prefix.entries[55]`。不使用GPU。

## 当前状态与证据范围

两组数值搜索均已actual wait0/reap关闭，producer均报告prefix9、完整witness0；完整独立检查尚未通过。候选第一次checker在加载旧外部mesh时失败，actual wait1/reap1、0响应/查询/内步，失败保留，正在按[资产迁移修复登记](../plans/v0.0.1/prefix_trust_asset_check_repair_20261029.md)接续。实际仍止于原cursor8，task/grasp0；本轮新增物理、世界模型调用和fresh案例均为0。原18条正确参考、12条WAM开发及最终24条留出任务门槛保持，总目标未完成。

候选保留原完整响应、九组约束审计、位置/旋转caps和全轨迹目标；增加每阶段最多5迭代、固定三半径、完整响应验收后重新定中心及逐请求计费记录。普通下降可以在prefix不增长时获接纳；完整H12 witness仍必须通过全部原门槛。此对照比较整套控制流，不能单独归因于半径。

每臂最多4000唯一完整响应、60累计迭代和60阶段；每条响应300数值内步/312查询，另计每臂一次current查询。两臂不重置跨阶段预算，不免费评价warm或差分探针。独立检查重新构造全部唯一响应并核对compact trace；只有initial/selected/closest保存完整轨迹，因此仅对这三种记录比较完整q/v/FK/torque数组，候选接受anchor另逐条比较完整审计。

## 实现与准入

- 求解器：`src/s2a/prefix_trust_region_plan.py`，冻结基线仍为`prefix_preserving_trajectory_plan.py`。
- 来源准入：`src/s2a/prefix_trust_control_evidence.py`，绑定上轮96响应及完整独立检查的成功wait/reap、来源SHA和共同warm bytes。
- 生产、独立检查及进程管理分别为`scripts/probe_tracking_v5_prefix_trust_control.py`、`scripts/check_tracking_v5_prefix_trust_control.py`和`scripts/execute_tracking_v5_prefix_trust_control.py`。
- 三个组件测试分别覆盖求解器验收/差分/预算、独立分类/计费/排序及实际执行入口的重置与计费。首次同版本native QA为59通过/1失败：真实SLSQP合成例在60迭代预算内未达到测试所预期的完整witness。失败凭据保留于`outputs/prefix_trust_native_qa_20261029_0rg1svty/result.json`（SHA`7f6fff11aaf009453d3d1927ad790529535a27584ebd752e05532129674a90e7`），actual wait退出1、9份来源未变。原测试源码按原QA SHA精确恢复并保存在同目录`failed_solver_test_source.py`（SHA`5ffa7d855d6b6acde68b74af7aede240b20a434aa316fdd551c7a0e08bb26a2b`），没有重跑该失败QA。

该困难合成例有显式合法零误差命令，但固定5迭代短阶段的callback反复失去首点prefix，只能接受较小合法差分探针；845响应/60迭代后仍无witness。生产算法、登记预算、原caps均未修改。测试现在保留已知可行但有限搜索失败的证据及诚实预算终止；另一个具有严格余量的合成例由真实SLSQP找到完整witness。不能从后者宣称原H12搜索必定成功。

最终统一QA：`outputs/prefix_trust_native_qa_20261029_4oymj468/result.json`，SHA`b6835c1c5816225c3c000d9402ec057c417541287ef758c0582bd7a21436c5ee`；61 passed/5.66s，actual wait退出0，9份来源在QA前后相同。Python3.11.15/NumPy1.26.4/SciPy1.15.3，CPU及空CUDA/LD；2条SciPy bounds-clipping warning来自上述保留困难例，不是正式搜索结果。

准入凭据：同目录`admission.json`，SHA`393c4bd94a1814deb4c973a7fa18cc988fee4526ceb6c664d5074ae62e552234`。Ruff检查/格式、两臂producer/checker metadata dry、两臂owner diagnose dry共8条命令actual wait退出0，全部9份源码与QA一致，新增真实响应0。共同warm SHA`16bdfbb266ef13d2588c8116e13511d93df936d72ae8a57d19df0b01ed3acee0`；1967个来源引用、8函数独立数学AST核对通过。

实现提交：`6b411c2`为solver/测试，`9df9930`为来源、生产、独立检查和进程管理。两臂将使用上述同一QA；源、测试和登记从真实搜索启动后冻结。

## 结果与后续

两臂producer由各自持有Popen句柄的owner执行。启动快照`outputs/prefix_trust_execution_checkpoint_20261029/producer_started.json`（SHA`d2c895f028c42eb72941f354b3a4135a41bb4cf89b19049a8233ffd4a679ea00`）记录了当次实际PID/start_ticks/PPID存活核对及两臂各600条已报告响应；这是运行中快照，不是完成或wait证据。baseline tool session为29743、candidate为87474。继续工作时必须复查这些原句柄/进程，不能因观察超时重启。

两臂owner登记的1969份来源SHA映射和native QA引用完全相同。输出位于`outputs/tracking_v5_prefix_trust_{baseline,candidate}_20261028`，owner为相应`_execution_20261028`；日志为owner目录的`capture/command.log`。上述运行快照保留；两producer现均完成、实际退出0且无signal/retry。

以下为producer保存结果，**尚待完整独立复核**：

| 指标 | 冻结baseline | 信赖域candidate |
| --- | ---: | ---: |
| 唯一完整响应 | 4000 | 3148 |
| 数值内步/查询 | 1200000 / 1248001 | 944400 / 982177 |
| 累计迭代/阶段 | 53 / 1 | 45 / 9 |
| 连续达标端点 | 9 | 9 |
| 完整H12 witness | 0 | 0 |
| selected trace index | 1254 | 2262 |
| cursor9位置误差 | 11.46476035 mm | 11.48335059 mm |
| cursor9旋转误差 | .05738804160 rad | .05742253674 rad |
| 停止原因 | 4000响应预算耗尽 | 三个固定半径均无可接纳下降 |

原门槛仍为10mm/.05rad，两臂均未达到。candidate用更少响应但没有更好精度或新达标端点，不据此提升为控制方案。producer及owner SHA完整列于新修复计划；只读汇总`outputs/prefix_trust_presentation_20261029/producer_summary.json` SHA`e290a3e90e85c7c728af6de7ee596bab5318f31a82f31595ed07541703fd9445`绑定两成功owner和来源，不新增响应。producer合计7148响应/2144400内步/2230178查询。

候选的6次新anchor全部来自差分探针，没有callback被接纳。cursor9最大归一化平方误差从1.324948101801降至1.318939090352，仍超过1；最后三个半径的callback都失去原prefix，主要在cursor4旋转、cursor5/8位置。最终selected #2262不同于最后anchor #1876，仅改善后续全轨迹loss，cursor9误差相同。完整selected审计的最紧非线性项为cursor8/joint index4的braking_upper，余量5.3219e-7rad；这不能反推仅有标量safety_min的其他callback具体失败关节。

candidate独立checker的失败结果SHA`cc4993e6e0e2772ad4e2172910cf7f93b634b1cb94f61753ae18499f588cea47`，失败owner SHA`335ea5abf56ea58b41d653b8d5ba7953ef22fc20f2e55ec985c433a6ccdf9a84`。错误发生在MjModel加载前期，0响应/0数值内步/0查询，不能写成已完成一条重构。baseline checker仅metadata dry通过，未实际执行。原XML与1969来源未变，但这些指纹没有覆盖当时外部资产加载字节；已找到96个安全副本全部匹配20261025已存资产SHA，修复只迁移路径并保留完整模型，不裁剪几何、改物理参数或重跑producer。

下一步先通过新的隔离加载器完成全部7148响应独立复算，再据保存ledger分析未收敛步的线性预测与实际前缀误差，避免盲目扩大搜索。当前无完整witness，不进入物理pilot；独立检查本身也不能授予实际控制或留出任务资格。

## 完整资产修复与v2准入（2026-10-29）

修复实现已提交`4484bc8`。新增完整资产快照、扩展来源准入、仅两处可逆AST转换的v2 checker及其5400s进程owner；原已执行source、测试、计划和失败结果未修改。新增文件与原实现共18份来源绑定同一native QA：`outputs/prefix_trust_asset_native_qa_20261029_fsj1vuyh/result.json` SHA`0430be2247cfee440062d7b236d374686f4bfad093286719a9925fb62f4451a5`，156 passed/19.58s、actual wait0、来源不变。两条SciPy warning仍来自保留的困难合成例。

一次性准入记录：`outputs/prefix_trust_asset_admission_20261029/result.json` SHA`92475d48eecd99d4a9a704421795cab519c9e31553d162eb57278756453d3358`。16个Python文件的Ruff检查与格式、snapshot dry、唯一完整编译、两臂checker与owner dry共8条命令均actual wait0，无信号和自动重试；18份来源及QA指纹一致。driver本身、逐命令日志/退出码/PID均保存，新响应、查询、数值内步、物理步、世界模型调用和fresh案例均0。

完整资产快照为`outputs/prefix_trust_asset_snapshot_20261029/manifest.json` SHA`c9eaaca963a82f504ed38e67df847fc54c144d030f660be52f6439db0b7b3566`：99个file引用、96资产逐项匹配历史SHA，原XML副本与路径逆替换均字节一致。所有file引用显式命名，无include；资产文件编号不会改变MuJoCo默认名称。安全外部源只读、旧路径未恢复。

唯一编译结果为同目录`compile.json` SHA`a0bbbbdfe379feeb4f9c2b83c7f1e66e9aa95d884bcfb19fd5131439f8dbe5ed`。完整MuJoCo2.3.7模型具有nq41/nv37/nu9、38 bodies、17 joints、240 geoms、83 meshes、18 textures、29 sites，Euler2ms；仅MjModel创建，未执行forward或step。Python3.11.15/NumPy1.26.4，空CUDA/LD及单线程CPU环境固定。此编译准入不补称producer加载时逐资产SHA，也不代表完整数值检查通过。

v2两臂输出独占新目录，仍须分别独立重构4000和3148响应并实际wait/reap关闭。当前准入已完成、正式v2尚未启动，数值资格待定；原完整witness0和实际cursor8/task0/grasp0保持。
