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

随后唯一producer已启动，原tool session17677、child PID3827991/start_ticks615340066、owner PID3825995。运行快照`outputs/prefix_optimization_reserve_execution_checkpoint_20261029/producer_started.json` SHA`00ecb0dfd94c0fb977a817f8bb8fb0bd23cb630b16caa0b9e606bfe88294767b`核对当时300响应、真实/proc父子身份与存活、8项CPU环境匹配、38份QA来源不变。owner登记2269份来源，登记SHA`0bde8f36d4f2a29c91aeb0bbcafc33a23938a670985cc25fe84e3cf5eaff5af2`，launch SHA`cdd56527287b73e9defe109a1c943bfcfce56545ef769dfb4a0d74c00b4f81a5`。快照保留主机实际UTC时间，不是完成或wait/reap凭据；接续观察原句柄，不能因观察超时重启。

[后续物理入口与路径调查](native_runtime_path_followup_20261029.md)保留了原manifest与安全配置/库/源码的只读比对、最小恢复协议及尚未闭合的bootstrap/动态库依赖；没有建立新运行时或执行物理。只有完整H12解及全部独立核验成功，才可另行登记物理pilot。

实际仍止于原cursor8、task/grasp0，新物理/世界模型调用/fresh均0，总目标尚未完成。
