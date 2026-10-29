# 恢复原运算顺序候选：完整18条执行中

## 已验证结果

先前case06基线334步逐值复现后，现已完成修正版case06候选334步及独立原生动作回放334步。位置RMSE 0.008463290078903724 m、SO(3)RMSE 0.038313634589607654 rad，尾部90周期成功，原stable40/.01m/.05rad门槛通过；独立回放全部端点状态最大差0。两个stage均actual_wait0、reaped=true，case06/closed.json已写入。该结果恢复了移植入口的此案例表现，不能作为限位优化收益证据。

机器人输入审查也对该案例334周期完成M/J/q/dq/site/twist与已保存原生端点绑定、历史创新按完成周期递推、计划边界与变化率检查，独立owner实际退出0，新增物理0。该审查范围不包括优化器全局最优性或完全不存在未声明输入。

## 完整矩阵与来源

修正版完整矩阵由 `scripts/execute_tracking_correct_reference_ordered.py` 执行，原18条、候选4907周期及独立回放4907周期，先case06再其余17条，种子/接口/原门槛不变，无WAM/fresh。只恢复生成SITE和query anchor变换的矩阵乘法顺序，旧源码与旧失败结果不覆盖。完整计划见 `plans/v0.0.1/tracking_ordered_candidate_20261027.md`。

- 注册：`outputs/tracking_correct_reference_ordered_execution_20261027/registration.json`
- 已关闭案例：该目录 `case06/closed.json` 与两个stage的exit/release
- 独立结果：`outputs/tracking_correct_reference_ordered_check_20261027/case06/verified.json`
- 输入审查：`outputs/tracking_ordered_robot_inputs_20261027/case06/verified.json`
- 输入审查owner：`outputs/tracking_ordered_robot_inputs06_execution_20261027/{registration,result}.json`

其余17条仍需完成；图像/标签适配审查脚本仅已准备，不计作修正版已执行证据。原失败版本矩阵另行继续，目前case00..13共14条完成、13条达标、case06失败，不与新版本结果混算。完整18/18、输入/图像/来源与生命周期审查完成前，不进入WAM/fresh。总目标未完成。

## case00与首两条图像/输入资格补充

case00候选164周期及独立回放关闭，原性能门槛通过。结合case06，当前修正版2/18条已完成执行/性能检查，均通过；其余未完成，不能宣布完整资格。

case06的335端点标签/mask与0/16/56端点相机审查通过，三组相机与独立物理回放逐像素一致。owner为 `outputs/tracking_ordered_labels_images06_execution_20261027/`，实际退出0，stage实际wait0/reap。结果为 `outputs/tracking_ordered_labels_images_20261027/case06/verified.json`。

case00另完成165端点标签/mask和对应三组图像逐像素回放比较，以及164周期机器人实际输入/历史递推审查；owner `outputs/tracking_ordered_audit00_execution_20261027/` 实际退出0，两个stage均实际wait0/reap，各结果为 `outputs/tracking_ordered_{robot_inputs,labels_images}_20261027/case00/verified.json`。这些审查新增物理周期0，WAM/fresh0，仍仅覆盖所列案例，整体因果输入与所有后续案例审查不外推。

原失败版本矩阵已完成case00..14，共15条/14条达标/1条任务失败；case15候选600周期已完成（全部 unchanged_free_feasible，adapter均值0.008315487722866237 s），独立回放待关闭，不计作已完成整条。此为保存的正确参考执行，不是新WAM闭环成功。新旧两个owner继续，各自source和结果不混用。

case01随后也完成候选与独立回放，原性能门槛通过；修正版更新为3/18条关闭且全部达标。图像/输入补充审查目前仅case00与06完整完成，不把case01性能通过视为所有附加审查通过。

## case02关闭及附加审查批次

case02候选与独立回放已关闭且通过原性能门槛，修正版当前case06、00、01、02共4/18条执行/性能通过。四条分别334/164/254/183周期，全部 unchanged_free_feasible，不包含限位优化触发证据。因此即使正确参考回归继续通过，也不能将其解释为限位优化分支已获验证，后续自己的当前图像WAM闭环仍是必要证据。

新增显式批次入口 `scripts/execute_tracking_ordered_audits.py`，用case列表和唯一run-id冻结已关闭案例，拒绝覆盖产物和自动重试。实际批次cases01_02完成四个stage：两条机器人物理输入/历史递推与两条标签/mask/相机审查；每stage实际wait0/reap，owner实际退出0，原生相机逐像素回放比较通过，新增物理/WAM/fresh0。结果 `outputs/tracking_ordered_audits_cases01_02_20261027/{registration,result}.json`，逐案例结果仍位于 `outputs/tracking_ordered_{robot_inputs,labels_images}_20261027/caseXX/verified.json`。与先前case00/06证据合并，首四条额外审查覆盖935控制周期/939端点；不外推到尚未关闭案例。新入口ruff检查和格式检查通过。

旧矩阵case15独立回放已关闭，更新为16条完成/15条达标/case06失败；不与修正版计数混合。新旧回归均继续；完整18条资格和总目标仍未完成。

case03随后通过候选与独立回放，修正版更新为5/18条执行/性能通过；额外输入/图像审查仍仅上述四条。旧版本case16也已关闭并达标，旧矩阵17条完成、16条通过、case06失败，尚余case17。以上为实际关闭凭据计数，两个执行owner仍运行，不预判最终结果。

## 旧矩阵完整关闭与修正版审查进展

旧移植入口全部18条已完成：原4907周期候选执行＋4907周期成功独立回放，17条性能达标、case06任务稳定性失败，完整18/18资格失败。额外52周期失败回放单独保留，初始化失败与观察顺序修复的各owner结果未抹去。`scripts/close_tracking_original_matrix.py` 核对18条预算、逐case实际wait0/reap和原门槛结果，闭合产物 `outputs/tracking_original_matrix_closed_20261027/verified.json`，SHA256 `bb7051877da0e71a1b9b5fe3d0285acd7101b791d2c7a201c2c115c61062c49e`；闭合检查实际退出0。这里只关闭旧版执行/性能矩阵，不宣称旧版全18图像/输入资格完成，也不混作修正版成功。

修正版case03、04输入/历史与标签/图像审查批次完成，四个stage均实际wait0/reap，owner实际退出0；相机与独立回放逐像素一致。批次为 `outputs/tracking_ordered_audits_cases03_04_20261027/{registration,result}.json`。修正版附加审查现覆盖case00..04与06共1275控制周期/1281端点；没有新增物理/WAM/fresh调用，仍不外推至其他案例。

修正版case05随后通过候选与独立回放，更新为7/18条执行/性能均通过（case00..06），其余继续。case05附加审查尚未执行，完整18条和总目标未完成，WAM/fresh未启动。

## case05附加资格及case07性能关闭

case05的输入/历史与标签/图像审查已完成，两stage实际wait0/reap，owner实际退出0。批次 `outputs/tracking_ordered_audits_case05_20261027/{registration,result}.json`；相机与独立回放逐像素一致。修正版case00..06的1519控制周期/1526端点均已补齐此范围附加审查，新增物理/WAM/fresh0；完整规划器因果性/模型最优性并未由这些检查证明。

case07随后完成候选与独立回放，原性能门槛通过，修正版执行/性能更新为8/18条全部通过。case07附加审查尚待执行，其他案例继续，总目标未完成。

后续入口核对：`src/s2a/serialized_preview_wam_simulation.py:595` 仍将实际arm命令设为原solve_preview结果，尚未调用tracking_margin_native；`scripts/collect_serialized_preview_wam_development.py:167` 导入的是这一原WAM执行器。因此现成WAM启动器不能直接作为新候选闭环证据，须另行实现并验证候选接入、实际命令提交和历史observe顺序，同时保持HTTP/当前图像/服务身份/预算及无expert接口；在完整正确参考资格和接入审查完成前不启动模型。以上为现有代码检查，不是新WAM接入已实现或已执行。

case08随后也通过执行与独立回放，修正版更新为9/18条执行/性能通过，当前case09执行中。附加审查仍仅case00..06完成。

## case07/08附加审查、case09性能与后续提交组件

case07/08的四个输入/历史、标签/相机stage完成且实际wait0/reap，批次owner实际退出0；产物 `outputs/tracking_ordered_audits_cases07_08_20261027/{registration,result}.json`，相机逐像素回放一致。附加审查覆盖修正版case00..08共1916控制周期/1925端点。case09随后完成候选/独立回放，原门槛通过，修正版10/18条执行/性能通过；其余继续，不预判18/18。

新增可复用 `src/s2a/tracking_preview_transaction.py`，供后续WAM入口明确区分：准备计划、核对实际发送命令后提交预测、收到已完成周期的实际端点后更新历史。计划只在真实首周期observe后成为下一次计算初值；准备失败后需显式reset；独立案例reset清除历史；返回日志的外部修改不能改变提交预测。本组件不调用env.step，不改变优化器或物理门槛，尚未接入WAM执行器，也未修改正在运行的冻结源码。

`tests/test_tracking_preview_transaction.py` 使用真实AdaptivePlanHistory检查命令错配拒绝、未commit不能observe、未observe不能重规划、实测位移/速度残差、计划延迟发布、reset隔离、返回对象修改隔离及失败状态恢复。实际CPU执行3 passed（3.15s），两文件ruff与格式检查通过。测试仅证明提交/历史协议，不是native refiner全链或WAM闭环证据。完整物理接入和新初态成功仍待验证。

## case10及提交组件的物理接入验证

修正版case10完成候选/独立回放，原门槛通过，当前11/18条执行/性能达标。case09/10的输入/历史及标签/图像批次也关闭，四stage实际wait0/reap，owner实际退出0；权威结果 `outputs/tracking_ordered_audits_cases09_10_20261027/{registration,result}.json`，相机逐像素回放一致。附加审查已覆盖case00..10共2251控制周期/2262端点，新增物理/WAM/fresh0。其余继续，完整资格未完成。

独立验证入口 `scripts/run_tracking_transaction.py` 在同一原运算顺序、参考、物理接口与夹爪设置下，用TrackingPreviewTransaction替换直接prepare/commit/observe调用，并检查实际全部周期已observe后方可正常关闭；旧冻结入口不修改。`scripts/execute_tracking_transaction06.py` 登记334物理周期的case06候选和原基线全周期动作/端点/目标/任务谓词比较；来源与结果写 `outputs/tracking_transaction06_execution_20261027/` 和 `outputs/tracking_transaction_20261027/case06_candidate/`。此为提交组件的独立物理接入检查，尚在执行，不是WAM接入、不是新18条资格，也不替代修正版原矩阵。

提交组件case06物理验证随后完成：完整334周期执行成功，所有物理端点、源/控制SITE位姿、开度和夹爪缓存与原基线最大差均0，逐步命令/目标与任务谓词比较通过，原性能门槛保持（尾部90周期成功）。candidate/parity两stage均实际wait0/reap，owner实际退出0。原生refiner确实通过新组件调用，但此案例全部unchanged_free_feasible；尚不覆盖优化触发或WAM协议。追加334物理周期、WAM/fresh0，不混入修正版18条矩阵预算。

case11随后完成候选与独立回放并通过原门槛，修正版更新为12/18条执行/性能通过；附加审查目前仍仅case00..10，case11及后续待补齐。执行owner持续运行，完整资格未达成。

## case11附加审查、case12性能与优化分支提交验证

case11输入/历史及标签/图像审查完成，两stage实际wait0/reap，owner实际退出0，批次 `outputs/tracking_ordered_audits_case11_20261027/`。修正版附加审查覆盖case00..11共2532控制周期/2544端点，新增物理0。case12随后通过候选与独立回放，修正版更新为13/18条执行/性能均通过；其他案例仍在执行。

新增优化触发场景的提交组件验证：`scripts/run_tracking_transaction_feedback.py` 保留原固定参考、初态与控制算法，仅以TrackingPreviewTransaction替换直接prepare/commit/observe；明确仅支持tracking分支。由 `scripts/execute_tracking_transaction_feedback.py` 登记原raw/task0/seed42的72周期前缀＋56周期分支，共128周期CPU/OSMesa物理执行，WAM/fresh0。来源/执行owner `outputs/tracking_transaction_feedback_execution_20261027/`，结果 `outputs/tracking_transaction_feedback_20261027/case00_tracking/`。

完成后 `check_tracking_transaction_feedback.py` 将与已有独立验证的tracking56周期结果比较实际命令、全部物理state、原生目标、TCP误差、历史创新、路由与任务谓词，原绝对1e-8容差保持，要求实际存在优化调用。此比较用于提交组件的行为一致性，不是新的任务资格、性能提升或WAM闭环证据。原参考案例任务失败与优化开销结论不因组件接入而消除。目前该验证仍在运行，不将已启动计为通过。

case13、14随后均通过原候选执行/独立回放门槛，修正版更新为15/18条（case00..14）均达标，正在执行case15。case12/13附加输入/标签/相机批次也完成：四stage实际wait0/reap、owner实际退出0，相机逐像素回放一致，批次 `outputs/tracking_ordered_audits_cases12_13_20261027/`。附加审查当前覆盖case00..13共2932控制周期/2946端点，无新增物理/WAM/fresh。

提交组件的优化触发分支worker仍为原活进程，尚未关闭；已观察其实际CPU运行，不能把结果文件暂缺解释为任务终止或触发重试。保留同一进程与owner，等待实际结束及parity检查。

## 提交组件的实际优化分支一致性关闭

原72周期前缀＋56周期分支已完成，新增128物理周期、WAM/fresh0。56周期中54周期实际进入优化，其余2周期保留seed。与原先独立验证的tracking轨迹相比，全部实际物理状态最大差0，命令、原生目标、TCP误差、历史创新、路由及任务谓词逐周期比较通过（仍为原绝对1e-8容差，相对0）。candidate/parity均实际wait0/reap，owner实际退出0。权威结果 `outputs/tracking_transaction_feedback_20261027/case00_tracking/transaction_parity.json`，owner `outputs/tracking_transaction_feedback_execution_20261027/result.json`。

这验证了新提交组件在原限位优化触发场景中的行为保持，不增加任务成功证据；该旧固定参考分支的任务失败、关节越界与计算开销结论保持，不能作为新WAM闭环性能。新组件在独立物理入口验证完成，但尚未接入受HTTP/当前图像身份约束的WAM执行器。

case14附加输入/标签/相机审查完成，两stage实际wait0/reap，owner实际退出0，相机与独立回放逐像素一致；批次 `outputs/tracking_ordered_audits_case14_20261027/`。附加审查覆盖case00..14共3107控制周期/3122端点，无新增物理。

修正版case15随后完成候选与独立回放，原门槛通过，更新为16/18条执行/性能均通过；case15附加审查已单独启动，尚不计为完成。case16、17及完整资格关闭仍待执行；WAM/fresh保持未启动，总目标未完成。

case15附加输入/标签/相机审查随后完成，两stage实际wait0/reap，owner实际退出0；原生回放像素逐值一致，批次 `outputs/tracking_ordered_audits_case15_20261027/`。附加审查更新为case00..15共3707控制周期/3723端点，新增物理0；case16候选执行仍为同一活进程，完整18条资格未完成。

## 完整正确参考资格关闭（最终追加）

修正版18条全部完成原4907周期候选及4907周期独立原生动作回放，18/18均达到stable40、位置RMSE≤.01 m、SO(3)RMSE≤.05 rad。根执行进程实际退出0。case16/17的最后四项附加审查也完成；现在全部18条、4907控制周期/4925端点均有机器人输入/历史递推、标签/mask与0/16/56端点图像审查，各案例相机与原生回放像素逐值一致。

`scripts/close_tracking_ordered_matrix.py` 逐条检查36个候选/回放阶段实际wait0/reap，36个附加审查阶段实际wait0/reap，原case预算、独立任务及位置/旋转门槛、actual CLI与审查类型/案例身份、结果引用一致、全部登记来源SHA未改变。汇总由 `scripts/execute_tracking_ordered_closure.py` 持有并实际wait0/reap，汇总owner实际退出0。权威资格 `outputs/tracking_ordered_matrix_closed_20261027/verified.json`，SHA256 `d547e318d22cf494e2ea013a16d3f96a20847b4544834e85b989022b1407b1c6`；汇总owner `outputs/tracking_ordered_closure_execution_20261027/result.json`。

重要范围：全部4907周期均 unchanged_free_feasible，因此这套正控证明新入口保持正确参考控制资格，不能证明限位优化收益。优化分支提交行为另由54周期实际优化调用的固定56周期比较证明保持，但该分支任务失败记录仍保留。整体最优性、实时可行性、WAM当前图像闭环性能和最终24条新初态成功均未由本资格证明。

本次新增WAM/fresh0。下一步是把已验证的提交组件接入独立WAM执行器，保留当前图像/因果状态、无expert输入、服务身份和预算协议，完成接入验证后按原12条开发门槛运行；最终24条新初态门槛保持。总目标仍未完成，不因正确参考18/18而宣布实现世界模型闭环目标。
