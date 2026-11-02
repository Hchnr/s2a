# 原正确参考剩余17条回归：执行中

上一轮已独立验证case00。这一轮登记并启动case01..17，每条执行候选后独立重放实际action，候选与回放各4743物理周期。权威入口、预算和限制见 `plans/v0.0.1/tracking_correct_reference_remaining_20261027.md`，注册为 `outputs/tracking_correct_reference_remaining_20261027/registration.json`。

截至本记录，case01候选完成254周期，位置RMSE 0.00022644769607187278 m，旋转RMSE 0.0005827859045638344 rad，尾部连续成功75周期。254周期全部 unchanged_free_feasible，adapter单次最大计算0.010636130347847939 s。独立254周期原生动作回放已完成，执行owner已写入case01/closed.json，两阶段正常关闭。独立原性能门槛通过；此结论仍不包含完整规划输入/图像/标签资格。

执行过程继续，以 `caseXX/closed.json`、stage实际wait/reap和最终result.json为准；没有最终结果即不得声称18条完成。当前新增世界模型请求/新初态均为0，总目标未完成。

## case02回放分歧（追加）

原剩余17条owner已实际退出1，不能继续将其称作运行中。case02候选完整183周期、位置RMSE 0.00026032499335978607 m、旋转RMSE 0.0007750095191630142 rad、尾部85周期成功，但v3独立回放在第52周期以最大状态差0.00778922失败；失败记录与源字节均保留，不能算已独立通过。case03..17尚未启动。

诊断发现原采集每步在current_observation后调用endpoint，endpoint内部再次current_observation并sim.forward；v3回放缺少这次调用，且没有匹配原0/16/56端点渲染。v4检查器对齐这些观测与渲染调用，保留原1e-8状态容差和原性能门槛，复用现有case02候选，单独登记183周期回放，不重新采集。回放路径为 `outputs/tracking_correct_reference_candidate02_replay_v4_20261027/`，检查结果为 `outputs/tracking_correct_reference_check_v4_20261027/case02/verified.json`。这属于验证观测顺序假设，不能在结果出来前断言根因已证实。

## case02恢复核验与剩余15条接续

v4完整183周期回放完成，全部端点状态最大差为0，原性能门槛通过；stage actual_wait0、reaped=true，恢复owner实际退出0。相同初态/命令/容差下，对齐端点观测和渲染调用恢复逐值一致，支持原v3分歧来自不一致的观测调用顺序；没有放宽状态容差或性能门槛。本次失败回放额外消耗52周期，成功重放183周期，须与成功矩阵预算分别记录。

新接续入口 `scripts/execute_tracking_correct_reference_remaining_v4.py` 冻结来源并启动case03..17，候选与独立回放各4306周期，复用case00/01的v3结果与case02的v4结果。新注册 `outputs/tracking_correct_reference_remaining_v4_20261027/registration.json`，不覆盖旧失败owner，不自动重试。原18条候选总预算仍4907周期；含失败诊断的累计物理执行不能伪称只有矩阵预算。当前完整18条结果尚未产生，WAM/fresh未启动，总目标保持。

## case03关闭及首四条机器人输入审查

case03候选与独立回放各170周期，独立端点状态差0，位置RMSE 0.00047644498190366826 m、旋转RMSE 0.0006185007249864774 rad，尾部85周期成功，原性能门槛通过。权威关闭文件为 `outputs/tracking_correct_reference_remaining_v4_20261027/case03/closed.json`。case04执行已开始，后续仍未完成。

新增 `scripts/audit_tracking_robot_inputs.py` 从保存model.xml和每周期原生state重建机器人M/J/q/dq、控制SITE位姿及世界速度，核对日志实际输入；按每条独立零初始化检查coupled与limited历史仅从已完成周期的actual状态和已提交预测递推，并检查完整计划的动作与变化率边界、首命令一致性、当前候选未应用limited创新的显式标记。原始state布局和机器人关节/site均显式核对。case00..03共771个周期输入审查通过，新增物理步0；四个审查stage均实际wait0/reap，审查owner实际退出0。

来源与结果：`outputs/tracking_robot_inputs_execution_20261027/{registration,result}.json`；逐案例 `outputs/tracking_robot_inputs_20261027/caseXX/verified.json`，各自绑定实际result/steps/model/arrays来源。审查owner沿用了单案例模板中的candidate_reused_ref（case02）与末案例verification_ref（case03），这两个字段不是四案例完整清单；完整集合以stages中的case_index和各案例verified.json为准。

该检查证明实际机器人输入与端点绑定、历史递推和计划边界，不独立证明优化器最优性或完全不存在未声明的其他输入；仅覆盖首四条，不外推到未完成案例。图像/标签完整资格及WAM/fresh仍未完成，总目标保持。

case04随后完成候选与独立回放，各170周期；原性能门槛通过。case00..04已有执行/性能检查结果，机器人输入额外审查目前仅case00..03。

## case05及首五条标签/图像审查

case05候选与独立回放各244周期，端点最大状态差0；位置RMSE 0.0001356464862609536 m、旋转RMSE 0.0004755022258233082 rad，尾部91周期成功，原门槛通过。case00..05累计6/18条已有执行与性能结果。case06仍在执行，完整矩阵未关闭。

新增 `scripts/audit_tracking_labels_images.py` 对case00..04的946个物理端点（941个控制周期）完成独立标签公式重算：TCP变换、column-rot6d状态、下一端点局部delta/action、下一开度、20Hz时间间隔及最后端点action无效mask均核对通过；同时验证源位姿由实际grip-site位置与EEF BODY旋转构成。

每案例的0/16/56端点图像均核对原生state哈希、时间和当前源状态/开度，检查无未来RGB、相机角色/翻转、原始256×256 RGB、画布PNG和RGB哈希。case02..04还与v4独立原生回放保存的相机像素逐值一致；case00/01的v3回放没有保存相机，只有原产物绑定/内容审查，不能计为独立像素复现。

权威结果为 `outputs/tracking_labels_images_20261027/case00..04/verified.json`，owner为 `outputs/tracking_labels_images_execution_20261027/{registration,result}.json`，五个stage均实际wait0/reap，owner实际退出0，新增物理/WAM/fresh均0。两个新脚本ruff检查和格式检查通过。此审查不证明渲染过程对所有未保存MuJoCo派生状态无影响；原v3/v4观测顺序诊断保留。未覆盖案例和完整因果输入审查仍需完成，WAM/fresh保持未启动，总目标未完成。

## 首个任务失败：case06

case06（task1_native_demo_46，wine rack）候选334周期及独立回放完整关闭，执行一致，但原性能资格失败：位置RMSE 0.009105850611537366 m、旋转RMSE 0.03984705650282093 rad均在原阈值内，任务仅6周期成功、末尾0周期，terminal/stable均false。334周期全部 unchanged_free_feasible，未进入限位优化。因此不能把失败归因于优化分支，且当前候选已无法达到原18/18资格，即使其他案例均通过也不能开启WAM/fresh。

原基线该案例尾部90周期成功。新旧命令首步差约4.44e-15；第75个零基周期首次超过1e-12（约0.0005294），第149周期首次超过0.01，最终最大差1。原始首步M/J/q/dq/site/twist逐值一致。离线探针 `scripts/diagnose_tracking_reference_roundoff.py` 与固定8线程变体对比参考变换两种乘法顺序，原乘法顺序将首命令最大原差从5.55e-15降到2.36e-16，仍非逐值一致。这只定位到数值运算顺序差异，不能证明完整退化原因或已恢复；没有修改运行中来源、重新采集case06或放宽性能门槛。

诊断结果为 `outputs/tracking_reference_roundoff_20261027/diagnostic.json` 与 `outputs/tracking_reference_roundoff_threads8_20261027/diagnostic.json`，实际探针子进程均退出0；新增物理0。可能的接触敏感性仍待后续按原执行路径的完整对照验证，不能将假设写成已证实因果结论。当前原入口只在case00做过完整基线复现，不能外推所有案例。

case07、08随后均通过原执行/性能检查；截至此追加为9条完成、8条达标、1条性能失败，case09及后续仍待完成。固定矩阵继续以保留各案例对照证据，最终总目标和原WAM/fresh门槛保持。

case04..07另外完成机器人端点输入与历史递推审查，共963周期，新增物理0；四个stage实际wait0/reap，owner实际退出0。覆盖合并至case00..07共1734周期，case06性能失败案例的已记录M/J/q/dq/site/twist和历史残差递推也通过此范围检查，不等同于规划器整体因果性或鲁棒性证明。审查owner为 `outputs/tracking_robot_inputs_cases04_07_20261027/`，沿用模板中的candidate_reused_ref仅指case02，不能视为这四案例清单；以stages和各案例verified.json为准。

case09随后关闭，原性能门槛通过；更新为10条完成、9条达标、case06失败。当前case10运行中。

## 原运算顺序的完整case06基线恢复

更正此前离线探针的环境范围：最初两个探针在.venv中执行，不是原物理运行所用NumPy环境。新 `diagnose_tracking_reference_roundoff_native.py` 在原 `outputs/libero_env311/bin/python`、8线程/空CUDA/LD下复核，移植入口首命令原差4.440892098500626e-15，恢复原乘法顺序后差0，产物 `outputs/tracking_reference_roundoff_native_20261027/diagnostic.json`。此前探针不是原环境的精确复现证据，均保留。

独立版本 `run_tracking_correct_reference_ordered.py` 只改变生成SITE与query anchor的坐标变换运算顺序，保留旧来源。case06 baseline完整334周期已完成，逐周期命令与原基线逐值一致，完整端点/控制目标/任务谓词独立比较通过，使用原1e-8绝对容差/相对0及原性能门槛。此结果验证了基线入口数值语义的恢复，不能证明新候选限位优化或新初态鲁棒性。

首次owner `outputs/tracking_correct_reference_ordered_port06_20261027/` 因新输出父目录缺失，在物理执行前失败，未消耗物理步；第二个owner `outputs/tracking_correct_reference_ordered_port06_v2_20261027/` 创建父目录后复用同一已冻结worker，baseline与parity均实际wait0/reap，owner实际退出0。独立结果 `outputs/tracking_correct_reference_ordered_20261027/case06_baseline/port_parity.json`。本次追加334物理周期，世界模型/新初态0，预算与旧候选矩阵分开。

case10、11原候选随后均通过，当前原矩阵12条完成、11条达标、case06失败；旧矩阵继续到固定18条，新修正尚未运行完整候选18条资格，不替换旧失败结果。下一步应在恢复原运算顺序的入口上独立验证候选，并完成必要完整回归后再判断资格。

## 最终旧版执行/性能结论

全部18条执行/独立回放已关闭，候选4907周期、成功回放4907周期，额外失败回放52周期另列。17/18性能达标，case06失败，未通过原18/18门槛。权威闭合结果 `outputs/tracking_original_matrix_closed_20261027/verified.json`，闭合程序实际退出0；原worker/回放实际wait/reap与失败owner状态均保留。本结论不包含全18条输入/图像附加资格，也不是修正版资格，后续修正版结果见 `reports/tracking_ordered_candidate_20261027.md`。
