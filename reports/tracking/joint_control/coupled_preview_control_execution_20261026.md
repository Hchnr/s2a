# 四步预览固定18条物理回归准入

最新完整结果：248四步预览完成18条/4907周期，18/18任务稳定且全部满足原位置/旋转门槛。成功采集/核验阶段、continuation owner及closure正常退出与来源关闭均完成，生产和独立WAM控制准入复核一致通过。原case00检查器失败及修复证据保留；下文阶段快照为历史记录。尚无新WAM/fresh成绩，总目标未完成。

## 已完成资格

- 连续重建历史18条4907周期、最大独立KKT残差3.5527136788005004e-10，实际退出0。
- bound完整控制日志164周期、164次JSON与SOURCE提交、22项损坏拒绝全部通过，最大独立KKT残差4.4408920985006255e-11，实际退出0。旧精确输入/数值重建混淆导致的失败保留。
- `outputs/coupled_preview_qualification_20261026`完成来源/资格审查，由owned_stage记录真实wait0/reap、未发送信号、子进程同身份退出；该审查不重跑或改写已有数值结果。
- 配置`configs/coupled_preview_control_registered_20261026.json`已创建，SHA为`f1c2d3443f374b694ef904282b201cc3fd77af34474be7cbbf189cb745cd6fe0`。原18 profiles、seed42、4907周期、079夹爪及性能门槛保持。

## 执行结构

`launch_coupled_preview_control.py`监督完整准入入口；入口通过来源冻结、相关CPU测试、18条实际原生CLI dry和独立checker导入后，才启动持久physical owner。owner执行36个collect/verify阶段；另安排独立closure监督，等待owner实际wait/reap后做全量统计和来源关闭。任一失败不自动重试或重新采集。

已完成collect/verify/owner/closure模块实际CPU导入以及相关脚本Ruff/格式检查。源码从启动起冻结，不为修复无关测试或格式问题改动。

最近核实准入worker PID1064328/start_ticks594924005仍同身份存活，父进程1063924；其环境CUDA/LD为空，三项线程数8。物理namespace `outputs/coupled_preview_control_20261026`尚未创建，阶段正在来源核验，不得将准入worker存活称为物理实验已开始。

## 保持的边界

全18条均任务稳定、位置RMSE≤0.01m、SO3 RMSE≤0.05rad，且所有阶段及owner正常退出/来源关闭后，才能进入世界模型开发。最终fresh24仍要求≥20/24稳定、各任务≥4/8。新回归仍是正确参考控制资格，不是WAM成绩。原SQP case09和case14性能失败已确认，剩余结果仍在收尾；原全量CPU测试的7失败/12错误单列，未宣称全套通过。

## 检查器修复与固定配置接续

原准入162项测试、18条原生CLI dry及checker导入通过。case00采集actual wait0，但原checker仍要求旧SQP运行时文件，核验实际退出1，原owner随后正常记录失败并停止；原始失败证据保留。新增registered checker要求真实preview引擎、runtime、executable和base来源，独立修复核验实际wait0，不重采case00、不放宽数值或性能门槛。

`outputs/coupled_preview_verification_repair_20261026/verified.json`确认复用原case00，0新增物理周期。`resume_coupled_preview_control.py`从case01继续剩余17条；配置、控制算法、初态和seed不变。原failed owner与新continuation owner分开保留，最终closure会分别核对case00修复receipt和后续每条receipt。

本次前5条/941周期证明、10个collect/verify实际wait0/reap与同身份进程消失已核对，见`outputs/coupled_preview_prefix_05_20261026.json`。位置RMSE依次0.533393、0.226448、0.259345、0.476452、0.472277mm；SO3 RMSE依次0.000975335、0.000582786、0.000770964、0.000618556、0.000800838rad，均稳定且达标。case02/04相对SQP的旋转误差略增，不能称所有指标均改善。

该快照通过/proc核实continuation owner PID1074770/start_ticks594969768、closure监督PID1074794/start_ticks594969860同身份存活；这是当时状态，不是全量完成证据。原18/18门槛和WAM/fresh门槛保持。

## 后续接入仍需完成

当前preview物理入口在`src/s2a/coupled_pose_preview_control_simulation.py:116`显式仅接受oracle；残留WAM分支不能视作可执行的新模型资格，其旧GPU数字检查也不能沿用。通过完整18条资格后，须按237另行接入允许UUID、raw/terminal058服务身份与当前图像请求，保留连续双历史和独立审查。现阶段不启动模型或消费fresh。
