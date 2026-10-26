# 四步预览固定18条物理回归准入

248配置已登记，正在执行来源冻结与准入；当前没有新候选物理成绩，原总目标未完成。

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
