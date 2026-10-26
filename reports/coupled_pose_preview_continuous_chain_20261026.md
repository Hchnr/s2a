# 四步预览连续历史回放与仿真入口准备

新运行时/显式reset候选开始全18条连续记录回放。`scripts/audit_coupled_preview_continuous_chain.py`每条仅初始化一次SOURCE转换器、独立SOURCE回放器、150历史、coupled历史和原生goal；随后候选命令、goal与两套innovation逐周期衔接，各条之间隔离。

端点来自原232归档raw MuJoCo状态，独立模型每次恢复并mj_forward核验几何、M/J、twist。每个新提案经过JSON往返、独立预览ODE/KKT、生成参考twist、首命令连续预测及原生goal检查。最终命令提交SOURCE后，独立标量回放核验名义位移/速度；下一归档端点用于验证两套历史计算。更新后的缓存用于下一候选。

这是“候选缓存沿记录端点推进”的接口资格，不是候选命令产生了归档物理状态。没有mj_step或WAM请求；任务成功仍属于原232而非本候选。

## 独立仿真入口准备

新`src/s2a/coupled_pose_preview_control_simulation.py`接入运行时预览，保留最终命令reader、原生goal检查、SOURCE提交和双历史更新。配置要求247显式reset策略及完整18/4907连续链证明，日志采用pose_preview专有字段并分开连续首命令预测与切向计划。

新`coupled_preview_arm_log_replay.py`与`coupled_preview_acceptance_checker.py`从独立raw状态及生成chunk重建参考窗口，验证预览计划、连续预测、原生goal、SOURCE及两套历史，并保留原机械/标签/图像检查。三个新文件Ruff/格式检查及CPU实际导入通过；尚未完成完整日志损坏测试、入口dry准入、登记或实际物理启动，不能宣称仿真入口已获得执行资格。

## 原SQP结果与完整CPU测试

原SQP已见前11条独立证明，其中case09任务稳定但位置RMSE0.01055009842m、SO3 RMSE0.05585927201rad，超过原0.01m/0.05rad。因此本版本18/18门槛已明确失败，剩余回归仍按冻结配置完成；不进入世界模型开发或fresh24。

提交9753cdd后的CPU全套测试结束：2656通过、7失败、12错误、1跳过、1排除，耗时2987.90秒。已回填中英文PR说明。旧路径迁移、现有真实实验产物与启动审查相关测试尚待处理，不修改被原回归指纹绑定的文件或删除真实证据来制造全绿。

## 连续链完整结束与日志检查修复

18条4907周期连续链全部结束，实际退出0，最大独立KKT残差3.5527136788005004e-10，来源SHA复核通过。完整证明为`outputs/coupled_pose_preview_continuous_chain_20261026/verified.json`。这是连续重建缓存资格，仍不代表新命令产生了归档端点。

首版完整控制日志检查实际退出1，checked_rows=1；第二步失败在独立重算innovation与生产输入的逐位相等断言，最大差8.1185e-15。独立ODE数值与生产矩阵指数不应要求逐位一致。旧失败保存在`outputs/coupled_preview_control_log_20261026/failure.json`，代码和证据保留。

新`coupled_preview_arm_log_replay_bound.py`先按原数值门槛独立核验日志history，再要求预览计划精确绑定该日志history；不放宽KKT、状态误差或独立历史核验门槛。对应admit和acceptance checker为独立bound版本，Ruff/格式检查通过，完整164周期及22种错误注入重验正在执行。新物理回归仍未启动。

原SQP后续快照：前15条13条综合达标。case11稳定且达标，修复232原wine失败；case14仅成功1步、末尾不稳定，位置RMSE0.01434648721m/SO3 RMSE0.09418886504rad，也不达标。case09超限仍在。因此原18/18资格失败确定，不能只报告case11改善；剩余3条继续完成全量对照。

bound日志重验完成：实际退出0，164个完整JSON控制周期、164次SOURCE提交与22项错误注入全部通过，独立控制器cache/双历史/原生goal检查完整。证明为`outputs/coupled_preview_control_log_bound_20261026/independent_verified.json`。下一步进行来源与实际退出资格汇总、固定18条配置登记和真实CLI准入，再启动独立物理回归；当前仍无新候选物理成绩。
