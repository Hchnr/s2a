# 044：固定训练数据的CPU运行时兼容资格

总目标仍沿036；043已实际尝试W1一次，在首标签重建处失败，0参数/成功fit/QR。原495源码、快照、错误、阈值和产物保持不变。首窗口的实际sim解释器逐值复现保存标签；main解释器存在微小float64差异，原False保留。不能从首窗口推广到完整训练。

本项只做有限、只读CPU数值审查，不执行新fit、QR、模型、物理或fresh，也不选择超参数。候选解释器固定为本仓库`outputs/libero_env311/bin/python`，Python3.11.15/NumPy1.26.4/SciPy1.15.3。空CUDA/LD，OMP/OpenBLAS/MKL各8，命令从当前仓库执行，不安装或改动环境。

输入固定为原041 prepared/admitted、完整D0 collection/独立审查和043失败收据。必须先通过原完整`load_context`和同PID/birth正常wait/reap、GPU释放资格，之后才读取既定训练数据。仅沿已验证的process-local源码proof回调修复；其他准入不变。

完整预算：

- 同一150训练锚点，30个来源archive；原production `load_label`与原independent `independent_label`各读取150个窗口，实际最多300次trajectory archive decode。
- 两条原路径的归一化FP32 active10、clip计数逐值/逐字节相同；生产路径原float64 action/state精确门不放宽。
- 同一300请求、每请求35节点、每节点40行，保留全部420000行。原production compact reader和原independent reader各一次，共最多21000次compact archive decode；完整chain、来源、noise seed、mask、sigma及raw C0沿原门检查。
- 两条原point-row路径的全部1290特征、10维clean target及固定全局权重逐字节相同；按固定行序保存独立stream SHA。只重建数据，不计算中心/尺度、Gram、参数解或QR，不将数据审查冒充拟合。
- 最多一个CPU worker，超时1800秒；真实Popen拥有并记录正常wait/reap或失败。失败保留并停止，0自动重试。

审查代码放`outputs/clean_training_runtime_qualification_20261024/source/`，原生产及独立算法源码不变；原495 source/snapshot以及新增plan/config/probe在执行前冻结并前后核SHA。分别记录两条重建路径的计数与摘要，禁止省略不一致节点或按结果缩小分母。

通过只代表该实际CPU运行时能读取并重建本次固定训练数据。实际数值求解、独立QR、D1/W2及任务效果仍未获资格；下一次拟合必须另行登记累计尝试预算、不可覆盖的新输出和独立准入，不恢复原失败supervisor。留出24条和20/24、各任务4/8门槛不变。
