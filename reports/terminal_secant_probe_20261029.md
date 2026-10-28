# 终端单行割线修正的未拟合方向检验

本项按[执行前计划](../plans/v0.0.1/terminal_secant_probe_20261029.md)检验旧 terminal 行的有限尺度修正是否能外推。前置[全固定网格诊断](directional_derivative_consistency_20261029.md)已完成独立闭合；其小步长差分未显示收敛，三旧方向也接近共线，不能直接据旧训练方向预测改善采用修正。

仅使用旧 stage24 正向 h=.001/.0005 两条已保存响应，以实际 float64 命令位移拟合零截距 rank-one 修正。新方向由旧三个方向和接近 box/rate 边界的有符号约束行构造：seed42 Gaussian 投影得到正交方向 v，再使用 v、e+v、e−v，分别固定五个正步长；15个新点都不参与拟合。全部准入在模型构造前完成，保持命令字节和原±1e-12容差。

固定完整执行为 anchor、15点、末尾 anchor，共17调用/16唯一命令，每次重置同一 actual368 状态和 completed367 latch。每阶段5100数值内步/5305查询，零优化迭代；独立 checker 全量重放含重复的17次，并独立重建几何、拟合、计账与全部预测误差。科学验收失败仍须完整关闭，不据此重试。

两个混合方向分别比较所有五尺度的最大绝对残差和最大按 h 归一残差，二者都须至少减半；同时要求纯正交负对照不受修正影响、15点的 torque clipping flags 与 anchor 相同、初末及历史完整 anchor exact。另列绝对预测误差是否低于原终端可用裕量约3.20e-9；有限样本通过也不代表导数存在或控制资格。

## 已完成的实现与组件检查

- 新 core、独立 audit、producer/checker/owner 及三测试模块已完成。分项 native CPU 检查分别14、19、49项通过；独立只读交叉审查未发现接线阻断。
- 统一 QA 实际执行26模块、655项通过，106.98秒，两个警告来自既有 SLSQP 合成测试。85个来源文件及 QA driver 摘要保持不变；Python3.11.15、NumPy1.26.4、SciPy1.15.3，CUDA/LD显式为空，数值线程数1。
- QA：`outputs/terminal_secant_probe_native_qa_20261029_2wj0oaca/result.json`，SHA `28db556f5af66b66ad915cd2575ed89cd8e830c5d830b9b6564e534cc1c39019`；日志SHA `ee84af46547dc54b91991978f4c4c0d2d576d0e7fea8693075ffee683f07bc62`。执行句柄11307实际exit0，子进程wait/reap，无信号、无重试。
- QA driver：`outputs/terminal_secant_probe_native_qa_driver_20261029.py`，SHA `2d9189f7cfee9b58cc50068cd0babddce46e6c18628f0fcfadf9ce6d08da38b8`。

相关85源 Ruff/format、producer/checker/diagnose owner 的三条实际 CLI dry 均已正常完成；五个准入子进程实际wait0/reap、无信号、无重试，新增响应/查询为0。producer/checker分别检查2395项来源，owner另绑定本次QA和日志。准入 `outputs/terminal_secant_probe_admission_20261029/result.json`，SHA `7f4d7fc88051b003a8b9616949427c19b18dc5b77573cfb15268cddd23250957`，执行句柄47791实际exit0。

实现已提交 `68ecd37`。正式 diagnose owner 已启动，原执行句柄52767保留；尚未取得完整17次响应及独立复核结果。组件测试与准入不计作新控制效果证据。新增物理/world-model/fresh/参考消费均0，实际仍 cursor8、task/grasp0；原18正确参考/12WAM开发/24最终留出门槛保持，GPU继续禁用，总目标未完成。
