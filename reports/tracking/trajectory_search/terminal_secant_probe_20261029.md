# 终端单行割线修正的未拟合方向检验

17次响应及全量独立复算已正常关闭，2422项来源核验通过。两个未拟合混合方向均未达到原误差减半门槛，绝对预测精度也不足；本次rank-one修正不采用。完整witness0，实际控制进度仍cursor8、task/grasp0。

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

实现提交为 `68ecd37`，准入及启动记录提交为 `1d90101`。以下为实际完整结果，组件测试与准入不计作控制效果证据。

## 完整执行与独立复核

生产和独立 checker 均完整完成17次调用/16唯一命令，各5100数值内步/5305查询，末尾重复也独立重放；零优化迭代。两个 owner 的实际执行句柄分别52767、39646，均实际exit0；child410736/start616647158与child427951/start616676507的wait0、reap、无信号及无重试均已核验。

| 证据 | 相对仓库路径 | SHA256 |
|---|---|---|
| producer | `outputs/tracking_v5_terminal_secant_probe_20261029/verified.json` | `39aff2ebda4d272b1b03e0cfe60e5bf26a69de10c4870f55afa4280052b98c34` |
| producer owner | `outputs/tracking_v5_terminal_secant_probe_execution_20261029/result.json` | `ffd7763b90f18fd16cbde76bcce5827e102fb948bd118d4439b4788d44ffb380` |
| checker | `outputs/tracking_v5_terminal_secant_probe_check_20261029/verified.json` | `06214022d72f13423c8829344630ed632e4f9a1b6a40125e0aeb2a2e222464ba` |
| checker owner | `outputs/tracking_v5_terminal_secant_probe_check_execution_20261029/result.json` | `cdf6e32104efc5f8344c51c5176eee87a3ebe3a80444dcfdff2938c88d1305a3` |
| check owner实际dry | `outputs/terminal_secant_probe_check_admission_20261029/result.json` | `c6381ae809efb3c8f7f3e52d12ee20c193905a0b52a4f04211b886466faece8d` |
| 完整来源闭合 | `outputs/terminal_secant_probe_final_closure_20261029.json` | `29e3b90d5f0919044abba53102411816f6b78a648a3509a5602a58f64436d779` |
| 全15点保存分析 | `outputs/terminal_secant_probe_analysis_20261029/result.json` | `44d9c06da3824ab17755750ac195ce60152d15e79fd536d17492d9b33a1268e0` |

check owner实际dry句柄3549退出0；闭合句柄79673退出0，2422项来源摘要保持一致，旧75源与新85源QA摘要子集一致；保存分析句柄11817退出0，复核2424项来源且新增响应/查询为0。闭合driver SHA `8569c3edd4b05ffe8f8a4d6f47d00270c494b8fad5ee9c1c9d0306e71d22ecb6`，分析driver SHA `52f7ed6d49e4858e960fb11afde68435a70101fb43b93a9f70fedc61c577b1ce`。

独立检查器核验了几何、两个历史训练点及旧终端行绑定、全部预测代数与严格分类，disagreement为空；完整数值复核最大差异1.27721e-12。另列的JIT/Python rotation goal差异最大5.96046e-8、151项不同，不混入原精度割线计算。初末与历史完整anchor canonical比较exact。

## 外推失败，不采用该修正

全15点及全部五尺度均保留在[完整表](../outputs/terminal_secant_probe_analysis_20261029/analysis.md)，该表SHA `1aa8493f6f872611b7b81440baf8f1e72f08c9149039592dc416246a2c8349c7`。下表按每方向全部五尺度聚合，误差单位为终端速度 slack 的rad/s，按h归一的量不宣称为极限导数。

| 方向 | 最大旧绝对残差 | 最大新绝对残差 | 新/旧 | 最大旧残差/h | 最大新残差/h | 新/旧 | clipping变化点数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v | 2.281520e-6 | 2.281520e-6 | 1.0000 | .002705802 | .002705802 | 1.0000 | 0/5 |
| e+v | 2.362636e-6 | 1.839638e-6 | .7786 | .002729142 | .002206144 | .8084 | 0/5 |
| e−v | 2.247499e-7 | 6.711753e-7 | 2.9863 | .001848057 | .002430811 | 1.3153 | 4/5 |

两个混合方向都未达到两个最大残差分别减半的原门槛；e−v的最大绝对残差增至约2.99倍。纯v负对照的最大预测变化仅2.17e-19，符合构造，但不等于其预测准确。最大新绝对残差2.281520e-6远高于原可用终端裕量3.204288e-9；`extrapolation_passed=False`、`safety_accuracy_passed=False`，故不将本次rank-one修正接入QP。

clipping变化发生在e−v的四个较大h，每点均在period9改变1个flag；最小h=.0001未改变flag。e+v的五点均无clipping变化，仍未达外推门槛，因此不能把整个失败唯一归因于已记录的力矩裁剪变化，也不能据flags一致排除其他数值分支或量化影响。

15个命令虽通过box/rate准入，响应未保持旧达标前缀：v和e+v各五点均prefix5，e−v五点均prefix4；cursor9的最大归一平方误差为1.31583–1.32498，均超原caps，完整witness0。它们仅为未物理执行的数值诊断，不是可替换控制动作。该结果支持下一步从不改变前11周期的末周期平移方向检验终端余量恢复，而非继续用这些跨全窗口方向修补边界预测。

下一项[末周期平移恢复计划](../plans/v0.0.1/tail_translation_reserve_20261029.md)另行预登记，不在本阶段执行QP或物理pilot。新增物理/world-model/fresh/参考消费均0，实际仍cursor8、task/grasp0；原18正确参考/12WAM开发/24最终留出门槛保持，GPU继续禁用，总目标未完成。
