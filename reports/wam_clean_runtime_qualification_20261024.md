# 044完整CPU数值兼容审查 — 2026-10-24

**实际sim运行时完整数据兼容审查通过；没有执行新的fit或QR。** 同一150训练窗口、300请求、10500节点和420000行全部保留，原生产路径与原独立检查路径的归一化标签、特征、clean target及全局权重逐字节相同。原043失败和main运行时首窗口False保持原记录，总目标仍未达到。

## 范围与实际结果

按[044固定协议](../plans/v0.0.1/044_clean_training_runtime_qualification_20261024.md)，使用本仓库实际sim解释器：Python3.11.15、NumPy1.26.4、SciPy1.15.3，空CUDA/LD、OMP/OpenBLAS/MKL各8。先通过原041完整D0元数据与正常退出资格，再读取既定训练数据；不加载模型、环境或fresh数据，不修改旧源码、标签或门槛。

| 检查 | 实际完成 |
| --- | --- |
| 生产标签重建 | 150/150；原float64 action/state精确门全部通过 |
| 独立标签重建 | 150/150；归一化FP32 active10及clip计数与生产路径一致 |
| 来源archive | 30个；两路径共300次trajectory archive读取 |
| 采集请求/节点 | 每路径300/10500，noise seed/顺序/chain/mask/sigma沿原门核验 |
| 完整数据行 | 每路径420000行，特征1290列、clean target10列、固定全局权重，全部逐字节相同 |
| compact archive读取 | 两路径共21000次，与实际np.load计数相符 |
| 拟合/QR/模型/物理/fresh | 全部0；solve调用另有显式阻断，attempted_solve_calls=0 |

原production `load_label`、`point_rows`、compact reader和原independent `independent_label`、`independent_point_rows`、compact reader保持原文件SHA。新审查只重建数据、比较与计算SHA，不计算均值/尺度、Gram、参数解或QR。已有独立label路径的1e-12几何容差保持原值；生产路径的精确门未放宽，最后全部归一化与训练行字节相同。

两路径的诊断行流SHA分别相同：features `1d6d753c08c82ca065c0c82356d2d35fcfd492dd24e194986483b5776ab458ac`，targets `0fab9a0b81189869fcc34bebac3add5e1629752167583585044be50ba58122dd`，weights `86c5363ef222e1bf1b2b83f91256777ec3d989486495d0f42ea38d8ce7a56d00`。这些是本次诊断的连续原始行字节摘要，不冒充原fit的replay摘要或参数。

## 生命周期与证据

CPU worker PID1957310由真实Popen持有，105.54秒后实际wait0、同birth reap、无signal；当前对应birth已不存在。498份源码及498份快照在执行前后逐SHA一致，保留全部150份label和300份request摘要；消费文件25161份。源码proof只校正原执行verifier回调一次，退出恢复/源守卫通过。

- [实际完整结果](../outputs/clean_training_runtime_qualification_20261024/result.json)，SHA `c2ebcee63bc79d63f4fa3fbd52cda1326ad7c57bc098b44e69a9b058eb63a562`。
- [真实owner](../outputs/clean_training_runtime_qualification_20261024/owner/result.json)，SHA `70541388baabca1ea535554e660d2d06b2d77e816b5923bbc43419dfd8090500`。
- [实际exit](../outputs/clean_training_runtime_qualification_20261024/owner/exit.json)，SHA `25066a133e064122a556829adb987272ab7d124c74b7201a27480380ebcae01c`。
- [同birth release](../outputs/clean_training_runtime_qualification_20261024/owner/release.json)，SHA `f4a0404ecda08021afb3714dbc1d9876b7cbd52d7d4b6fb4778a57cf5bb16c4b`。

本审查未重建原D0省略的vision/flat/head数值，继续继承其原online及037＋039证据边界。它证明指定运行时可完整重建固定D0训练数据，不证明参数解、W1/W2收益、开发成功、fresh成绩或实时/真机能力。

下一步为另行固定的新W1生产拟合及原独立QR：使用新输出目录并保留043失败；累计记原043一次失败尝试，新协议最多再一次W1尝试，不能按成功fit为0抹去失败。实际求解、独立重算和后续D1仍须各自准入及真实正常退出；当前044不授权这些调用。原20/24、每任务4/8最终验收门不变。
