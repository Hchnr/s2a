# 首步保护长反解：加速版本固定快照对照

2026-10-27。总目标未完成。原八周期闭环仿真继续运行，本轮不热改其冻结实现，独立比较相同首步保护优化器使用通用QP和单严格面闭式投影的行为。

实现 `src/s2a/fast_guarded_long_joint_inverse.py` 保留原目标、首步位置/SO(3)保护、动作范围、变化率、60迭代/160响应及候选选择逻辑，只将展开切换为已验证的 `fast_long_joint_preview`。不新增控制器自由度，不改变内环25×2ms。

## 固定快照结果

`scripts/probe_fast_guarded_inverse_v2.py` 读取同一两个已登记快照的当前q/qdot、历史末动作和生成参考；同一原H4 seed保持末条扩展至H8，分别调用原版和加速版。实际进程exit0，前后来源SHA核验完成。

| 案例 | 原版秒 | 加速版秒 | 原版模型代价 | 加速版模型代价 | 响应数（两者） | 终止（两者） |
|---|---:|---:|---:|---:|---:|---|
| raw | 17.13495 | 2.55950 | 1.6695228079601114 | 1.6695228079601434 | 91 | 60迭代上限 |
| terminal058 | 1.39241 | 1.54634 | .03303744622012006 | .03303744622012006 | 83 | 收敛 |

raw 完整H8动作计划最大差9.527e-13、首命令最大差1.144e-13；terminal计划完全相同。两者均通过独立记录计算的首步位置/旋转保护、动作范围和0.25变化率检查。raw约6.69倍反解加速，但仍明显超过0.05s周期；terminal约慢11%，不报告普遍加速。

使用单线程CPU，旧八周期物理实验同时消耗CPU；各设置只测一次，不视为隔离性能基准。此处terminal反解仅是对照计算，实际native入口的无风险旁路仍应保留原命令，不会因标签强制优化。

动作数值一致只在这两个快照成立，不证明所有后续反馈状态的优化器路径或实际控制结果一致，也不证明任务成功。原版raw本身到达迭代上限，不声称任一版本最优。

## 检查与追溯

- CPU `tests/test_single_face_projection.py tests/test_guarded_long_joint_inverse.py tests/test_long_joint_preview.py tests/test_joint_limit_preview.py`：14 passed in5.96s，实际exit0。首步保护的零/非零误差测试分别覆盖两个优化器版本。
- 新反解、探针与更新测试Ruff check、format check通过。
- 产物 `outputs/fast_guarded_inverse_probe_v2_20261027/verified.json`，SHA256 `c4123ef64eea403dedca26459de1ef319bcd831343fa487f13cd72d1eadd9731`。
- 初次临时探针缺少require导入，实际exit1，没有完成比较。保留日志和注册；原脚本精确字节已归档 `outputs/fast_guarded_inverse_probe_20261027/failed_source.py`，原路径/SHA和安全副本映射见同目录 `source_archive.json`。删除了该临时错误入口，修正入口为独立v2文件，没有覆盖已登记失败源或产物。

本轮探针新增物理执行、世界模型调用和fresh案例均0；原完整物理实验仍由句柄31071/原owner管理，已有两个单步分支wait0/reaped，raw40分支实际进程仍存活，尚无完整轨迹结果。下一步先等待并独立验证原实验，再决定加速版本的原生闭环对照，不以模型等价替代控制资格。

原正确参考18/18、WAM开发门槛及最终留出24条要求不变，尚无新候选资格或稳定任务成功证据。
