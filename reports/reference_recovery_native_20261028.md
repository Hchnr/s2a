# 实际保持反馈与恢复推进（2026-10-28，CPU）

目标仍为原固定接口下可靠跟踪和新初态任务稳定成功。本次只检验有界的arm保持/恢复机制，不取得原时序、全路径、夹爪协调、WAM或fresh控制资格。GPU继续禁用。

## 实际producer结果

按[预登记计划](../plans/v0.0.1/reference_recovery_native_20261028.md)，重放378周期实际前缀并执行16新反馈周期（step378..393），共394物理周期，400个新内环q/qvel/time样本。每周期先尝试原cursor9..16，均nonlinear_search_no_witness；随后保持cursor8均找到局部模型witness。16周期全部保持、恢复推进0，最后消费仍cursor8，剩余9..39全部未消费。不能把短时保持写成沿原轨迹推进。

实际保持点误差最大位置0.00334615924m、旋转0.04158785168rad，最小内环关节裕量0.05324711229rad；这16周期均在已声明.01m/.05rad/.05rad门槛内。任务与grasp谓词全部false。未进行任何世界模型调用或fresh案例。结果表明继续保持0.8s仍未恢复推进；同状态搜索及真实多周期保持均未解决参考前进问题，下一步不应简单延长等待或任意增加初值重启。

producer owner实际wait=true/returncode0、reaped=true/returncode0。`outputs/tracking_v5_reference_recovery_native_20261028/verified.json` SHA `57c830cecdd7421f61c6896549d777afe8b137779db773654d2572e3827491e1`。这里complete/passed只表示登记协议正常执行，control_qualified明确false。

## 独立复核状态与数值比较口径

首次独立命令重放在第1个前缀端点的bitwise检查失败，34/79分量不同，最大3.0731925e-16；干预尚未开始。原失败产物保留于`outputs/tracking_v5_reference_recovery_native_check_20261028/verified.json`，不能写成独立通过。该次实际消耗1物理周期，owner实际wait/reap代码均1。

v2使用独立新目录，预先登记state、inner q/qvel及历史重建的绝对容差1e-10、相对容差0；记录实际最大state/q/qvel差异。命令与参考相位仍严格一致；物理误差/关节门槛不改变。不导入规划器、不运行solver，只重放保存命令及核验真实状态与评价。v2实际在58个物理周期后失败：endpoint58最大state分量差异0.17327147，不能仅以初始舍入差异解释，更不能继续放宽容差。检查源码确认checker漏掉producer初始及每周期后的current_observation；该函数执行sim.forward()，因此不同传感刷新调度不能假设数值/接触warm-start等价。此处修正复核协议，不改producer动作或历史结果。

v3准备登记刷新修正，但owner错误地尝试迁移未包含在原v5来源清单的test_clean_training_continuation_bundle.py，因KeyError在探针执行前终止，实际物理0。保留supervisor日志，不覆盖。v4只登记原清单实际包含的两份测试修复，刷新时序对齐producer，仍使用预先固定的1e-10/rtol0口径，实际执行中。未确认关闭前仍不能宣称独立通过。

原18条4907步、完整12条WAM开发和final24新初态门槛均保持，尚无总目标完成证据。
