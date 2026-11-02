# 正确参考执行入口：原基线完整周期复现

原目标保持：固定机器人与控制接口，把世界模型期望轨迹、当前实测状态和因果历史转换为实际 action，并在留出新初态稳定完成任务。本记录仅证明新入口在原 case00 上复现基线，不能证明候选控制资格。

## 输入和方法

`scripts/prepare_tracking_correct_reference.py` 与 `configs/tracking_correct_reference_20261027.json` 已冻结原18条、共4907周期的参考、初态和控制接口。新入口 `scripts/run_tracking_correct_reference.py` 区分 baseline 与 candidate；baseline 保留原四步规划、每8步更新40步参考和 LoadedVelocityControl 夹爪控制。原来源文件未修改。

本次 `scripts/execute_tracking_reference_port.py` 只运行原 case00 的完整164周期，然后运行独立 `scripts/check_tracking_reference_parity.py`，未运行 candidate。均为 CPU/OSMesa，CUDA_VISIBLE_DEVICES 与 LD_LIBRARY_PATH 为空。没有世界模型调用或新初态样本。

## 结果

完整164周期通过逐步动作、控制器位置/旋转目标、原生任务谓词与全部165端点比较，绝对容差1e-8、相对容差0。最大 sim_states 差为8.744746840441252e-10；source_pose/controller_site_pose 最大差2.5432683736781314e-11；开度差7.263079027097774e-13；夹爪缓存逐值一致。

位置RMSE为0.0005333929385835592 m，SO(3) RMSE为0.0009753354238612831 rad；尾部连续成功93周期，保持原 stable40、位置0.01 m与旋转0.05 rad门槛。baseline 和 parity 子进程均实际 wait 返回0并 reap，根执行进程实际返回0。

权威产物：

- `outputs/tracking_correct_reference_20261027/prepared.json`
- `outputs/tracking_correct_reference_port_20261027/registration.json`
- `outputs/tracking_correct_reference_port_20261027/result.json`
- `outputs/tracking_correct_reference_port_20261027/{baseline,parity}/{exit,release}.json`
- `outputs/tracking_correct_reference_20261027/case00_baseline/{result,port_parity}.json`

## 限制与下一步

这是单条原基线的执行入口一致性证据，不是新候选的独立资格，也不是全部18条复现证据。新候选改变规划规则，不能套用要求精确复现旧优化器的检查器。下一步需要独立检查候选的因果输入、动作/变化率边界、原生控制目标、物理端点、任务谓词及原指标，再进入完整18条正确参考回归。原18/18、WAM开发门槛与最终24条新初态门槛全部保持，总目标未完成。
