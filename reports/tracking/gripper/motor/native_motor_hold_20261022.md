# 032：夹爪motor保持的完整因子结果

新工作区完成8条新增物理分支、8环境/1648步/40canvas；原baseline/latched_pose八条缓存不重采。独立审计完整通过，全部native前缀（含controller/gripper cache）逐位相同，41200次真实500Hz更新及affine ctrl独立重构通过。两owned child正常actualwait0/reap/noSignal、owner工具session7558实际exit0。

| 方法 | 稳定案例 | demo1成功端点 | demo35成功端点 | demo12/13 |
| --- | ---: | ---: | ---: | --- |
| baseline | 2/4 | 1/40 | 38/40 | 40/40，40/40 |
| latched_pose | 3/4 | 40/40 | 39/40 | 40/40，40/40 |
| motor_hold | 2/4 | 1/40 | 0/40 | 40/40，40/40 |
| pose_motor_hold | 3/4 | 40/40 | 0/40 | 40/40，40/40 |

结论：把保持阶段夹爪命令从原-1改0，确实保持了native N端点motor cache/ctrl，但没有提高四例稳定率，demo35反而从38/40或39/40降至0/40。因此不推广motor_hold，也不把motor目标保留等同实测opening或成功状态保持。固定pose锁存的demo1收益保留，不能拼接每例最佳方法。这个因果比较只涉及固定四例native控制，不是WAM成绩，未唯一证明物体接触变化的动力学机制。

## 迁移与验证边界

用户指定share-mixed根后恢复源码/数据并修复本地Python链接。当前env311实际Python3.11.15，旧patch未知；本实验完整native前缀逐位保护已证明所测范围内可比，不认证所有模拟数值或模型CUDA路径等价。历史JSON声明及SHA未改，actual I/O通过显式resolver读取本仓库/安全备份。26个旧Cosmos源码当前缺失，逐SHA归档只证明历史字节，不用于模型运行。

相关路径/源归档/子步/独立checker/clean原语60项CPU测试通过，395文件Ruff check/format通过。完整集合826 passed/1 skipped/2 deselected/1 failed/12 errors保留，失败/错误属于未迁移的历史绝对路径入口；没有称整仓全绿。新CPU clean head只实现数学原语和合成测试，0真实数据fit/NN/CUDA。

质量门槛仍False，原030稳定28/30训练资格False、原历史canvas精确False都保留。最终fresh24尚未执行，真实WAM至少20/24且各任务4/8、末40连续成功仍未达标。

## 完整证据

- `outputs/wam_stability_20261021/native_motor_hold_diagnostic/preparation/prepared.json` SHA `57ce8bb2466d182d7674d5578ca5eb64effc9b429889cbbb14990c8c1445739b`
- `outputs/wam_stability_20261021/native_motor_hold_diagnostic/collection/result.json` SHA `9cd261151ef6ccc85e6a55c995029c658cf3ad05d3a1d82b0ac05c5e83bc53af`
- `outputs/wam_stability_20261021/native_motor_hold_diagnostic/review/verified.json` SHA `676ba0e941f382bea48f5593ebc0f8469d35f497cd304f8e47b2d7c11b53582f`
- `outputs/wam_stability_20261021/native_motor_hold_diagnostic/orchestration/result.json` SHA `1b6156a94bd9850f056014eca412d645a9f7d4bd80e6c7e5b20f98bdf4140f92`

下一步优先重新登记统一的训练数据合同：区分完整native示范与人为追加的释放/稳定阶段，固定全部30来源、anchor时刻、追加时长与质量分母，不篡改旧结果、不筛28条追认资格。pose锁存下demo35原先仅最初1端点失败，支持检查统一释放/稳定阶段的定义，但尚无延长运行结果，不能外推其一定成功。新数据合格或新研究用途明确后，才能把已实现clean原语接入真实35步sampler、固定噪声/两轮轨迹聚合和最终统一任务评测；不再以反复teacher指标代替任务收益。
